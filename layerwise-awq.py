# -*- encoding:utf-8 -*-
@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    calib_data="pileval",  # data for calibration
    skip_first: int = 0,  # number of initial layers to keep in full precision
    first_n: int = 0,  # number of initial layers to apply first quant
    w_bit_first: int | None = None,
    w_bit_rest: int | None = None,
    # --- mixed-precision strategy --------------------------------------------------
    strategy: str = "layer",  # "layer" (default): original solve layer-by-layer; "auto": structured mixed-precision
    m_auto: int | None = None,  # number of high-bit layers when strategy == "auto"; defaults to 25% of L
    hi_bit: int = 4,
    lo_bit: int = 2,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
    k_energy: int = 32,
    metrics_csv: str | None = None,  # optional explicit path to metrics CSV (delta_ppl,erank_diff,topk_energy_diff)
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    if "bigcode" in str(model.__class__).lower():
        # otherwise attention_mask will always be on cpu.
        model.transformer.bias = model.transformer.bias.to("cuda")

    layers = get_blocks(model)

    samples = get_calib_dataset(
        data=calib_data, tokenizer=enc, n_samples=n_samples, block_size=seqlen
    )
    samples = torch.cat(samples, dim=0)

    inps = []
    layer_kwargs = {}

    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    # patch layer 0 to catch input and kwargs
    layers[0] = Catcher(layers[0])
    try:
        if model.__class__.__name__ == "LlavaLlamaModel":
            model.llm(samples.to(next(model.parameters()).device))
        elif model.__class__.__name__ == "InternVL3":
            model.language_model(samples.to(next(model.parameters()).device))
        else:
            model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")

    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
    }

    # ---------------------------------------------------------------------------
    # Determine per-layer bit-widths according to the requested *strategy*
    # ---------------------------------------------------------------------------

    if strategy.lower() == "auto":
        # -------------------------------------------------------------------
        # Use qpRANK pre-computed diagnostics to decide per-layer precision.
        # Users may place the JSON files (drop_layer_ppl.json, diff_erank_values.json)
        # under the project root (default path) or supply env QPRANK_METRICS_DIR.
        # -------------------------------------------------------------------

        import json, os, math, csv

        def _load_metrics_from_csv(csv_path: str):
            """Return delta_ppl, erank_diff, topk_energy_diff lists from a csv file."""
            delta_ppl, erank, topk = [], [], []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    delta_ppl.append(float(row.get("delta_ppl", 0)))
                    erank.append(abs(float(row.get("erank_diff", 0))))
                    topk_val = row.get("topk_energy_diff")
                    if topk_val is not None and topk_val != "":
                        topk.append(float(topk_val))
            # Ensure all same length
            assert len(delta_ppl) == len(erank), "CSV length mismatch"
            if len(topk) != len(delta_ppl):
                topk = [0.0] * len(delta_ppl)
            return delta_ppl, erank, topk

        delta_ppl: List[float]
        delta_r: List[float]
        delta_e: List[float]

        # Priority 1: explicit CSV path
        if metrics_csv is not None and os.path.isfile(metrics_csv):
            delta_ppl, delta_r, delta_e = _load_metrics_from_csv(metrics_csv)
        else:
            # Priority 2: auto-detect inside QPRANK directory structure
            base_dir = os.getenv("QPRANK_METRICS_DIR", os.path.expanduser("~/qpRANK/src"))

            # Derive a crude model identifier from config
            cfg_name = getattr(model, "config", None)
            model_id = (
                getattr(cfg_name, "_name_or_path", "model").replace("/", "_")
                if cfg_name is not None
                else "model"
            )

            # Traverse to find a metrics_long.csv matching pattern
            candidate_csv = None
            for root, dirs, files in os.walk(base_dir):
                if "metrics_long.csv" in files and model_id in root:
                    candidate_csv = os.path.join(root, "metrics_long.csv")
                    break

            if candidate_csv and os.path.isfile(candidate_csv):
                delta_ppl, delta_r, delta_e = _load_metrics_from_csv(candidate_csv)
            else:
                # Fallback to old JSON files (legacy)
                metrics_dir = os.getenv("QPRANK_METRICS_DIR", os.path.expanduser("~/qpRANK"))
                ppl_path = os.path.join(metrics_dir, "drop_layer_ppl.json")
                erank_path = os.path.join(metrics_dir, "diff_erank_values.json")

                if not (os.path.isfile(ppl_path) and os.path.isfile(erank_path)):
                    raise FileNotFoundError(
                        "Cannot locate per-layer metric files for auto strategy. Provide metrics_csv path or set QPRANK_METRICS_DIR appropriately."
                    )

                delta_ppl = json.load(open(ppl_path, "r"))["delta_ppl"]
                erank_json = json.load(open(erank_path, "r"))

                keys = [k for k in ("q", "k", "v") if k in erank_json]
                delta_r = [
                    sum(erank_json[k][i] for k in keys) / len(keys)
                    for i in range(len(delta_ppl))
                ]

                delta_e = erank_json.get("topk_energy_diff", [0.0] * len(delta_ppl))
        #! layer 的数量
        L_total = len(delta_ppl)

        # Normalise
        def _norm(arr):
            m = max(arr) if max(arr) > 0 else 1.0
            return [x / m for x in arr]

        ppl_hat = _norm(delta_ppl)
        r_hat = _norm(delta_r)
        e_hat = _norm(delta_e)

        scores = [
            alpha * ppl_hat[i] + beta * r_hat[i] + gamma * e_hat[i]
            for i in range(L_total)
        ]

        #! 1/4 的 layer 
        if m_auto is None:
            m_auto = max(1, L_total // 4)

        idx_sorted = sorted(range(L_total), key=lambda i: scores[i], reverse=True)
        #! 前 1/4 的 layer 用 high bit, 其他的用 low bit
        hi_set = set(idx_sorted[:m_auto])
        
        #! 每个 layer 的 bit 数量的分配
        #! 我们也是在这边修改成得到我们的 layer 分配就好了
        bits_per_layer = [hi_bit if i in hi_set else lo_bit for i in range(L_total)]

        # ---- verbose print & log ----
        try:
            import logging
            _logger = logging.getLogger(__name__)
        except ImportError:
            _logger = None

        print("[AUTO] Per-layer bit-width allocation (index:bit):")
        mapping_str = ", ".join(f"{idx}:{bits_per_layer[idx]}b" for idx in range(L_total))
        print(mapping_str)

        if _logger is not None:
            _logger.info("AUTO bit-width allocation: " + mapping_str)

        print(f"[AUTO] Layers @ {hi_bit}-bit: {sorted(list(hi_set))}")
        print(f"[AUTO] Layers @ {lo_bit}-bit: {sorted([i for i in range(L_total) if i not in hi_set])}")

        if _logger is not None:
            _logger.info(f"Layers_{hi_bit}bit: {sorted(list(hi_set))}")
            _logger.info(f"Layers_{lo_bit}bit: {[i for i in range(L_total) if i not in hi_set]}")

    else:
        # Fallback to original scheme (uniform or head/tail mixed precision).
        bits_per_layer = None  # will be decided on the fly as before

    # solve layer by layer
    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        # print(f"Layer {i} of {len(layers)-1}")
        layer = layers[i]

        # Flag: whether to apply quantization to this layer
        #! 他们也指定了超参数从第几层开始量化
        quantize_this = i >= skip_first

        # Determine bit-width for this layer
        if strategy.lower() == "auto" and bits_per_layer is not None:
            current_w_bit = bits_per_layer[i]
            if i == 0:
                # show a brief summary once for user awareness
                print(
                    f"[AUTO] Using structured mixed-precision: {sum(b == hi_bit for b in bits_per_layer)} layers @ {hi_bit}-bit, {sum(b == lo_bit for b in bits_per_layer)} layers @ {lo_bit}-bit."
                )
        else:
            # original rule-based selection
            if i < first_n:
                current_w_bit = w_bit_first if w_bit_first is not None else w_bit
                print(
                    f"Layer {i} is quantizing with {current_w_bit} bits. (when this sentence isnt printed, it is quantizing with {w_bit_rest} bits)"
                )
            else:
                current_w_bit = w_bit_rest if w_bit_rest is not None else w_bit

        
        #! 从这边往后就和原来的代码一样
        layer = layer.cuda()
        named_linears = get_named_linears(layer)
        
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        inps = layer(inps, **layer_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        torch.cuda.empty_cache()

        if (
            auto_scale
        ):  # if it applies, we should also modify the input_feat with scales
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=current_w_bit, #! 改成 current_w_bit 就可以
                q_config=q_config,
                input_feat=input_feat,
            )
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        # Clear GPU memory
        torch.cuda.empty_cache()
        # for line in torch.cuda.memory_summary().splitlines():
        #     if "Allocated" in line:
        #         print(line)

        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=current_w_bit, #! 改成 current_w_bit 就可以
                q_config=q_config,
                input_feat=input_feat,
            )
            apply_clip(layer, clip_list)
            # append prefix to make names global
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model, layer) + "."
            )

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
        # for line in torch.cuda.memory_summary().splitlines():
        #     if "Allocated" in line:
        #         print(line)

    return awq_results