from datasets import load_dataset

# 替换成你要用的任务名称
datasets_to_download = [
    "winogrande", 
    "hellaswag", 
    "arc", 
    "lambada_openai", 
    "piqa", 
    "trivia_qa"
]

for name in datasets_to_download:
    print(f"Downloading dataset: {name}")
    try:
        # 加载并缓存数据集（默认缓存路径 ~/.cache/huggingface/datasets）
        load_dataset(name)
    except Exception as e:
        print(f"Failed to download {name}: {e}")
