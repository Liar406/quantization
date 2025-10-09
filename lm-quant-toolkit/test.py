import pandas as pd
import json
import os
import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument(
        "--model",
        type=str,
        default="Llama-3.1-70B-Instruct",
    )
args = parser.parse_args()

file_path =f'/mnt/bn/life-mllm/users/cxr/quantization/lm-quant-toolkit/tmp/kurtosis-dump/{args.model}/kurtosis-models.csv'
df = pd.read_csv(file_path)
kurtosis_means = df.groupby("layer")["kurtosis"].mean().tolist()

with open(f"kurtosis_means/kurtosis_means-{args.model}.json", "w") as f:
    json.dump(kurtosis_means, f, indent=4)

