import json
import os
import argparse
parser = argparse.ArgumentParser(description="parser")
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--layer_index", type=str, required=True, default="")
parser.add_argument("--bits", type=str, required=True)
args = parser.parse_args()
print(args)
layer_index = [int(idx) for idx in args.layer_index.split(",") if idx]
save_path = os.path.join(args.save_dir, f"bits_{args.bits}")
os.makedirs(save_path, exist_ok=True)

for i in range(3, 30):
    data = [4] * 32
    for layer in layer_index:
        data[layer] = 2
    if data[i] == 2:
        continue
    data[i] = 2
    print(data)
    with open(os.path.join(args.save_dir, f"bits_{args.bits}", f"configure_{i}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
