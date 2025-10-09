import json
import os
import numpy as np
import re
import argparse
parser = argparse.ArgumentParser(description="parser")
parser.add_argument("--result_dir", type=str, required=True)
parser.add_argument("--bits", type=int, required=True)
parser.add_argument("--configure_dir", type=str, required=True)
args = parser.parse_args()

# print(args)
target_dir = os.path.join(args.result_dir, f"bits_{args.bits-1}")
if os.path.exists(target_dir):
    paths = os.listdir(target_dir)
    scores = {}
    for path_name in paths:
        if 'json' in path_name:
            path = os.path.join(target_dir, path_name)
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                score = 0.0
                for task, result in data['results'].items():
                    score += result['acc,none']
                score /= len(data['results'])
                # score = data['results']['hellaswag']['acc,none']
                scores[path] = score
                
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    match = re.search(r'configure_(\d+)', sorted_scores[0][0])
    if match:
        number = match.group(1)
    configure_path = os.path.join(args.configure_dir, f"bits_{args.bits-1}", f"configure_{number}.json")
    with open(configure_path, "r") as f:
        config = json.load(f)
    indices = [i for i, val in enumerate(config) if val == 2]
    print(",".join(str(x) for x in indices))
else:
    print("")
