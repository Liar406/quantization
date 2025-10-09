import os
import json
import numpy as np
import re

paths = os.listdir('./')
# print(paths)
scores = {}
for path in paths:
    if 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            score = 0.0
            # for task, result in data['results'].items():
            #     score += result['acc,none']
            # score /= len(data['results'])
            score = data['results']['hellaswag']['acc,none']

            match = re.search(r'configure_(\d+)', path)
            if match:
                number = match.group(1)
            scores[f"layer {number} to 2bit"] = round(score*100, 2)

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for path, score in sorted_scores:
    print(f"{path}: {score}")

