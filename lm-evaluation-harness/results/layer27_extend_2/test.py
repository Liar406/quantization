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
            for task, result in data['results'].items():
                score += result['acc,none']
            score /= len(data['results'])
            # score = data['results']['hellaswag']['acc,none']
            scores[path] = score
            

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print(sorted_scores)
match = re.search(r'configure_(\d+)', sorted_scores[0][0])
if match:
    number = match.group(1)
print(number)