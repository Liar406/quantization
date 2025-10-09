import os
import json
import numpy as np

paths = os.listdir('./')
# print(paths)
scores = {}
for path in paths:
    if 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            score = []
            if len(data['results']) < 20:
                continue
            tasks = ['hellaswag','rte','mmlu']
            for task in tasks:
                score.append(round(100 * data['results'][task]['acc,none'], 2))
            
            scores[path] = score

for key in scores:
    print(key, scores[key])

