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
            score = 0.0
            for task, result in data['results'].items():
                score += result['acc,none']
            score /= len(data['results'])
            scores[path] = round(score, 4)

print(scores)
indices = np.argsort(scores)[::-1]  # 从大到小的索引

indices = np.argsort(
[
0.3567,
0.3559,
0.3548,
0.3541,
0.355,
0.361,
0.3504,
0.3542,
0.3553,
0.3523,
0.3512,
0.3557,
0.3545,
0.3527,
0.3488,
0.3533,
0.3517,
0.3522,
0.3521,
]
)[::-1]
print(indices+10) 