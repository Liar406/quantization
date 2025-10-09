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
                print(task, result['acc,none'])


