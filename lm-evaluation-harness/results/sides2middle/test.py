import os
import json
import numpy as np

paths = os.listdir('./')
configure_paths = os.listdir("/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/bit_layers/sides2middle")
configure_dir = "/mnt/bn/life-mllm/users/cxr/quantization/quantization_metric/bit_layers/sides2middle"
tasks = ['piqa', 'winogrande', 'arc_easy', 'arc_challenge', 'hellaswag', 'boolq']
scores = {}
for configure_path in configure_paths:
    if 'json' not in configure_path:
        continue
    with open(os.path.join(configure_dir, configure_path), 'r', encoding='utf-8') as f:
        configure_data = json.load(f)
    configure_name = configure_path.split('.')[0]
    
    configure_name = configure_name + '_'
    print(configure_name)
    for path in paths:
        if configure_name in path:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                score = []
                # for task, result in data['results'].items():
                #     score += result['acc,none']
                # score /= len(data['results'])
                for task in tasks:
                    score.append(round(100 * data['results'][task]['acc,none'], 2))
                
                scores[','.join([str(x) for x in configure_data])] = score

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=False)
for path, score in sorted_scores:
    print(f"{path}: {score}")

