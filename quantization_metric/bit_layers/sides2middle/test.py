import json
import os

data = [4] * 32
for i in range(10, 29):
    data[i] = 2

id = 2
left = 10
right = 28
while sum(data) < 128:
    data[left] = 4
    with open(f"configure_{id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    id += 1
    left += 1
    print(data)
    data[right] = 4
    with open(f"configure_{id}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    right -= 1
    id += 1
    print(data)
