import json

file_path = "original_configure.json"

for i in range(5, 31):
    data = [4] * 32
    data[27] = 2
    data[22] = 2
    data[28] = 2

    if data[i] == 2:
        continue

    data[i] = 2
    print(data)
    with open(f"configure_{i}-4to2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
