import json

file_path = "original_configure.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(data[10])
for i in range(10, 29):
    data[i] = 4
    print(data)
    
    with open(f"configure_{i}-2to4.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    data[i] = 2
