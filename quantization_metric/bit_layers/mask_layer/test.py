import json

file_path = "original_configure.json"


data = [4] * 32
for i in range(0, 32):
    data[i] = 2
    print(data)
    
    with open(f"configure_{i}-4to2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    data[i] = 4
    
