import json

data = [4] * 32
for i in [23,24,27,28,29]:
    data[i] = 2
print(data)
with open(f"configure_0.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    
