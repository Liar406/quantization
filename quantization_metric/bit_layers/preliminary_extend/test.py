import json
import os
import re
# file_path = "original_configure.json"

# Top5Kurtosis = [0, 1, 31, 30, 22]
# Top5HeavyTailAlphaValues = [0, 11, 10, 1, 8]
# Lowest5Kurtosis = [25, 27, 23, 28, 18]
# Lowest5HeavyTailAlphaValues = [21, 30, 20, 22, 24]

# data = [8] * 32
# for i in Lowest5HeavyTailAlphaValues:
#     data[i] = 4

# data[30] = 8

# print(data)
# with open(f"preliminary_10.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)
def change_filename(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        old_num = int(match.group(1))
        new_num = old_num + 10
        new_filename = re.sub(r'\d+', str(new_num), filename)
        return new_filename  # 输出：preliminary_15.json
    else:
        return filename

for file in os.listdir('.'):
    if file.endswith('.json'):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = [dt//2 for dt in data]
            new_filename = change_filename(file)
            with open(new_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            


