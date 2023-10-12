import json

# 读取包含中文的JSON文件
with open('train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理数据

# 将数据写入LDJSON文件
with open('train.ldjson', 'w', encoding='utf-8') as file:
    for obj in data:
        json.dump(obj, file, ensure_ascii=False)
        file.write('\n')

# 读取包含中文的JSON文件
with open('test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理数据

# 将数据写入LDJSON文件
with open('test.json', 'w', encoding='utf-8') as file:
    for obj in data:
        json.dump(obj, file, ensure_ascii=False)
        file.write('\n')

# 读取包含中文的JSON文件
with open('dev.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 处理数据

# 将数据写入LDJSON文件
with open('dev.ldjson', 'w', encoding='utf-8') as file:
    for obj in data:
        json.dump(obj, file, ensure_ascii=False)
        file.write('\n')