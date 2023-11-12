import json


file = 'train-ocr.json'
label_mapping = {'0':'non-rumor','1':'rumor','2':'unverified'}

with open(file,encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    label = item['label']   
    
    # 倒数第3个字符就是[]里的数字
    number = label[-3]
    # number = label
    
    # 替换label为提取的数字
    if number in label_mapping:
        temp = label_mapping[number]
        item['label'] = temp
    else:
        print(f'number {number} not defined in label_mapping')
    

# 写回JSON文件    
with open(file, 'w',encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)