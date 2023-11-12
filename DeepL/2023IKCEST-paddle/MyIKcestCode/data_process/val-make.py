import process_data as pd
from tqdm import tqdm
import json
# sample = {'label': label, 'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap}
# 假设 data 是一个包含100个数据的列表
data = pd.val_dataset()
bar =tqdm(data,total=len(data))

# 初始化一个空列表，用于存储每个数据的 caption 和 qCap  和 label字段
output_data_list = []

label_mapping = {0:'non-rumor',1:'rumor',2:'unverified'}
i=0
for item in bar:
    # i+=1
    # 提取 caption 和 qCap
    caption = item['caption']
    qCap = item['qCap']
    label = item['label']
    qImg = item['qImg']
    imgs = item['imgs']
    # input = 'qCap:'+str(qCap) + 'caption:'+str(caption)

    # 转换label为含义标签
    label = label_mapping.get(label,str(label))

    # 将数据保存为字典
    output_data = {
        'label': label, 'caption': caption,'imgs': imgs,  'qImg': qImg, 'qCap': qCap
    }
    output_data_list.append(output_data)
    # if(i>10):
    #     break

# 将列表保存为 JSON 文件
with open('val-ocr.json', 'w', encoding='utf-8') as f:
    json.dump(output_data_list, f, ensure_ascii=False)