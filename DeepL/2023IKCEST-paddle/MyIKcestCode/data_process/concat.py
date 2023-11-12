import json


file = 'test-ocr.json'
# label_mapping = {'0':'non-rumor','1':'rumor','2':'unverified'}
# output_data = {
#         'label': label, 'caption': caption,'imgs': imgs,  'qImg': qImg, 'qCap': qCap
#     }

with open(file,encoding='utf-8') as f:
    data = json.load(f)
output_data_list=[]
id=0
for item in data:
    # label = item['label']
    qImg = item['qImg']
    imgs = item['imgs']
    caption = item['caption']
    qCap = item['qCap']
    sentence1 = str(qImg) + str(qCap)
    sentence2 = str(imgs) + str(caption)

    # 将数据保存为字典
    # output_data = {
    #     "sentence1":sentence1,
    #     "sentence2":sentence2,
    #     "label":label
    # }
    # test
    output_data = {"id":id,"sentence1":sentence1,"sentence2":sentence2}
    id+=1
    output_data_list.append(output_data)

# 将列表保存为 JSON 文件
with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(output_data_list, f, ensure_ascii=False)
    

