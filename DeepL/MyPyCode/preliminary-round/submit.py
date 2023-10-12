import csv
import json
label_mapping = {0:'non-rumor',1:'rumor',2:'unverified'}
with open('./data/test_results-64-6.json') as f:
    data = json.load(f)

with open('result.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id','label'])

    for i,label in enumerate(data['label']):
        # 转换label为含义标签
        label = label_mapping.get(label, str(label))
        writer.writerow([i,label])