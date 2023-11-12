# -*- coding: utf-8 -*-
import json

max_len = 20000

with open('./data/test.json', 'r', encoding='utf-8') as f_in:
    with open('data_out.ldjson', 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)

            if len(data['sentence2'].encode('utf-8')) > max_len:
                data['sentence2'] = data['sentence2'][:max_len]
                print(data['sentence2'])

            json_str = json.dumps(data, ensure_ascii=False)
            f_out.write(json_str + '\n')

print('处理完成!')