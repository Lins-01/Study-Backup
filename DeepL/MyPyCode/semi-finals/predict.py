import data_all
import json
import os
import utils
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
import paddle.nn.functional as F
import functools
import numpy as np
import paddle

from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from tqdm import tqdm
import pandas as pd
import sys


def evaluate(model, result_csv):
    results = []
    # 切换model模型为评估模式，关闭dropout等随机因素
    model.eval()
    count = 0
    for batch in test_dataloader:
        count += 1
        cap_batch, img_batch, qCap_batch, qImg_batch = batch
        logits = model(qCap=qCap_batch, qImg=qImg_batch, caps=cap_batch, imgs=img_batch)
        # 预测分类
        probs = F.softmax(logits, axis=-1)
        label = paddle.argmax(probs, axis=1).numpy()
        results += label.tolist()
        print(count)
    test_str = {0: "non-rumor", 1: "rumor", 2: "unverified"}
    print(results[:5])
    # 输出结果
    # id/label
    id_list = range(len(results))
    frame = pd.DataFrame({'id': id_list, 'label': results})
    frame = frame['label'].map(test_str)
    frame.to_csv(result_csv, index=False, sep=',')


# 数据预处理函数，利用分词器将文本转化为整数序列
def preprocess_function(examples, tokenizer, max_seq_length, is_test=False):
    encoded_inputs = tokenizer(
        text=examples['sentence1'],
        text_pair=examples.get('sentence2', None),  # 可能不存在 sentence2
        max_seq_len=max_seq_length,
        padding='max_length',
        truncation=True
    )

    if not is_test:
        encoded_inputs['labels'] = examples['label']

    return encoded_inputs


if __name__ == '__main__':
    # test_csv = sys.argv[1]  # 测试集路径
    # result_csv = sys.argv[2]  # 结果文件路径

    # 上传的时候，注释掉这三个，解开上面的两个，和下面test_path的那个
    test_csv = r"E:/Document/CodeSpace/Data_set/Paddle2023IKCEST/queries_dataset_merge/dataset_items_test.json"  # 测试集路径
    result_csv = 'result.csv'  # 结果文件路径
    #
    data_items_test = json.load(open(test_csv, encoding="utf-8"))
    # # 获取当前运行文件的路径，再获取他的目录路径、__file__指当前运行文件
    # # test_path = os.path.dirname(os.path.realpath(__file__))
    #
    # # 处理测试集,处理完生成test.json文件
    test_path = r'E:/Document/CodeSpace/Data_set/Paddle2023IKCEST/queries_dataset_merge'
    # # test_path = os.path.dirname(os.path.realpath(__file__))  # 仿照sub的例子，这里应该是test文件夹所在目录
    data_all.make_test(data_items_test, test_path)

    test_ds = load_dataset(utils.read, data_path='test-ocr-1.json', is_test=True, lazy=False)
    # for i in range(len(test_ds)):
    #     print(test_ds[i])
    #     print(test_ds[i]['sentence1'])

    model_path = r'D:\Software\TEMP-model\checkpoint-4000'
    # model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
    # print('test_path ==========  : ', test_path)
    # print('model_path ==========  : ', model_path)
    num_classes = 3
    # 这里直接传入checkpoint文件夹目录，就可以加载微调后的模型进行预测了
    model = ErnieForSequenceClassification.from_pretrained(model_path, num_classes=num_classes)
    tokenizer = ErnieTokenizer.from_pretrained(model_path)

    # 测试集数据预处理，利用分词器将文本转化为整数序列

    trans_func_test = functools.partial(preprocess_function, tokenizer=tokenizer, max_seq_length=512, is_test=True)
    test_ds_trans = test_ds.map(trans_func_test)

    # 进行采样组batch
    collate_fn_test = DataCollatorWithPadding(tokenizer)
    test_batch_sampler = BatchSampler(test_ds_trans, batch_size=1, shuffle=False)
    test_data_loader = DataLoader(dataset=test_ds_trans, batch_sampler=test_batch_sampler, collate_fn=collate_fn_test)

    # Adam优化器、交叉熵损失函数、accuracy评价指标
    optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    # 模型预测分类结果

    label_map = {0: 'non-rumor', 1: 'rumor', 2: 'unverified'}
    results = []
    model.eval()
    bar = tqdm(test_data_loader)
    for batch in bar:
        # input_ids: 表示输入文本的token ID。 分词后的词语在词表中对应的编号
        # token_type_ids: 表示token所属的句子（Transformer类预训练模型支持单句以及句对输入）。->eg:属于sentence1为0，sentence1为1
        # print('input_ids shape:',batch['input_ids'].shape)
        # print('token_type_ids shape:',batch['token_type_ids'].shape)
        input_ids, token_type_ids = batch['input_ids'], batch['token_type_ids']
        logits = model(batch['input_ids'], batch['token_type_ids'])
        probs = F.softmax(logits, axis=-1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        preds = [label_map[i] for i in idx]
        results.extend(preds)

    print('results:', results[:3])
    # 输出结果
    # id/label
    id_list = range(len(results))
    frame = pd.DataFrame({'id': id_list, 'label': results})
    frame.to_csv(result_csv, index=False, sep=',')
