# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from paddlenlp.utils.log import logger


# 预处理函数
def preprocess_function(examples, tokenizer, max_length, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.
    构建模型对于分类任务的输入，通过拼接和加上特殊tokens
    """
    result = tokenizer(examples["text"], max_length=max_length, truncation=True)
    if not is_test:
        # 不是测试时，将样本的labels转化为整数
        result["labels"] = np.array([examples["label"]], dtype="int64")
    return result


# 将连续出现的符号变为单个
# eg:你好！！！-》你好！
def get_solo(text):
    # 列表推导式，用于创建新的列表
    # dules为['。。', '，，', '！！', ';;', '；；']
    duels = [x + x for x in list('。，!;；')]
    # 如需增加标点符号在list中添加即可.
    for d in duels:
        while d in text:
            # 出现'。。'就替换为单个的'。'
            text = text.replace(d, d[0])
    return text


def read_local_dataset(path, label2id=None, is_test=False):
    """
    Read dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if is_test:  # 测试时
                # line.strip()去掉行首尾的空白字符，处理后的结果返回
                sentence = line.strip()
                # yield返回一个包含字典的生成器
                # 和return差不多，就是函数不结束。
                # 再调用next()就再次执行yield
                # 生成器是一个特殊的函数，每次调用yield时会产生一个值，然后暂停函数的执行，等待下一次调用
                yield {"text": sentence}
            else:  # 训练时
                # 去掉行首位的空白字符，并将文本按制表符'\t'分割为多个项，存储在列表items中
                # 他应该是数据处理的时候用\t处理了数据，在数据中间加入\t就可以自动空格显示
                items = line.strip().split("\t")
                # items[-1]最后一项作为标签
                label = items[-1]
                # 然后移除标签，因为标签不是文本
                items.pop()
                # 将items中每一项用；拼接，合并成一个文本作为一条数据样本
                text = ';'.join(items)
                # 返回一个字典
                yield {"text": get_solo(text), "label": label2id[label]}


def log_metrics_debug(output, id2label, dev_ds, bad_case_path):
    """
    Log metrics in debug mode.
    """
    predictions, label_ids, metrics = output
    pred_ids = np.argmax(predictions, axis=-1)
    logger.info("-----Evaluate model-------")
    logger.info("Dev dataset size: {}".format(len(dev_ds)))
    logger.info("Accuracy in dev dataset: {:.2f}%".format(metrics["test_accuracy"] * 100))
    logger.info(
        "Macro average | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            metrics["test_macro avg"]["precision"] * 100,
            metrics["test_macro avg"]["recall"] * 100,
            metrics["test_macro avg"]["f1-score"] * 100,
        )
    )
    for i in id2label:
        l = id2label[i]
        logger.info("Class name: {}".format(l))
        i = "test_" + str(i)
        if i in metrics:
            logger.info(
                "Evaluation examples in dev dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
                    metrics[i]["support"],
                    100 * metrics[i]["support"] / len(dev_ds),
                    metrics[i]["precision"] * 100,
                    metrics[i]["recall"] * 100,
                    metrics[i]["f1-score"] * 100,
                )
            )
        else:
            logger.info("Evaluation examples in dev dataset: 0 (0%)")
        logger.info("----------------------------")

    with open(bad_case_path, "w", encoding="utf-8") as f:
        f.write("Text\tLabel\tPrediction\n")
        for i, (p, l) in enumerate(zip(pred_ids, label_ids)):
            p, l = int(p), int(l)
            if p != l:
                f.write(dev_ds.data[i]["text"] + "\t" + id2label[l] + "\t" + id2label[p] + "\n")

    logger.info("Bad case in dev dataset saved in {}".format(bad_case_path))
