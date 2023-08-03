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

import functools
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from utils import log_metrics_debug, preprocess_function, read_local_dataset

from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    CompressionArguments,
    EarlyStoppingCallback,
    PdArgumentParser,
    Trainer,
)
from paddlenlp.transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    export_model,
)
from paddlenlp.utils.log import logger

# yapf: disable
@dataclass
# 装饰器
# 从而让这些数据类具备了自动处理数据初始化和打印信息的功能。
# 不用写__init__、__repr__、__eq__ 等，少写了重复的代码
class DataArguments:
    # field()返回的是一个特殊的描述符(descriptor)对象，用于对字段进行配置和属性访问控制。
    # 这个描述符对象实际上并不存储字段的值，而是用于管理数据类的字段。

    # int 是数据类中应该有的，用来显示的表明类型
    # 类型注解
    max_length: int = field(default=128, metadata={"help": "Maximum number of tokens for the model."})
    early_stopping: bool = field(default=False, metadata={"help": "Whether apply early stopping strategy."})
    early_stopping_patience: int = field(default=4, metadata={"help": "Stop training when the specified metric worsens for early_stopping_patience evaluation calls"})
    train_path: str = field(default='./data/train.txt', metadata={"help": "Train dataset file path."})
    dev_path: str = field(default='./data/dev.txt', metadata={"help": "Dev dataset file path."})
    test_path: str = field(default='./data/dev.txt', metadata={"help": "Test dataset file path."})
    label_path: str = field(default='./data/label.txt', metadata={"help": "Label file path."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="ernie-3.0-tiny-medium-v2-zh", metadata={"help": "Build-in pretrained model name or the path to local model."})
    #  Optional[str]
    # 表示export_model_dir字段可以是一个字符串类型的值，也可以是None，即可选的字符串类型。
    # 在数据类中，如果某个字段可以允许为空，就可以使用Optional来表示可选类型。
    export_model_dir: Optional[str] = field(default=None, metadata={"help": "Path to directory to store the exported inference model."})
# yapf: enable


def main():
    """
    Training a binary or multi classification model
    """

    parser = PdArgumentParser((ModelArguments, DataArguments, CompressionArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.do_compress:
        training_args.strategy = "dynabert"
    if training_args.do_train or training_args.do_compress:
        training_args.print_config(model_args, "Model")
        training_args.print_config(data_args, "Data")
    paddle.set_device(training_args.device)

    # Define id2label
    # id转label
    id2label = {}
    label2id = {}
    with open(data_args.label_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            l = line.strip()
            id2label[i] = l
            label2id[l] = i

    # Define model & tokenizer
    # 如果路径是目录，进入目录读取模型
    if os.path.isdir(model_args.model_name_or_path):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, label2id=label2id, id2label=id2label
        )
    else: # 如果不是目录，直接从路径读取模型
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, num_classes=len(label2id), label2id=label2id, id2label=id2label
        )
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # load and preprocess dataset
    train_ds = load_dataset(read_local_dataset, path=data_args.train_path, label2id=label2id, lazy=False)
    dev_ds = load_dataset(read_local_dataset, path=data_args.dev_path, label2id=label2id, lazy=False)
    # 定义数据预处理函数
    trans_func = functools.partial(preprocess_function, tokenizer=tokenizer, max_length=data_args.max_length)
    # train_ds.map()：对数据集中每个样本应用trans_func函数
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    # Define the metric function.
    # 用于存储模型在验证数据集上的预测结果和真实标签
    def compute_metrics(eval_preds):
        # 使用numpy中argmax函数，找到预测概率值中最大值对应的类别标签
        pred_ids = np.argmax(eval_preds.predictions, axis=-1)
        # 创建个空字典，用于存储计算得到的评估指标
        metrics = {}
        #metrics["accuracy"] = accuracy_score(y_true=eval_preds.label_ids, y_pred=pred_ids)
        
        # 使用scikit-learn库中的precision_recall_fscore_support函数
        # 计算模型在验证数据集上的精确率、召回率、和F1分数，y_true是真实标签
        precision, recall, f1, _ = precision_recall_fscore_support(
            # y_preds=的参数是模型的预测标签
            # macro表示计算宏平均、即对每个类别单独计算指标，然后对所有类别的指标取平均值
            # 这里的 _ 变量表示我们不关心的返回值，因为在宏平均下，每个类别的指标都是有意义的，而不是单个标量值。
            y_true=eval_preds.label_ids, y_pred=pred_ids, average="macro")
        
        # 将计算得到的宏平均精确率存储到metrics字典中
        # 键为“macro_precision”，值为precision
        metrics[f"{average}_precision"] = precision
        metrics[f"{average}_recall"] = recall
        metrics[f"{average}_f1"] = f1
        return metrics

    # Define the early-stopping callback.
    if data_args.early_stopping:
        # 创建一个早停回调对象，存储在 *列表* callbacks中
        # EarlyStoppingCallback在训练过程中会监测指定的评估指标，
        # 如果在连续若干次的评估中该指标没有显著改善，就会终止训练过程
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)]
    else:
        callbacks = None

    # Define Trainer
    # 创建一个Trainer对象，并传入许多参数来配置训练过程
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # criterion criteria：标准。这里也指损失函数的意思
        criterion=paddle.nn.loss.CrossEntropyLoss(),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=callbacks, # 回调函数列表
        data_collator=DataCollatorWithPadding(tokenizer), # 数据收集器（用于数据批次的处理）
        compute_metrics=compute_metrics,#评估指标计算函数
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()# 通过trainer.train()开始训练，并且返回一个对象，包含训练过程中的一些统计信息，如训练损失、训练时间等
        metrics = train_result.metrics # 把结果中的metrics拿出来
        trainer.save_model()# 保存模型
        trainer.log_metrics("train", metrics)# 将训练过程中的指标值记录下来，方便后续查看和分析
        #trainer.log_metrics方法可以将指标值与指定的阶段（这里是 "train"）一起记录下来。

        # 删除训练过程中的检查点文件，因为训练全部结束后中间的就可以不要了
        for checkpoint_path in Path(training_args.output_dir).glob("checkpoint-*"):
            shutil.rmtree(checkpoint_path)

    # Evaluate and tests model
    # 评估和测试模型
    if training_args.do_eval:
        eval_metrics = trainer.evaluate() # 开始模型评估，并返回一些信息
        trainer.log_metrics("eval", eval_metrics)

    # export inference model
    if training_args.do_export: # 如果需要导出推理模型
        if model.init_config["init_class"] in ["ErnieMForSequenceClassification"]:
            # 如果模型是ErnieMForSequenceClassification少一个输入：token_type_ids
            input_spec = [paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids")]
        else:
            input_spec = [
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                paddle.static.InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            ]
        if model_args.export_model_dir is None:
            model_args.export_model_dir = os.path.join(training_args.output_dir, "export")
        # 导出模型到指定路径，并根据定义的input_spec来设置模型的输入
        export_model(model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir)
        # 用于保存tokenizer，以便在加载模型时能够正确进行tokenization
        tokenizer.save_pretrained(model_args.export_model_dir)
        # 获取保存id2label字典的文件路径
        id2label_file = os.path.join(model_args.export_model_dir, "id2label.json")
        # 上下文管理器，用于打开文件并确保在代码块结束后正确地关闭文件
        with open(id2label_file, "w", encoding="utf-8") as f:
            # 将id2label字典以json格式写入到文件中，以便在推理时能够正确地将预测结果映射回标签
            # ensure_ascii=false为了保证能够正确处理非ascii字符
            json.dump(id2label, f, ensure_ascii=False)
            # 用于记录id2label文件保存的位置，以便在日志中查看
            logger.info(f"id2label file saved in {id2label_file}")

    # compress
    # 模型压缩
    if training_args.do_compress:
        trainer.compress() # 根据指定的压缩策略对模型进行压缩
        for width_mult in training_args.width_mult_list: # 遍历包含不同压缩比例的列表
            # 生成压缩后模型的保存路径
            # 使用round(width_mult,2)是为了将width_mult保留两位小数，以便区分不同的压缩比例
            pruned_infer_model_dir = os.path.join(training_args.output_dir, "width_mult_" + str(round(width_mult, 2)))
            # 用于保存tokenizer，以便在加载压缩后的模型时能够正确的进行tokenization
            tokenizer.save_pretrained(pruned_infer_model_dir)
            # 获取保存id2label的路径
            id2label_file = os.path.join(pruned_infer_model_dir, "id2label.json")
            with open(id2label_file, "w", encoding="utf-8") as f:
                #  这行代码用于将 id2label 字典以 JSON 格式写入到文件中，以便在推理时能够正确地将预测结果映射回标签。
                # ensure_ascii=False 参数是为了确保能够正确处理非 ASCII 字符。
                json.dump(id2label, f, ensure_ascii=False)
                # 用于记录 id2label 文件保存的位置，以便在日志中查看
                logger.info(f"id2label file saved in {id2label_file}")

    # 用于遍历 training_args.output_dir 下的所有名为 "runs" 的文件或目录
    for path in Path(training_args.output_dir).glob("runs"):
        # 删除所有名为 "runs" 的目录，清理不需要的文件
        shutil.rmtree(path)


if __name__ == "__main__":
    main()




