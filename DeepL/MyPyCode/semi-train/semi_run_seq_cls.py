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
# 方便提示python函数的输入类型和输出类型的，一个实验性功能，不用管
from __future__ import annotations

import json
import os
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
from paddle.metric import Accuracy, Precision, Recall
from utils import DataArguments, ModelArguments, load_config, seq_convert_example

import paddlenlp
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import (
    EarlyStoppingCallback,
    PdArgumentParser,
    Trainer,
    TrainingArguments,
    get_last_checkpoint,
)
from paddlenlp.transformers import ErnieForSequenceClassification, ErnieTokenizer
from paddlenlp.utils.log import logger


# 写在utils里面
# @dataclass
#
#
# class DataArguments:
#     early_stopping: bool = field(default=False, metadata={"help": "Whether apply early stopping strategy."})
#     early_stopping_patience: int = field(default=4, metadata={
#         "help": "Stop training when the specified metric worsens for early_stopping_patience evaluation calls"})


def main():
    parser = PdArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Log model and data config
    model_args, data_args, training_args = load_config(
        model_args.config, "SequenceClassification", data_args.dataset, model_args, data_args, training_args
    )
    # Print model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    paddle.set_device(training_args.device)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, world_size: {training_args.world_size}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    data_args.dataset = data_args.dataset.strip()
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.dataset)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    # 加载数据集，已集成在paddlenlp中
    data_files = ("./data/train.ldjson", "./data/dev.ldjson", "./data/test.ldjson")
    raw_datasets = load_dataset("clue", data_args.dataset, splits=("train", "dev", "test"), data_files=data_files)
    print('raw_datasets========================')
    print(raw_datasets["train"])
    data_args.label_list = getattr(raw_datasets["train"], "label_list", None)
    num_classes = len(raw_datasets["train"].label_list)

    # Define tokenizer, model, loss function.
    model = ErnieForSequenceClassification.from_pretrained(model_args.model_name_or_path, num_classes=num_classes)
    tokenizer = ErnieTokenizer.from_pretrained(model_args.model_name_or_path)
    criterion = nn.loss.CrossEntropyLoss() if data_args.label_list else nn.loss.MSELoss()

    # Define dataset pre-process function
    trans_fn = partial(
        seq_convert_example,
        tokenizer=tokenizer,
        label_list=data_args.label_list,
        max_seq_len=data_args.max_seq_length,
        dynamic_max_length=data_args.dynamic_max_length,
    )

    # Define data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Dataset pre-process
    logger.info("Data Preprocessing...")
    if training_args.do_train:
        train_dataset = raw_datasets["train"].map(trans_fn, lazy=training_args.lazy_data_processing)
    if training_args.do_eval:
        eval_dataset = raw_datasets["dev"].map(trans_fn, lazy=training_args.lazy_data_processing)
    if training_args.do_predict:
        test_dataset = raw_datasets["test"].map(trans_fn, lazy=training_args.lazy_data_processing)

    # Define the metrics of tasks.
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

        preds = paddle.to_tensor(preds)
        label = paddle.to_tensor(p.label_ids)

        metric = Accuracy()
        metric.reset()
        result = metric.compute(preds, label)
        metric.update(result)
        accu = metric.accumulate()
        metric.reset()
        return {"accuracy": accu}
        # paddle.metric里面的precision和recall只能评估二分类任务
        # # precision
        # precision = Precision()
        # precision.reset()
        # precision.update(preds, label)
        # prec = precision.accumulate()
        # precision.reset()
        #
        # # recall
        # recall = Recall()
        # recall.reset()
        # recall.update(preds, label)
        # rec = recall.accumulate()
        # recall.reset()
        #
        # f1 = (2 * prec * rec) / (prec + rec)

        # 这个可以运行，但结果好像也不太对，感觉还是只能二分类
        # m = paddlenlp.metrics.AccuracyAndF1()
        # correct = m.compute(preds, label)
        # m.update(correct)
        # res = m.accumulate()
        # return {"accuracy": accu, "f1_score": res}

    # Define the early-stopping callback.
    if data_args.early_stopping:
        # 创建一个早停回调对象，存储在 *列表* callbacks中
        # EarlyStoppingCallback在训练过程中会监测指定的评估指标，
        # 如果在连续若干次的评估中该指标没有显著改善，就会终止训练过程
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)]
    else:
        callbacks = None

    trainer = Trainer(
        model=model,
        criterion=criterion,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        # 每次保存的时候，不会运行到这里，只有最后全结束才打印
        # print("=" * 20 + "运行到这里111" + "=" * 20)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        # print("=" * 20 + "运行到这里222222" + "=" * 20)

        # # 删除训练过程中的检查点文件，因为训练全部结束后中间的就可以不要了
        # for checkpoint_path in Path(training_args.output_dir).glob("checkpoint-*"):
        #     shutil.rmtree(checkpoint_path)

    # Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    if training_args.do_predict:
        test_ret = trainer.predict(test_dataset)
        trainer.log_metrics("test", test_ret.metrics)
        logits = test_ret.predictions
        max_value = np.max(logits, axis=1, keepdims=True)
        exp_data = np.exp(logits - max_value)
        probs = exp_data / np.sum(exp_data, axis=1, keepdims=True)
        out_dict = {"label": probs.argmax(axis=-1).tolist(), "confidence": probs.max(axis=-1).tolist()}
        out_file = open(os.path.join(training_args.output_dir, "test_results.json"), "w")
        json.dump(out_dict, out_file)

    # Export inference model
    if training_args.do_export:
        # You can also load from certain checkpoint
        # trainer.load_state_dict_from_checkpoint("/path/to/checkpoint/")
        input_spec = [
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ]
        model_args.export_model_dir = os.path.join(model_args.export_model_dir, data_args.dataset, "export")
        paddlenlp.transformers.export_model(
            model=trainer.model, input_spec=input_spec, path=model_args.export_model_dir
        )
        trainer.tokenizer.save_pretrained(model_args.export_model_dir)


if __name__ == "__main__":
    main()
