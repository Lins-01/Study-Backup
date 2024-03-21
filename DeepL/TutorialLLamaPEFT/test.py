from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


dataset = load_dataset("./data/OpenAssistant___oasst_top1_2023-08-25")
print(dataset)

print(dataset["test"][0]["text"])
print("====="*20)

modelpath="D:\Software\Model-temp\Mistral-7b-v0.1"
# 加载4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.bfloat16, # 指定为bf16格式
)

# Load (slow) Tokenizer, fast tokenizer sometimes ignores added tokens
# 加载  (slow)tokenizer , fast tokenizer有时候会忽略增加的tokens
# use_fast=False即可
tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast=False)

# Add tokens <|im_start|> and <|im_end|>, latter is special eos token
# 增加tokens
tokenizer.pad_token = "</s>"
tokenizer.add_tokens(["<|im_start|>"])
tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
# 重新设置模型的embedding的维度大小-》改解构也有这么方便的api、、、
model.resize_token_embeddings(len(tokenizer))
# 获取特殊token对应的id（token化后的数字）也这么方便
model.config.eos_token_id = tokenizer.eos_token_id

print(model)
print("====="*20)

def tokenize(element):
    print(f"element-text:   {element['text']}")
    return tokenizer(
        element['text'], # 数据集的每个样本是用text作为key来索引的
        truncation=True, # 是否截断
        max_length=512, # 最大长度2048个token
        add_special_tokens=False,
    )

# dataset.map用于数据预处理，因为他可以对数据集中的每个元素应用指定的函数。
# 这里是对每个样本进行tokenize，就是将text转为数字

# dataset.map的返回值直接就是DatasetDict类型的对象，他包含一个或多个Dataset对象，一个对象就像是一个train、test测试集。直接用["train"]来访问就得到了
dataset_tokenized = dataset.map(
    tokenize,
    batched=True,
    num_proc=1, # multithreaded 多线程 ->windows下python的多线程还是垃圾
    remove_columns=["text"] # 现在不用text了，现在开始就用tokens了   dont need the strings anymore,we have tokens from here on
)


# collate function - to transform list of dictionaries [{input_ids:[123,..]},{.. ] to single batch dictionary {input_ids:[..], labels:[..],attention_mask:[..]}
def collate(elements):
    tokenlist = [e["input_ids"] for e in elements]
    tokens_maxlen = max([len(t) for t in tokenlist])  # length of longest input

    input_ids, labels, attention_masks = [], [], []
    # 每个都做padding
    for tokens in tokenlist:
        # how many pad tokens to add for this sample
        pad_len = tokens_maxlen - len(tokens)

        # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content, otherwise 0
        # 将tokens列表和填充列表（由pad_token_id重复pad_len次组成） 连接起来，并添加到input_ids列表中
        input_ids.append(tokens + [tokenizer.pad_token_id] * pad_len)
        # 将tokens列表和填充列表（由-100重复pad_len组成）......
        labels.append(tokens + [-100] * pad_len)
        # 生成mask，tokens长度部分为1，填充部分为0
        attention_masks.append([1] * len(tokens) + [0] * pad_len)

    # 组成batch
    batch = {
        "input_ids": torch.tensor(input_ids),  # 先转成tensor
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks)
    }

    return batch

collate(dataset_tokenized)