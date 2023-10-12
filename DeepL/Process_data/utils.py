import jieba
import numpy as np
import paddle
from paddlenlp.data import Vocab 

def convert_example(example, tokenizer, is_test=False):
    """
    jieba 分词,转换id,再转换为numpy的tensor(方便加快运算)
    """
    
    # tokens = tokenizer.cut(example["text"])
    input_ids = tokenizer.encode(example["text"])
    # print("="*30+'example')
    # print(example)
    # ==============================example
    # {'text': '赢在心理，输在出品！杨枝太酸，三文鱼熟了，酥皮焗杏汁杂果可以换个名（九唔搭八）', 'label': '0'}
    # print("="*30+'tokens')
    # print('tokens  len:',len(tokens))
    # print(tokens)
    # print("="*30+'input_ids')
    # print(input_ids)
    # print('input_ids  len:',len(input_ids))
    
    # ==============================tokens
    # ['赢', '在心', '理', '，', '输在', '出品', '！', '杨枝', '太酸', '，', '三文鱼', '熟', '了', '，', '酥皮', '焗', '杏汁', '杂果', '可以', '换个', '名', '（', '九唔', '搭八', '）']
    # tokens  len: 25
    # 和下面数字在，senta_word-dict里面的序号不对应，这是怎么分的？
    # 可能 -> 还是一个模型网络吧？不是固定写在文件里的？
    # ==============================input_ids
    # [656582, 967208, 318502, 1106339, 1, 693836, 1106328, 728300, 34934, 1106339, 677464, 1168226, 823066, 1106339, 706897, 1078813, 895713, 76982, 660347, 1, 179592, 1106335, 554600, 1, 1106336]
    # input_ids  len: 25
    
    # paddle这里把数据都转换成np形式了，换成了tensor，没有像torch自己弄一种tensor形式
    # 因为变成tensor会加快运算
    # 数据类型变了，内容没变
    valid_length = np.array(len(input_ids), dtype='int64') # array(25, dtype=int64)
    input_ids = np.array(input_ids, dtype='int64')

    if not is_test:
        label = np.array(example["label"], dtype="int64") # 把label对应的 0/1也变成了np的0/1
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length

def create_dataloader(dataset,
                      trans_fn=None,
                      mode='train',
                      batch_size=1,
                      use_gpu=False,
                      pad_token_id=0,
                      batchify_fn=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        use_gpu(obj:`bool`, optional, defaults to obj:`False`): Whether to use gpu to run.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.apply(trans_fn, lazy=True)

    if mode == 'train' and use_gpu:
        sampler = paddle.io.DistributedBatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        shuffle = True if mode == 'train' else False
        sampler = paddle.io.BatchSampler(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_sampler=sampler,
        return_list=True,
        collate_fn=batchify_fn)
    return dataloader
