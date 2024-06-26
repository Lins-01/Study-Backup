from functools import partial
import numpy as np
import time
import os
import copy
import json
import random
from tqdm import tqdm

import paddle
from paddlenlp.datasets import load_dataset
import paddle.nn.functional as F
import paddle.nn as nn
import paddlenlp as ppnlp
from paddlenlp.transformers import LinearDecayWithWarmup
import pandas as pd

# 读取数据
import json

# data_items_train = json.load(open("/home/has/IKCEST/queries_dataset_merge/dataset_items_train.json", encoding="utf-8"))
# data_items_val = json.load(open("/home/has/IKCEST/queries_dataset_merge/dataset_items_val.json", encoding="utf-8"))


import paddle
from paddle.vision import transforms as T
from paddle.io import Dataset
import json
from urllib.parse import urlparse
from PIL import Image
import os
import imghdr
from paddleocr import PaddleOCR


def process_string(input_str):
    input_str = input_str.replace('&#39;', '')
    input_str = input_str.replace('<b>', '')
    input_str = input_str.replace('</b>', '')
    # input_str = unidecode(input_str)
    return input_str


class NewsContextDatasetEmbs(Dataset):
    def __init__(self, context_data_items_dict, queries_root_dir, split):
        # json.load的json文件，里面每条是个词典
        self.context_data_items_dict = context_data_items_dict
        self.queries_root_dir = queries_root_dir
        # 获取json文件中的0、1、2~xxx
        self.idx_to_keys = list(context_data_items_dict.keys())
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 划分不同数据集执行不同步骤、train、val、test
        self.split = split
        # self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False)

        self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        # self.ocr1 = PaddleOCR(use_angle_cls=True,lang="ch")

    def __len__(self):
        # 数据集的长度
        return len(self.context_data_items_dict)

        # 别的格式的图片全转为rgb

    def load_img_pil(self, image_path):
        # 根据图像文件的前几个字节识别图像文件格式。
        # 如果是gif转为rgb
        if imghdr.what(image_path) == 'gif':
            try:
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    return img.convert('RGB')
            except:
                return None
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    # 获取到数据集中用于证据的所有图片
    def load_imgs_direct_search(self, item_folder_path, direct_dict):
        # list_imgs_tensors = []
        # count = 0
        keys_to_check = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']
        img = ''
        for key1 in keys_to_check:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    image_path = os.path.join(item_folder_path,
                                              page['image_path'].split('/')[-1])  # 第二个参数拿到具体图片名称，eg：2433.jpg
                    # image_path = rf"{image_path}"
                    # print('type img_path'+"======"*10)
                    # print(type(image_path))
                    # print('image_path111111'+"======"*10)
                    # print(image_path)
                    # image_path = os.path.abspath(image_path)
                    # print('image_path-abspath'+"======"*10)
                    # print(image_path)
                    # image_path = str(image_path)
                    # image_path = image_path.replace("\\","/")
                    # print('image_path222'+"======"*10)
                    # print(image_path)
                    # image_path = rf"{image_path}"

                    # try:
                    #     pil_img = self.load_img_pil(image_path)
                    # except Exception as e:
                    #     print(e)
                    #     print(image_path)
                    # if pil_img == None: continue
                    if image_path == None: continue
                    # result =self.ocr1(image_path,cls=True)
                    result = self.ocr.ocr(image_path, cls=True)
                    text = ''
                    if result:
                        for idx in range(len(result)):
                            res = result[idx]
                            # for line in res:
                            #     print(line)
                            txts = res[1][0]
                            text += f"{idx}、" + txts
                    img = text
                    # print('img111'+"======"*10)
                    # print(img)
                    # transform_img = self.transform(pil_img)
                    # count = count + 1
                    # list_imgs_tensors.append(transform_img)
        # stacked_tensors = paddle.stack(list_imgs_tensors, axis=0)
        # print('img222'+"======"*10)
        # print(img)
        return img

    def load_captions(self, inv_dict):
        captions = ['']
        pages_with_captions_keys = ['all_fully_matched_captions', 'all_partially_matched_captions']
        for key1 in pages_with_captions_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        item = page['title']
                        item = process_string(item)
                        captions.append(item)

                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue
                            sub_captions_list.append(sub_caption_filter)
                            unfiltered_captions.append(sub_caption)
                        captions = captions + sub_captions_list

        pages_with_title_only_keys = ['partially_matched_no_text', 'fully_matched_no_text']
        for key1 in pages_with_title_only_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        title = process_string(page['title'])
                        captions.append(title)
        return captions

    def load_captions_weibo(self, direct_dict):
        captions = ['']
        keys = ['images_with_captions', 'images_with_no_captions', 'images_with_caption_matched_tags']
        for key1 in keys:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    if 'page_title' in page.keys():
                        item = page['page_title']
                        item = process_string(item)
                        captions.append(item)
                    if 'caption' in page.keys():
                        sub_captions_list = []
                        unfiltered_captions = []
                        for key2 in page['caption']:
                            sub_caption = page['caption'][key2]
                            sub_caption_filter = process_string(sub_caption)
                            if sub_caption in unfiltered_captions: continue
                            sub_captions_list.append(sub_caption_filter)
                            unfiltered_captions.append(sub_caption)
                        captions = captions + sub_captions_list
                        # print(captions)
        return captions

    # 获取到img中的所有待证明图片
    def load_queries(self, key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.queries_root_dir, self.context_data_items_dict[key]['image_path'])
        # pil_img = self.load_img_pil(image_path)
        # transform_img = self.transform(pil_img)
        result = self.ocr.ocr(image_path, cls=True)
        text = ''
        if result:
            for idx in range(len(result)):
                res = result[idx]
                # for line in res:
                #     print(line)
                txts = res[1][0]
                text += f"{idx}、" + txts
        qimg = text
        return qimg, caption

        # result = result[1]
        # result = np.array(result)
    def __getitem__(self, idx):
        # print(idx)
        # print(self.context_data_items_dict)
        # idx = idx.tolist()
        # key是当前对应第几个样本。test_dataset[953]中的953
        key = self.idx_to_keys[idx]
        # print(key)
        # context_data_items_dict是那个总的json文件，根据key就是index索引到对应的内容
        item = self.context_data_items_dict.get(str(key))
        # print(item)
        # 如果为test没有label属性
        # print(self.split)
        if self.split == 'train' or self.split == 'val':
            label = int(item['label'])
            direct_path_item = os.path.join(self.queries_root_dir, item[
                'direct_path'])  # eg :.../Paddle2023IKCEST/queries_dataset_merge + val/img_html_news/389

            inverse_path_item = os.path.join(self.queries_root_dir,
                                             item['inv_path'])  # 索引到img_html_news和inverse具体样本的文件夹，这里不用判空
            try:
                inv_ann_dict = json.load(
                    open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding="utf-8"))
            except FileNotFoundError:
                inv_ann_dict = {}
            # inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding="utf-8"))
            if inv_ann_dict:
                captions = self.load_captions(inv_ann_dict)
            else:
                captions = ['']
            # captions = self.load_captions(inv_ann_dict)
            try:
                direct_dict = json.load(
                    open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding="utf-8"))  # 每个样本目录下的总json文件
            except FileNotFoundError:
                direct_dict = {}

            if direct_dict:
                captions += self.load_captions_weibo(direct_dict)
                imgs = self.load_imgs_direct_search(direct_path_item, direct_dict)
            else:
                captions += ['']
                imgs = ['']
            # print('direct_dict'+"="*40)
            # print(direct_dict)
            # 传入的是样本目录，和它的json文件
            # imgs = self.load_imgs_direct_search(direct_path_item, direct_dict)
            qImg, qCap = self.load_queries(key)
            sample = {'label': label, 'caption': captions, 'imgs': imgs, 'qImg': qImg, 'qCap': qCap}
        else:
            direct_path_item = os.path.join(self.queries_root_dir, item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir, item['inv_path'])
            try:
                inv_ann_dict = json.load(
                    open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding="utf-8"))
            except FileNotFoundError:
                inv_ann_dict = {}
            # inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding="utf-8"))
            if inv_ann_dict:
                captions = self.load_captions(inv_ann_dict)
            else:
                captions = ['']
            # captions = self.load_captions(inv_ann_dict)
            try:
                direct_dict = json.load(
                    open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding="utf-8"))  # 每个样本目录下的总json文件
            except FileNotFoundError:
                direct_dict = {}

            if direct_dict:
                captions += self.load_captions_weibo(direct_dict)
                imgs = self.load_imgs_direct_search(direct_path_item, direct_dict)
            else:
                captions += ['']
                imgs = ['']
            # inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json'), encoding="utf-8"))
            # direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json'), encoding="utf-8"))
            # captions = self.load_captions(inv_ann_dict)
            # captions += self.load_captions_weibo(direct_dict)
            # imgs = self.load_imgs_direct_search(direct_path_item, direct_dict)
            qImg, qCap = self.load_queries(key)
            sample = {'caption': captions, 'imgs': imgs, 'qImg': qImg, 'qCap': qCap}

        # print(sample)
        # print(len(captions))
        # print(type(imgs))
        # print(imgs.size)
        # print(imgs.shape)
        return sample


#### load Datasets ####


# def train_dataset():
#     train_dataset = NewsContextDatasetEmbs(data_items_train, '/home/has/IKCEST/queries_dataset_merge', 'train')
#     return train_dataset

#
# def val_dataset():
#     val_dataset = NewsContextDatasetEmbs(data_items_val, '/home/has/IKCEST/queries_dataset_merge', 'val')
#     return val_dataset


# def test_dataset():
#     test_dataset = NewsContextDatasetEmbs(data_items_test, '/home/has/IKCEST/queries_dataset_merge', 'test')
#     return test_dataset

# 制作测试集文件
def make_test(data_items_test, test_path):
    # 直接加载测试集
    test_dataset = NewsContextDatasetEmbs(data_items_test,
                                          test_path,
                                          'test')
    # test_dataset = {'caption': captions, 'imgs': imgs, 'qImg': qImg, 'qCap': qCap}

    # sample = {'label': label, 'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap}
    # 假设 data 是一个包含100个数据的列表
    data = test_dataset
    bar = tqdm(data, total=len(data))

    # 初始化一个空列表，用于存储每个数据的 caption 和 qCap  和 label字段
    output_data_list = []

    label_mapping = {0: 'non-rumor', 1: 'rumor', 2: 'unverified'}
    id = 0
    max_len = 20000
    with open('test-ocr-1.json', 'w', encoding='utf-8') as f:
        for item in bar:

            # 提取 caption 和 qCap
            caption = item['caption']
            qCap = item['qCap']
            # label = item['label']
            qImg = item['qImg']
            imgs = item['imgs']
            # input = 'qCap:'+str(qCap) + 'caption:'+str(caption)

            # 转换label为含义标签
            # label = label_mapping.get(label,str(label))

            # concat
            sentence1 = str(qImg) + str(qCap)
            sentence2 = str(imgs) + str(caption)
            # 去掉超长的特殊文本
            if len(sentence1) > max_len:
                sentence1 = sentence1[:max_len]
            if len(sentence2) > max_len:
                sentence2 = sentence2[:max_len]

            # 将数据保存为字典
            output_data = {"id": id, "sentence1": sentence1, "sentence2": sentence2}
            id += 1
            # output_data_list.append(output_data)

            # 将列表保存为 ldJSON 文件
            # 不存在内存中，每次都写入一次json文件，并且用\n隔开，就是ldjson的格式了
            json.dump(output_data, f, ensure_ascii=False)
            f.write('\n')

            # 测试用
            # if id > 3:
            #     break


# __name__是python文件的一个属性，当这个文件被直接运行的时候，属性为__main__
# 如果这个文件被当作包调用，就不执行这部分
# 一般用来在包中测试代码用
if __name__ == '__main__':
    data_items_test = json.load(
        open(r"E:/Document/CodeSpace/Data_set/Paddle2023IKCEST/queries_dataset_merge/dataset_items_test.json",
             encoding="utf-8"))

    test_path = r'E:/Document/CodeSpace/Data_set/Paddle2023IKCEST/queries_dataset_merge'

    make_test(data_items_test, test_path)
