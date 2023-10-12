import numpy as np
import json
import os
import sys
import paddle
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.vision import transforms as T
from paddle.io import Dataset
import pandas as pd
from PIL import Image
import imghdr
from paddle.vision import models
from paddlenlp.transformers import ErnieMModel,ErnieMTokenizer,ErnieMConfig
from paddle.nn import functional as F

# TODO 1: 测试集数据预处理
def process_string(input_str):
    input_str = input_str.replace('&#39;', ' ')
    input_str = input_str.replace('<b>','')
    input_str = input_str.replace('</b>','')
    #input_str = unidecode(input_str)  
    return input_str
    
class NewsContextDatasetEmbs(Dataset):
    def __init__(self, context_data_items_dict, queries_root_dir, split):
        self.context_data_items_dict = context_data_items_dict
        self.queries_root_dir = queries_root_dir
        self.idx_to_keys = list(context_data_items_dict.keys())
        self.transform =T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        self.split=split
    def __len__(self):
        return len(self.context_data_items_dict)   


    def load_img_pil(self,image_path):
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
    def load_imgs_direct_search(self,item_folder_path,direct_dict):   
        list_imgs_tensors = []
        count = 0   
        keys_to_check = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
        for key1 in keys_to_check:
            if key1 in direct_dict.keys():
                for page in direct_dict[key1]:
                    image_path = os.path.join(item_folder_path,page['image_path'].split('/')[-1])
                    try:
                        pil_img = self.load_img_pil(image_path)
                    except Exception as e:
                        print(e)
                        print(image_path)
                    if pil_img == None: continue
                    transform_img = self.transform(pil_img)
                    count = count + 1 
                    list_imgs_tensors.append(transform_img)
        stacked_tensors = paddle.stack(list_imgs_tensors, axis=0)
        return stacked_tensors
    def load_captions(self,inv_dict):
        captions = ['']
        pages_with_captions_keys = ['all_fully_matched_captions','all_partially_matched_captions']
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
                    
        pages_with_title_only_keys = ['partially_matched_no_text','fully_matched_no_text']
        for key1 in pages_with_title_only_keys:
            if key1 in inv_dict.keys():
                for page in inv_dict[key1]:
                    if 'title' in page.keys():
                        title = process_string(page['title'])
                        captions.append(title)
        return captions

    def load_captions_weibo(self,direct_dict):
        captions = ['']
        keys = ['images_with_captions','images_with_no_captions','images_with_caption_matched_tags']
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
        #print(captions)
        return captions
        #加载img文件夹
    def load_queries(self,key):
        caption = self.context_data_items_dict[key]['caption']
        image_path = os.path.join(self.queries_root_dir,self.context_data_items_dict[key]['image_path'])
        pil_img = self.load_img_pil(image_path)
        transform_img = self.transform(pil_img)
        return transform_img, caption
    def __getitem__(self, idx):
        #print(idx)
        #print(self.context_data_items_dict)      
        #idx = idx.tolist()               
        key = self.idx_to_keys[idx]
        #print(key)
        item=self.context_data_items_dict.get(str(key))
        #print(item)
        # 如果为test没有label属性
        #print(self.split)
        if self.split=='train' or self.split=='val':
            label = paddle.to_tensor(int(item['label']))
            direct_path_item = os.path.join(self.queries_root_dir,item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir,item['inv_path'])
            inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
            direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
            captions= self.load_captions(inv_ann_dict)
            captions += self.load_captions_weibo(direct_dict)
            imgs = self.load_imgs_direct_search(direct_path_item,direct_dict)     
            qImg,qCap =  self.load_queries(key)
            sample = {'label': label, 'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap}
        else:
            direct_path_item = os.path.join(self.queries_root_dir,item['direct_path'])
            inverse_path_item = os.path.join(self.queries_root_dir,item['inv_path'])
            inv_ann_dict = json.load(open(os.path.join(inverse_path_item, 'inverse_annotation.json')))
            direct_dict = json.load(open(os.path.join(direct_path_item, 'direct_annotation.json')))
            captions= self.load_captions(inv_ann_dict)
            captions += self.load_captions_weibo(direct_dict)
            imgs = self.load_imgs_direct_search(direct_path_item,direct_dict)     
            qImg,qCap =  self.load_queries(key)
            sample = {'caption': captions,'imgs': imgs,  'qImg': qImg, 'qCap': qCap}
        #print(sample)
        #print(len(captions)) 
        #print(imgs.shape)#[5,3,224,224]  
        return sample,  len(captions), imgs.shape[0]


# TODO 2: 预测测试集
def collate_context_bert_test(batch):
    samples = [item[0] for item in batch]
    max_captions_len = max([item[1] for item in batch])
    max_images_len = max([item[2] for item in batch])
    qCap_batch = []
    qImg_batch = []
    img_batch = []
    cap_batch = []
    for j in range(0,len(samples)):  
        sample = samples[j]    
        captions = sample['caption']
        cap_len = len(captions)
        for i in range(0,max_captions_len-cap_len):
            captions.append("")
        if len(sample['imgs'].shape) > 2:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1],sample['imgs'].shape[2],sample['imgs'].shape[3])
        else:
            padding_size = (max_images_len-sample['imgs'].shape[0],sample['imgs'].shape[1])
        padded_mem_img = paddle.concat((sample['imgs'], paddle.zeros(padding_size)),axis=0)
        img_batch.append(padded_mem_img)
        cap_batch.append(captions)
        qImg_batch.append(sample['qImg'])
        qCap_batch.append(sample['qCap'])        
    img_batch = paddle.stack(img_batch, axis=0)
    qImg_batch = paddle.stack(qImg_batch, axis=0)
    return cap_batch, img_batch, qCap_batch, qImg_batch

class EncoderCNN(nn.Layer):
    def __init__(self, resnet_arch = 'resnet101'):
        super(EncoderCNN, self).__init__()
        if resnet_arch == 'resnet101':
            resnet = models.resnet101()
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2D((1, 1))
    def forward(self, images, features='pool'):
        out = self.resnet(images)
        if features == 'pool':
            out = self.adaptive_pool(out)
            out = paddle.reshape(out, (out.shape[0],out.shape[1]))
        return out


from paddle.vision import models
import paddle
from paddlenlp.transformers import ErnieMModel, ErnieMTokenizer, BertModel, ErnieViLModel, ErnieViLProcessor
from paddle.nn import functional as F
from paddle import nn
import matplotlib.pyplot as plt
import numpy as np


class EncoderCNN(nn.Layer):
    def __init__(self, resnet_arch='resnet101'):
        super(EncoderCNN, self).__init__()
        if resnet_arch == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2D((1, 1))

    def forward(self, images, features='pool'):
        out = self.resnet(images)
        if features == 'pool':
            out = self.adaptive_pool(out)
            out = paddle.reshape(out, (out.shape[0], out.shape[1]))
        return out


class NetWork(nn.Layer):
    def __init__(self, mode):
        super(NetWork, self).__init__()
        self.mode = mode

        self.fuse = BertModel.from_pretrained("bert-base-multilingual-cased").encoder

        ernie_vil = ErnieViLModel.from_pretrained("ernie_vil-2.0-base-zh")
        self.text_encoder = ernie_vil.text_model
        self.visual_encoder = ernie_vil.vision_model
        self.processor = ErnieViLProcessor.from_pretrained("ernie_vil-2.0-base-zh")

        self.fuse.layers = self.fuse.layers[:4]
        self.norm = nn.LayerNorm(768)
        # self.resnet = EncoderCNN()
        # self.projection = nn.Linear(2*(768+2048),1024)
        self.classifier = nn.Linear(768, 3)
        # self.attention_text = nn.MultiHeadAttention(768,16)
        # self.attention_image = nn.MultiHeadAttention(2048,16)
        if self.mode == 'text':
            self.classifier = nn.Linear(768, 3)

    def forward(self, qCap, qImg, caps, imgs):
        encode_dict_qcap = self.processor(text=qCap, max_length=128, truncation=True, padding='max_length')
        input_ids_qcap = encode_dict_qcap['input_ids']
        input_ids_qcap = paddle.to_tensor(input_ids_qcap)
        with paddle.no_grad():
            qcap_feature, pooled_output = self.text_encoder(input_ids_qcap)  # (b,length,dim)
        if self.mode == 'text':
            logits = self.classifier(qcap_feature[:, 0, :].squeeze(1))
            return logits
        # print(len(caps))
        # print(caps[0])
        # print(caps[0].shape)
        # print(imgs.shape)
        # print(qImg.shape)

        caps_feature = []
        with paddle.no_grad():
            for i, caption in enumerate(caps):
                encode_dict_cap = self.processor(text=caption, max_length=128, truncation=True, padding='max_length')
                input_ids_caps = encode_dict_cap['input_ids']
                input_ids_caps = paddle.to_tensor(input_ids_caps)
                cap_feature, pooled_output = self.text_encoder(input_ids_caps)  # (b,length,dim)
                caps_feature.append(cap_feature)
        caps_feature = paddle.stack(caps_feature, axis=0)  # (b,num,length,dim)
        caps_feature = caps_feature.mean(axis=1)  # (b,length,dim)

        # caps_feature = self.attention_text(qcap_feature,caps_feature,caps_feature) #(b,length,dim)

        imgs_features = []
        with paddle.no_grad():
            for img in imgs:
                imgs_feature, pooled_output = self.visual_encoder(img)  # (length,dim)
                imgs_features.append(imgs_feature)

        imgs_features = paddle.stack(imgs_features, axis=0)  # (b,length,dim)
        qImg_features = []
        with paddle.no_grad():
            for qImage in qImg:
                qImg_feature, pooled_output = self.visual_encoder(qImage.unsqueeze(axis=0))  # (1,dim)
                qImg_features.append(qImg_feature)
        qImg_feature = paddle.stack(qImg_features, axis=0)  # (b,1,dim)
        imgs_features = imgs_features.mean(axis=1)
        # imgs_features = self.attention_image(qImg_feature,imgs_features,imgs_features) #(b,1,dim)
        # b,n,l,dim = imgs_features.mean(axis=1)
        # print(len(qcap_feature))
        # print(qcap_feature[0].shape)
        # print(caps_feature.shape)
        # print(qImg_feature.shape)
        # print(imgs_features.shape)

        # [1, 128, 768] [1, 128, 768] [1, 1, 2048] [1, 1, 2048] origin
        # print(qcap_feature.shape,caps_feature.shape,qImg_feature.shape,imgs_features.shape)
        # print((qcap_feature[:,0,:].shape,caps_feature[:,0,:].shape,qImg_feature.squeeze(1).shape,imgs_features.squeeze(1).shape))
        # ([1,768], [1 , 768], [1, 2048], [1,  2048])
        # feature = paddle.concat(x=[qcap_feature, paddle.reshape(caps_feature,[b,-1,dim]), qImg_feature.squeeze(1), paddle.reshape(imgs_features,[b,-1,dim])], axis=1)
        feature = paddle.concat(x=[qcap_feature, caps_feature, qImg_feature.squeeze(1), imgs_features], axis=1)
        feature = self.fuse(feature).mean(axis=1)
        feature = self.norm(feature)
        logits = self.classifier(feature)
        return logits


def evaluate(model, result_csv):
    results = []
    # 切换model模型为评估模式，关闭dropout等随机因素
    model.eval()
    count=0
    for batch in test_dataloader:
        count+=1
        cap_batch, img_batch, qCap_batch, qImg_batch = batch
        logits = model(qCap=qCap_batch,qImg=qImg_batch,caps=cap_batch,imgs=img_batch)
        # 预测分类
        probs = F.softmax(logits, axis=-1)
        label = paddle.argmax(probs, axis=1).numpy()
        results += label.tolist()
        print(count)
    test_str={0:"non-rumor",1:"rumor",2:"unverified"}
    print(results[:5])
    # 输出结果
    # id/label
    id_list=range(len(results))
    frame = pd.DataFrame({'id':id_list,'label':results})
    frame = frame['label'].map(test_str)
    frame.to_csv(result_csv,index=False,sep=',')


if __name__ == '__main__':
    test_csv = sys.argv[1]  # 测试集路径
    result_csv = sys.argv[2]  # 结果文件路径
    #test_csv = '/home/aistudio/admin-evaluation/paddlepaddle/dataset_items_test.json'  # 测试集路径
    #result_csv = '/home/aistudio/admin-evaluation/paddlepaddle/result.csv'  # 结果文件路径
    data_items_test = json.load(open(test_csv))# 处理测试集数据
    test_dataset = NewsContextDatasetEmbs(data_items_test,os.path.dirname(os.path.realpath(__file__)),'test')
    #test_dataset = NewsContextDatasetEmbs(data_items_test,'/home/aistudio/admin-evaluation/paddlepaddle/','test')
    test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=4, shuffle=False, collate_fn = collate_context_bert_test, return_list=True)
    model = NetWork("image")
    state_dict = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model/model_best.pdparams'))
    #state_dict = paddle.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), '/home/aistudio/admin-evaluation/paddlepaddle/tests/model/model_best.pdparams'))
    # 加载自己的权重
    model.set_dict(state_dict)
    # 加载模型
    evaluate(model, result_csv=result_csv)  # 预测测试集
