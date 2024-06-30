import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers import BertModel, BertConfig
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time


class BERT4TS(nn.Module):

    def __init__(self, configs, device):
        super(BERT4TS, self).__init__()
        self.is_bert = configs.is_bert
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_bert:
            if configs.pretrain:
                self.bert = BertModel.from_pretrained('bert-base-uncased', output_attentions=True,
                                                      output_hidden_states=True)  # 加载预训练的BERT模型
            else:
                print("------------------no pretrain------------------")
                self.bert = BertModel(BertConfig())
            self.bert.encoder.layer = self.bert.encoder.layer[:configs.bert_layers]
            print("bert = {}".format(self.bert))

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.bert.named_parameters()):
                if 'LayerNorm' in name or 'position_embeddings' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.bert, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        if self.is_bert:
            outputs = self.bert(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
