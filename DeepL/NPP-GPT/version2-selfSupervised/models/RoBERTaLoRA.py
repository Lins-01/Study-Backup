import math

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers import RobertaModel, RobertaConfig
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        self.scale = 0.01

        # Initialize the LoRA parameters
        # LoRA论文中，A矩阵初始化为随即正态分布，B矩阵初始化为零
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_layer(x) + self.scale * self.lora_B(self.lora_A(x))

class RoBERTa4TS(nn.Module):

    def __init__(self, configs, device):
        super(RoBERTa4TS, self).__init__()
        self.is_roberta = configs.is_roberta
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_roberta:
            if configs.pretrain:
                self.roberta = RobertaModel.from_pretrained('roberta-base', output_attentions=True,
                                                            output_hidden_states=True)  # 加载预训练的RoBERTa模型
            else:
                print("------------------no pretrain------------------")
                self.roberta = RobertaModel(RobertaConfig())
            self.roberta.encoder.layer = self.roberta.encoder.layer[:configs.roberta_layers]


        # Apply LoRA to each encoder layer in RoBERTa
        for i in range(len(self.roberta.encoder.layer)):
            self.roberta.encoder.layer[i].attention.self.query = LoRALayer(self.roberta.encoder.layer[i].attention.self.query)
            # self.roberta.encoder.layer[i].attention.self.key = LoRALayer(self.roberta.encoder.layer[i].attention.self.key)
            self.roberta.encoder.layer[i].attention.self.value = LoRALayer(self.roberta.encoder.layer[i].attention.self.value)

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.roberta.named_parameters()):
                if 'LayerNorm' in name or 'position_embeddings' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.roberta, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

        print("roberta = {}".format(self.roberta))

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
        if self.is_roberta:
            outputs = self.roberta(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
