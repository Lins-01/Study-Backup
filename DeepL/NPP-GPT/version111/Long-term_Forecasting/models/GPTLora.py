import numpy as np
import torch
import torch.nn as nn
from torch import optim
import math

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.in_features = original_layer.weight.size(0)
        self.out_features = original_layer.weight.size(1)
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        self.scale = 0.01

        # Initialize the LoRA parameters
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_layer(x) + self.scale * self.lora_B(self.lora_A(x))


class GPTLora(nn.Module):

    def __init__(self, configs, device):
        super(GPTLora, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]


        # Apply LoRA to each attention layer in GPT-2
        for i in range(len(self.gpt2.h)):
            self.gpt2.h[i].attn.c_attn = LoRALayer(self.gpt2.h[i].attn.c_attn)

        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                # if 'ln' in name or 'wpe' in name or 'lora' in name or 'wte' in name or 'mlp' in name:
                if 'ln' in name or 'wpe' in name or 'lora' in name or 'wte' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()

        self.cnt = 0

        print("gpt2 = {}".format(self.gpt2))
        # 打印预训练模型的一些权重
        print("Pretrained GPT-2 model weights (first few layers):====================")
        for name, param in list(self.gpt2.named_parameters())[:5]:
            print(f"{name}: {param.data[:2]}")

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
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs



