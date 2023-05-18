# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:10:27 2023

@author: lisha
"""
from __future__ import print_function, division
#import torchsummary
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch.optim as optim
from sklearn.metrics import accuracy_score,multilabel_confusion_matrix
from torch.optim import lr_scheduler
# npmetrics
import mulroc_pr
import datainfor
from src.loss_functions.losses import AsymmetricLoss  #引入损失函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lamda=0.01
num_class=5
# Transformer Parameters
d_model=512# Embedding Size,512-1024,128-512
q_model=256
d_ff=1024# FeedForward dimension
n_layers=4  # number of Encoder of Decoder Layer
n_heads=4  # number of heads in Multi-Head Attention
d_k = d_v =int(d_model/n_heads) # dimension of K(=Q), V
batch_size=50
num_fea=4

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) 
        #print('scores',scores.size())
        #将K的最后两个维度进行转置
        # scores : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        #print('attn',attn.size())
        #attn就是注意力权重
        context = torch.matmul(attn, V) 
        # [batch_size, n_heads, len_q, d_v]
        #print('context',context.size())
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(q_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        batch_size=input_Q.size(0)
        #input_Q=input_Q.view(batch_size, -1, d_model)
        residual=input_K
        
        #print('residual',residual.size())
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # (B,D) -proj-> (B, D_new) -split-> (B, H, W)
        #改变输入到合适的形式
        # Q = self.W_Q(input_Q).view(batch_size, n_heads, d_k)
        # # # Q: [batch_size, n_heads, len_q, d_k]
        # K = self.W_K(input_K).view(batch_size, n_heads, d_k)  
        # # # K: [batch_size, n_heads, len_k, d_k]
        # V = self.W_V(input_V).view(batch_size, n_heads, d_v)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # print('Q:',Q.size())
        # print('K:',K.size())
        
        #attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) 
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) 
        # context: [batch_size, len_q, n_heads * d_v]
        output_ = self.fc(context) # [batch_size, len_q, d_model]
        #print('attn_output_',output_.size())
        output=nn.LayerNorm(d_model)(output_ + residual)
        #print('multioutput',(output_ + residual).size())
        return output, attn
    #返回子层结果和attn

#前向神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model, bias=False),
            #nn.Dropout()
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        
        residual = inputs
        #print('input',inputs.size())
        output = self.fc(inputs)
        #print('FF',output.size())
        return nn.LayerNorm(d_model)(output + residual)
    # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        
    def forward(self, Q_inputs,enc_inputs):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(Q_inputs, enc_inputs, enc_inputs) 
        #print('multiattn输出:',enc_outputs.size())
        # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) 
        #print('ff输出:',enc_outputs.size())
        # enc_outputs: [batch_size, src_len, d_model]
        #print('EncoderLayer的输出：',enc_outputs.size())
        return enc_outputs, attn
enc_self_attns=[]
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc=  nn.Sequential(
            nn.Linear(num_fea*d_model, num_class, bias=False),
            
            nn.Sigmoid()
            #nn.Softmax()
        )
        
    def forward(self,Q_inputs,enc_intputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        #word_emb = self.src_emb(enc_inputs) # [batch_size, src_len, d_model]
        #pos_emb = self.pos_emb(enc_inputs) # [batch_size, src_len, d_model]
        #enc_outputs = word_emb + pos_emb
        #enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # [batch_size, src_len, src_len]
        #enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_intputs, enc_self_attn = layer(Q_inputs,enc_intputs)
            #print('encoder输出',enc_intputs.size())
            batch_size=enc_intputs.size(0)
            enc_self_attns.append(enc_self_attn)
            #enc_intputs数据嵌入后的维数
        enc_intputs=enc_intputs.view(batch_size,-1)
        #print('分类前的输出',enc_intputs.size())
        outputs=self.fc(enc_intputs)
            
            
            
            #print('outputs',outputs.size())
        return outputs, enc_self_attns
    
# def preclass(outputs,lamda): #判断标签
#     #print('class_outputs',outputs.size())
#     #print('class_outputs',outputs)
#     max_value, preds=torch.max(outputs,1)
    
#     #print(max_value)
#     value1=max_value-lamda
#     #print(value1)   
#     duibi=value1
#     for i in range(outputs.shape[1]-1):
#         duibi=torch.cat([duibi,value1],0)
#     c=torch.reshape(duibi,(outputs.shape[1],outputs.shape[0]))
#     d=c.transpose(0,1)
#     preds=torch.gt(outputs,d)
#     preds=preds.float()
#     #print('preds',preds)
#     return preds

def preclass(outputs,lamda): #判断标签
    #print(lamda)
    A= pd.read_csv("先验信息.csv")
    A=np.array(A) 
    A=A[:,1:]
    a=A
    A=torch.from_numpy(A)
  #找到预测的最大值
    max_value, location=torch.max(outputs,1)
    location=location.numpy()
    O_=torch.zeros(outputs.shape[0],outputs.shape[1])
    preds=np.zeros((outputs.shape[0],outputs.shape[1]))
    for i in range(0,outputs.shape[0]):
        index=[]
        a1=[]
        #preds[i,location[i]]=1
          #转化概率矩阵
        O_[i,:]=outputs[i,:]/outputs[i,location[i]]-lamda
        #location[i]是第i个样本对应的max索引
        #每个输出概率除以最大输出概率
        preds1=torch.ge(O_[i,:],A[location[i],:])
        preds1=preds1.float()
        #preds1=preds1.numpy()
        preds[i,:]=preds1
        a1=a[location[i],:]
        index=[i for i,x in enumerate(a1) if x==0]
        preds[i,index]=0
        preds[i,location[i]]=1
    #print(preds)
    preds=torch.from_numpy(preds)
    return preds


def newacc(preds,labels):
    #preds=np.cpu().array(preds)
    copy_preds = preds.clone().detach().cpu()
    copy_labels = labels.clone().detach().cpu()
    copy_preds = copy_preds.numpy()
    copy_labels= copy_labels.numpy()
    # #labels=np.cpu().array(labels)
    # acc=accuracy_score(copy_labels, copy_preds)
    # acc_num=acc*copy_labels.shape[0]
    # acc_num=int(acc_num)
    acc = np.mean(np.all(np.equal(copy_labels, copy_preds), axis=1).astype("float32"))
    acc_num=acc*copy_labels.shape[0]
    acc_num=int(acc_num)
    #print(acc_num)
    return acc_num

import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_focal_loss(gamma=2, **_):
    def func(input, target):
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)
        #clamp(min=0),就相当于relu函数的等效，max(0,wx+b)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        #这个
        loss = (invprobs * gamma).exp() * loss
        return loss.mean()

    return func


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss()  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability
        
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def cross_entropy(**_):
    return torch.nn.BCEWithLogitsLoss()


def get_loss(config):
    f = globals().get(config.loss.name)
    return f(**config.loss.params)
# train_pre=[]
# train_output=[]
# train_label=[]
# test_pre=[]
# test_output=[]
# test_label=[]


