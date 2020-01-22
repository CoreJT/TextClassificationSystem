from torch import nn
import torch
import torch.nn.functional as F
from .BasicModel import BasicModule


class FastText(BasicModule): #继承自BasicModule 其中封装了保存加载模型的接口,BasicModule继承自nn.Module

    def __init__(self, vocab_size,opt): #opt是config类的实例 里面包括所有模型超参数的配置
        super(FastText, self).__init__()


        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, opt.embed_size) #词嵌入矩阵 每一行代表词典中一个词对应的词向量；
        # 词嵌入矩阵可以随机初始化连同分类任务一起训练，也可以用预训练词向量初始化（冻结或微调）

        self.content_fc = nn.Sequential( #可以使用多个全连接层或batchnorm、dropout等 可以把这些模块用Sequential包装成一个大模块
            nn.Linear(opt.embed_size, opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            #可以再加一个隐层
            # nn.Linear(opt.linear_hidden_size,opt.linear_hidden_size),
            # nn.BatchNorm1d(opt.linear_hidden_size),
            # nn.ReLU(inplace=True),
            #输出层
            nn.Linear(opt.linear_hidden_size, opt.classes)
        )


    def forward(self, inputs):
        #inputs(batch_size,seq_len)
        embeddings = self.embedding(inputs) # (batch_size, seq_len, embed_size)

        #对seq_len维取平均
        content = torch.mean(embeddings,dim=1) #(batch_size,1,embed_size)

        out = self.content_fc(content.squeeze(1)) #先压缩seq_len维 (batch_size,embed_size) 然后作为全连接层的输入
        #输出 (batch_size,classes)

        return out
