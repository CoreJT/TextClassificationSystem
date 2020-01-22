from torch import nn
import torch
import torch.nn.functional as F
from .BasicModel import BasicModule


class MulBiLSTM(BasicModule):#继承自BasicModule 其中封装了保存加载模型的接口,BasicModule继承自nn.Module
    def __init__(self, vocab_size,opt):#opt是config类的实例 里面包括所有模型超参数的配置
        super(MulBiLSTM, self).__init__()

        #嵌入层
        self.embedding = nn.Embedding(vocab_size, opt.embed_size)#词嵌入矩阵 每一行代表词典中一个词对应的词向量；
        # 词嵌入矩阵可以随机初始化连同分类任务一起训练，也可以用预训练词向量初始化（冻结或微调）

        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=opt.embed_size,
                               hidden_size=opt.recurrent_hidden_size,
                               num_layers=opt.num_layers,
                               bidirectional=True,
                               dropout=opt.drop_prop
                               )
        self.fc = nn.Linear(4 * opt.recurrent_hidden_size, opt.classes)  # 初始时间步和最终时间步的隐藏状态作为全连接层输入

    def forward(self, inputs):
        # inputs的形状是(批量大小, 词数)，因为上述定义的LSTM没有设置参数batch_first=True(默认False),所以需要将序列长度(seq_len)作为第一维，所以将输入转置后再提取词特征
        embeddings = self.embedding(inputs.permute(1,0)) # (seq_len, batch_size,embed_size)

        # rnn.LSTM只传入输入embeddings（第一层的输入），因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(seq_len, batch_size, 2 * recurrent_hidden_size)
        outputs, _ = self.encoder(embeddings)  # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (batch_size, 4 * recurrent_hidden_size)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.fc(encoding)
        #(batch_size,classes)
        return outs




