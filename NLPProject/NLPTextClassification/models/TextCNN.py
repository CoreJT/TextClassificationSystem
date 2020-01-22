from torch import nn
import torch
import torch.nn.functional as F
from .BasicModel import BasicModule

#自定义时序（全局）最大池化层
class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2]) #  (batch_size, channel, 1)


# 多输入通道的一维卷积和单输入通道的2维卷积等价
# 这里按多输入通道的一维卷积来做 也可以用单输入通道的2维卷积来做
class TextCNN(BasicModule): #继承自BasicModule 其中封装了保存加载模型的接口,BasicModule继承自nn.Module

    def __init__(self, vocab_size, opt):#opt是config类的实例 里面包括所有模型超参数的配置

        super(TextCNN, self).__init__()


        # 嵌入层
        self.embedding = nn.Embedding(vocab_size,opt.embed_size)#词嵌入矩阵 每一行代表词典中一个词对应的词向量；
        # 词嵌入矩阵可以随机初始化连同分类任务一起训练，也可以用预训练词向量初始化（冻结或微调）

        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(opt.num_channels, opt.kernel_sizes): #num_channels定义了每种卷积核的个数 kernel_sizes定义了每种卷积核的大小
            self.convs.append(nn.Conv1d(in_channels=opt.embed_size,
                                        out_channels=c,
                                        kernel_size=k))
        #定义dropout层
        self.dropout = nn.Dropout(opt.drop_prop)
        #定义输出层
        self.fc = nn.Linear(sum(opt.num_channels), opt.classes)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()


    def forward(self, inputs):
        # inputs(batch_size,seq_len)
        embeddings = self.embedding(inputs) # (batch_size, seq_len, embed_size)

        # 根据conv1d的输入要求 把通道维提前(这里的通道维是词向量维度)
        # （batch_size,channel/embed_size,seq_len)
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，会得到一个(batch_size,num_channel(卷积核的个数),seq_len-kernel_size+1）大小的tensor
        # 在时序最大池化后会得到一个形状为(batch_size, num_channel, 1)的 tensor
        # 使用squeeze去掉最后一维 并在通道维上连结 得到(batch_size,sum(num_channels))大小的tensor
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)

        # 应用丢弃法后使用全连接层得到输出 (batch_size,classes)
        outputs = self.fc(self.dropout(encoding))
        return outputs


