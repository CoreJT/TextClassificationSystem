from torch import nn
import torch
import torch.nn.functional as F
from .BasicModel import BasicModule

class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()

        self.channel_size = channel_size
        #将序列长度减半
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0), #在每个通道上(一维) 一边填充0个0(不填充) 另一边填1个0
            nn.MaxPool1d(kernel_size=3, stride=2) #序列长度减半 height = (height-kernel_size+padding+stride)//stride=height // 2
        )
        self.conv = nn.Sequential(  #等长卷积 不改变height
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),

            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(),
            nn.Conv1d(self.channel_size, self.channel_size, kernel_size=3, padding=1),
        )

    def forward(self, x): #(batch_size,channel_size,seq_len)
        x_shortcut = self.maxpool(x) # (batch_size,channel_size,seq_len//2)
        x = self.conv(x_shortcut)#(batch_size,channel_size,seq_len//2)
        x = x + x_shortcut#(batch_size,channel_size,seq_len//2) shortcut 残差连结

        return x

class DPCNN(BasicModule):#继承自BasicModule 其中封装了保存加载模型的接口,BasicModule继承自nn.Module

    def __init__(self,vocab_size,opt): #opt是config类的实例 里面包括所有模型超参数的配置

        super(DPCNN, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, opt.embed_size)#词嵌入矩阵 每一行代表词典中一个词对应的词向量；
        # 词嵌入矩阵可以随机初始化连同分类任务一起训练，也可以用预训练词向量初始化（冻结或微调）

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(opt.embed_size, opt.channel_size, kernel_size=3, padding=1), #same卷积 不改变height/序列长度
            nn.BatchNorm1d(num_features=opt.channel_size),
            nn.ReLU(),
            nn.Dropout(opt.drop_prop_dpcnn)
        )

        #卷积块 same卷积 不改变height
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=opt.channel_size),
            nn.ReLU(),
            nn.Conv1d(opt.channel_size, opt.channel_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=opt.channel_size),
            nn.ReLU(),
            nn.Conv1d(opt.channel_size, opt.channel_size, kernel_size=3, padding=1),
        )

        self.seq_len = opt.max_len #序列最大长度
        resnet_block_list = []  #存储多个残差块

        while (self.seq_len > 2): #每经过一个残差块 序列长度减半 只要长度>2 就不停地加残差块
            resnet_block_list.append(ResnetBlock(opt.channel_size))
            self.seq_len = self.seq_len // 2

        #将残差块 构成残差层 作为一个子模块
        self.resnet_layer = nn.Sequential(*resnet_block_list)

        #输出层 分类
        self.linear_out = nn.Linear(self.seq_len * opt.channel_size, opt.classes)

    def forward(self, inputs):

        embeddings = self.embedding(inputs) #(batch_size,max_len,embed_size)

        x = embeddings.permute(0, 2, 1) #(batch_size,embed_size,max_len) 交换维度 作为1维卷积的输入 embed_size 作为通道维
        x = self.region_embedding(x) #（batch_size,channel_size,max_len)

        x = self.conv_block(x) #(batch_size,channel_size,max_len)

        x = self.resnet_layer(x) #经过多个残差块 每次长度减半 (batch_size,channel_size,self.seq_len)

        x = x.permute(0, 2, 1)  #(batch_size,self.seq_len,channel_size)

        x = x.contiguous().view(x.size(0), -1) #(batch_size,self.seq_len*channel_size) 拉伸为向量
        out = self.linear_out(x)      #(batch_size,classes)
        return out

