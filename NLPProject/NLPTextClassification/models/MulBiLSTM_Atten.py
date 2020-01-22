from torch import nn
import torch
import torch.nn.functional as F
from .BasicModel import BasicModule

class MulBiLSTM_Atten(BasicModule):#继承自BasicModule 其中封装了保存加载模型的接口,BasicModule继承自nn.Module

    def __init__(self,vocab_size,opt):#opt是config类的实例 里面包括所有模型超参数的配置

        super(MulBiLSTM_Atten, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, opt.embed_size)#词嵌入矩阵 每一行代表词典中一个词对应的词向量；
        # 词嵌入矩阵可以随机初始化连同分类任务一起训练，也可以用预训练词向量初始化（冻结或微调）

        #多层双向LSTM 默认seq_len作为第一维 也可以通过batch_first=True 设置batch_size 为第一维
        self.lstm = nn.LSTM(opt.embed_size,opt.recurrent_hidden_size, opt.num_layers,
                            bidirectional=True, batch_first=True, dropout=opt.drop_prop)

        self.tanh1 = nn.Tanh()
        self.u = nn.Parameter(torch.Tensor(opt.recurrent_hidden_size * 2, opt.recurrent_hidden_size * 2))
        #定义一个参数（变量） 作为Attention的Query
        self.w = nn.Parameter(torch.Tensor(opt.recurrent_hidden_size*2))

        #均匀分布 初始化
        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.u, -0.1, 0.1)

        #正态分布 初始化
        #nn.init.normal_(self.w, mean=0, std=0.01)
        self.tanh2 = nn.Tanh()
        #最后的全连接层
        self.content_fc = nn.Sequential(
            nn.Linear(opt.recurrent_hidden_size*2, opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(opt.drop_prop),
            # 可以再加一个隐层
            # nn.Linear(opt.linear_hidden_size,opt.linear_hidden_size),
            # nn.BatchNorm1d(opt.linear_hidden_size),
            # nn.ReLU(inplace=True),
            # 输出层
            nn.Linear(opt.linear_hidden_size, opt.classes)
        )

    def forward(self, inputs):
        #由于batch_first = True 所有inputs不用转换维度
        embeddings = self.embedding(inputs)  # (batch_size, seq_len, embed_size)

        outputs,_ = self.lstm(embeddings)   #(batch_size,seq_len,recurrent_hidden_size*2)

        #M = self.tanh1(outputs) #(batch_size,seq_len,recurrent_hidden_size*2)
        M = torch.tanh(torch.matmul(outputs, self.u)) #也可以先做一个线性变换 再通过激活函数  作为Key

        #M (batch_size,seq_len,recurrent_hidden_size*2) self.w (recurrent_hidden_size*2,)
        #torch.matmul(M, self.w) (batch_size,seq_len) w作为Query与各个隐藏状态（Key） 做内积
        #再对第一维seq_len做softmax 转换为概率分布 (batch_size,seq_len)  得到权重
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # (batch_size,seq_len,1)

        #对各个隐藏状态和权重 对应相乘
        out = alpha * outputs #(batch_size,seq_len,recurrent_hidden_size*2)

        #对乘积求和 out为加权求和得到的特征向量
        out = torch.sum(out,dim=1) #(batch_size,recurrent_hidden_size*2)

        #out = F.relu(out)

        out = self.content_fc(out)  #(batch_size,classes)

        return out



