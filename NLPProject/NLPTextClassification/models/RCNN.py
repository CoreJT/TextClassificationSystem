from torch import nn
import torch
import torch.nn.functional as F
from .BasicModel import BasicModule

class RCNN(BasicModule):#继承自BasicModule 其中封装了保存加载模型的接口,BasicModule继承自nn.Module

    def __init__(self,vocab_size,opt):#opt是config类的实例 里面包括所有模型超参数的配置

        super(RCNN, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, opt.embed_size)#词嵌入矩阵 每一行代表词典中一个词对应的词向量；
        # 词嵌入矩阵可以随机初始化连同分类任务一起训练，也可以用预训练词向量初始化（冻结或微调）

        #双向lstm 由于RCNN中双向lstm一般只有一层 所以opt.drop_prop_rcnn=0.0(丢弃率)
        self.lstm = nn.LSTM(opt.embed_size,opt.recurrent_hidden_size,num_layers=opt.num_layers_rcnn,
                            bidirectional=True,batch_first=True,dropout=opt.drop_prop_rcnn)

        #全连接层 维度转换 卷积操作可以用全连接层代替
        self.linear = nn.Linear(2*opt.recurrent_hidden_size+opt.embed_size,opt.recurrent_hidden_size)

        #池化层
        self.max_pool = nn.MaxPool1d(opt.max_len)

        #全连接层分类
        self.content_fc = nn.Sequential(
            nn.Linear(opt.recurrent_hidden_size, opt.linear_hidden_size),
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
        #inputs(batch_size,seq_len)
        # 由于batch_first = True 所以inputs不用转换维度
        embeddings = self.embedding(inputs)  # (batch_size, seq_len, embed_size)

        outputs,_ = self.lstm(embeddings) #(batch_size, seq_len, recurrent_hidden_size*2)

        #将前后向隐藏状态和embedding拼接
        outputs = torch.cat((outputs[:,:,:outputs.size(2)//2],embeddings,outputs[:,:,outputs.size(2)//2:]),dim=2) #(batch_size, seq_len, embed_size+recurrent_hidden_size*2)

        #做维度转换
        outputs = self.linear(outputs) #(batch_size, seq_len, recurrent_hidden_size)

        #沿seq_len维做最大池化（全局池化）
        #先调整维度 交换recurrent_hidden_size维和seq_len维
        #即把recurrent_hidden_size作为通道维 符合一维池化的输入
        outputs = outputs.permute(0,2,1)  #(batch_size,recurrent_hidden_size,seq_len)
        outputs = self.max_pool(outputs).squeeze(2)   #(batch_size,recurrent_hidden_size)

        #通过全连接层 分类
        outputs = self.content_fc(outputs) #(batch_size,classes)

        return outputs
