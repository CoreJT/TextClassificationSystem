import torch

class DefaultConfig(object):

    model = 'FastText'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 256  # batch size
    num_workers = 4  # 加载数据使用的线程数

    #下载数据集 解压缩后得到的文件夹所在的路径
    data_root = '/Users/apple/Downloads/THUCNews-1'


    max_epoch = 20
    lr = 0.01  # initial learning rate
    weight_decay = 1e-4  # 损失函数 正则化
    embed_size = 100 #词嵌入维度
    drop_prop = 0.5 #丢弃率
    classes = 14  #分类类别数
    max_len = 500 #序列最大长度

    #学习率衰减相关超参数
    use_lrdecay = True #是否使用学习率衰减
    lr_decay = 0.95  # 衰减率
    n_epoch = 1  #每隔n_epoch个epoch衰减一次 lr = lr * lr_decay

	
    #TextCNN相关的超参数
    kernel_sizes = [3,4,5] #一维卷积核的大小
    num_channels = [100,100,100] #一维卷积核的数量

    #FastText相关的超参数
    linear_hidden_size =512  #隐层单元数

    #MulBiLSTM/MulBiLSTM_Atten相关超参数
    recurrent_hidden_size = 128 #循环层 单元数
    num_layers = 2      #循环层 层数
    
    #RCNN相关超参数
    num_layers_rcnn = 1 #循环层 层数
    drop_prop_rcnn = 0.0 #1个循环层设置为0 丢弃率  
	
    #DPCNN相关超参数
    channel_size = 250
    drop_prop_dpcnn = 0.2

    #梯度裁剪相关超参数
    use_rnn = False
    norm_type = 1
    max_norm = 5

    #预训练词向量相关超参数
    use_pretrained_word_vector = False
    word_vector_path = '/Users/apple/Downloads/sgns.sogou.word' #下载的预训练词向量 解压后的文件所在的路径
    frozen = False

    #待分类文本
    text="众所周知，一支球队想要夺冠，超级巨星必不可少，不过得到超级巨星并不简单，方式无非两种，一是自己培养，这种方式适用于所有球队，二是交易，这种方式基本只适用于大市场球队——事实就是，30支球队之间并非完全公平，超级巨星依然更愿意前往大城市。"

    #预测时是否对文本进行填充或截断
    predict_pad = False

def parse(self, kwargs):
    '''
    根据字典kwargs 更新 默认的config参数
    '''
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            # 警告还是报错，取决个人喜好
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    # 打印配置信息
    print('user config:')
    for k, v in self.__class__.__dict__.items(): #python3 中iteritems()已经废除了
        if not k.startswith('__'):
            print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()
