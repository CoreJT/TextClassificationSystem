import os
from torch.utils.data import Dataset
import numpy as np
import random
import json
import jieba
import collections
import torchtext.vocab as Vocab
import torch
import sys
sys.path.append('../')
from config import opt


#数据预处理
from tqdm import tqdm
def read_cnews():
    #opt.data_root为数据集解压后 文件夹所在路径  定义在config.py中
    data = []
    #所有的主题标签
    labels = [label for label in os.listdir(opt.data_root) if label != '.DS_Store']
    print(labels)
    #标签到整数索引的映射
    labels2index = dict(zip(labels, list(range(len(labels)))))
    #整数索引到标签的映射
    index2labels = dict(zip(list(range(len(labels))),labels))
    print(labels2index)
    print(index2labels)

    #存储整数索引到标签的映射 以便预测时使用
    with open('index2labels.json','w') as f:
        json.dump(index2labels,f)
    #存储类别标签 打印分类报告会用到
    with open('labels.json','w') as f:
        json.dump(labels,f)


    for label in labels:
        folder_name = os.path.join(opt.data_root,label)
        datasub = []    #存储某一类的数据 [[string,index],...]
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                #去除文本中空白符
                review = f.read().decode('utf-8').replace('\n', '').replace('\r','').replace('\t','')
                datasub.append([review,labels2index[label]])
        data.append(datasub) #存储所有类的数据[[[string,index],...],[[string,index],...],...]


    return data

def split_data(data):
    #切分数据集 为训练集、验证集和测试集
    train_data = []
    val_data = []
    test_data = []

    #对每一类数据进行打乱
    #设置验证集和测试集中每一类样本数都为200（样本均衡）
    for data1 in data: #遍历每一类数据
        np.random.shuffle(data1) #打乱
        val_data += data1[:200]
        test_data += data1[200:400]
        train_data += data1[400:]

    np.random.shuffle(train_data) #打乱训练集 测试机和验证集不用打乱

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))

    return train_data,val_data,test_data

#读取停用词
def stopwords(fileroot):
    #fileroot为下载的停用词表所在的路径
    with open(fileroot,'r') as f:
        stopword = [line.strip() for line in f]
    print(stopword[:5])
    return stopword



#分词 去除停用词
def get_tokenized(data,stopword):
    """
    data: list of [string, label]
    """
    def tokenizer(text):
        return [tok for tok in jieba.lcut(text) if tok not in stopword]
    return [tokenizer(review) for review, _ in data]

#构建词典
def get_vocab(data,stopword):
    tokenized_data = get_tokenized(data,stopword) #分词、去除停用词
    counter = collections.Counter([tk for st in tokenized_data for tk in st]) #统计词频
    return Vocab.Vocab(counter, min_freq=5,specials=['<pad>','<unk>']) #保留词频大于5的词 <pad>对应填充项（词典中第0个词） <unk>对应低频词和停止词等未知词（词典中第1个词）


def preprocess_imdb(data, vocab,stopword):
    #将训练集、验证集、测试集中单词转换为词典中对应的索引
    max_l = 500  # 将每条新闻通过截断或者补0，使得长度变成500（所有数据统一成一个长度，方便用矩阵并行计算（其实也可以每个 batch填充成一个长度，batch间可以不一样））

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized(data,stopword) #分词、去停止词
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data]) #把单词转换为词典中对应的索引，并填充成固定长度，封装为tensor
    labels = torch.tensor([score for _, score in data])
    return features, labels

if __name__ == '__main__':

    data = read_cnews()
    print(len(data))
    train_data,val_data,test_data = split_data(data)
    stopword = stopwords('./哈工大停用词表.txt')

    vocab = get_vocab(train_data,stopword)
    print(vocab.itos[:5])
    with open('word2index.json','w') as f: #保存词到索引的映射，预测和后续加载预训练词向量时会用到
        json.dump(vocab.stoi,f)
    print("----------------")
    print(len(vocab))
    print(len(vocab.itos))
    print(len(vocab.stoi))
    print("----------------")

    X_train,y_train = preprocess_imdb(train_data,vocab,stopword)
    X_val, y_val = preprocess_imdb(val_data, vocab, stopword)
    X_test, y_test = preprocess_imdb(test_data, vocab, stopword)

    print("----------------")
    print(len(vocab))
    print(len(vocab.itos))
    print(len(vocab.stoi))
    print("----------------")

    with open('vocabsize.json','w') as f: #保存词典的大小（因为我们基于词频阈值过滤低频词，词典大小不确定，需要保存，后续模型中会用到）
        json.dump(len(vocab),f)

    #保存预处理好的训练集、验证集和测试集 以便后续训练时使用
    torch.save(X_train,'X_train.pt')
    torch.save(y_train, 'y_train.pt')
    torch.save(X_val, 'X_val.pt')
    torch.save(y_val, 'y_val.pt')
    torch.save(X_test, 'X_test.pt')
    torch.save(y_test, 'y_test.pt')

    print(X_train.shape,X_train[0])
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)

