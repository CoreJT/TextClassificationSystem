from config import opt
import torch
import numpy as np

def read_word_vector(path): #path为 下载的预训练词向量 解压后的文件所在的路径
    #读取预训练词向量
    with open(path, 'r') as f:
        words = set()  # 定义一个words集合
        word_to_vec_map = {}  # 定义词到向量的映射字典
        for line in f:  #跳过文件的第一行
            break

        for line in f:  # 遍历f中的每一行
            line = line.strip().split()  # 去掉首尾空格，每一行以空格切分  返回一个列表  第一项为单词 其余为单词的嵌入表示
            curr_word = line[0]  # 取出单词
            words.add(curr_word)  # 加到集合/词典中
            # 定义词到其嵌入表示的映射字典
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map



def load_pretrained_embedding(word2index, word2vector):#word2index是构建的词典（单词到索引的映射），word2vector是预训练词向量（单词到词向量的映射）

    embed = torch.zeros(len(word2index), opt.embed_size) # 初始化词嵌入矩阵为0
    oov_count = 0 # 找不到预训练词向量的词典中单词的个数

    for word, index in word2index.items(): #遍历词典中的每个单词 及其在词典中的索引
        try: #如果单词有对应的预训练词向量 则用预训练词向量对词嵌入矩阵的对应行进行赋值
            embed[index, :] = torch.from_numpy(word2vector[word])
        except KeyError:
            oov_count += 1

    if oov_count > 0:
        print("There are %d oov words."%oov_count)
    return embed #返回词嵌入矩阵

