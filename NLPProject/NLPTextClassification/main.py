from config import opt
import os
import torch
import models
import torch.utils.data as Data
import time
from torch.optim import lr_scheduler
import json
import jieba
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
import numpy as np
from load_word_vector import read_word_vector,load_pretrained_embedding

from tqdm import tqdm



def train(**kwargs):

    # 根据命令行参数更新配置 否则使用默认配置
    opt.parse(kwargs)

    # step1: 数据
    #词典大小
    with open('./data/vocabsize.json') as f:
        vocab_size = json.load(f)
    print("词典大小:",vocab_size)
    #标签
    with open('./data/labels.json') as f:
        labels = json.load(f)

    #读取之前预处理过程 保存的处理好的训练集、验证集和测试集
    X_train = torch.load('./data/X_train.pt')
    y_train = torch.load('./data/y_train.pt')
    X_val = torch.load('./data/X_val.pt')
    y_val = torch.load('./data/y_val.pt')
    X_test = torch.load('./data/X_test.pt')
    y_test = torch.load('./data/y_test.pt')

    #封装成DataSet
    trainset = Data.TensorDataset(X_train,y_train)
    valset = Data.TensorDataset(X_val,y_val)
    testset = Data.TensorDataset(X_test,y_test)

    #使用DataLoader并行加载数据
    train_iter = Data.DataLoader(trainset,opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    val_iter = Data.DataLoader(valset,opt.batch_size)
    test_iter = Data.DataLoader(testset,opt.batch_size)

    # step2: 模型
    model = getattr(models, opt.model)(vocab_size,opt)
    if opt.load_model_path:
        model.load(opt.load_model_path)

    #加载预训练词向量
    if opt.use_pretrained_word_vector:
        words,word2vec = read_word_vector(opt.word_vector_path) #opt.word_vector_path为下载的预训练词向量 解压后的文件所在的路径
        print("预训练词向量读取完毕！")
        #读取之前预处理过程保存的词典（词到索引的映射）
        with open('./data/word2index.json') as f:
            word2index = json.load(f)

        model.embedding.weight.data.copy_(load_pretrained_embedding(word2index, word2vec)) #使用加载完预训练词向量的词嵌入矩阵 对embdding层的词嵌入矩阵赋值
        print("预训练词向量加载完毕！")
        if opt.frozen: #冻结还是finetuning
            model.embedding.weight.requires_grad = False


    print("使用设备：",device)
    if torch.cuda.device_count() > 1: #使用多GPU进行训练
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    model.to(device)

    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                             lr = opt.lr,
                             weight_decay = opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer,opt.n_epoch,opt.lr_decay)
    # 训练
    batch_count = 0
    best_f1_val = 0.0


    for epoch in range(opt.max_epoch):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        if opt.use_lrdecay:
            scheduler.step()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            if opt.use_rnn: #梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.max_norm, norm_type=opt.norm_type)
            optimizer.step()
            train_l_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1


        #一个epoch后在验证集上做一次验证
        val_f1,val_acc = evaluate_accuracy(val_iter, model)
        if val_f1 > best_f1_val:
            best_f1_val = val_f1
            # 保存在验证集上weighted average f1最高的参数（最好的参数）
            if torch.cuda.device_count() > 1: #多GPU训练时保存参数
                print("Saving on ", torch.cuda.device_count(), "GPUs!")
                model.save_multiGPU()
            else:
                print("Saving on one GPU!")#单GPU训练时保存参数
                model.save()
            #使用当前最好的参数，在测试集上再跑一遍
            best_f1_test,best_acc_test = evaluate_accuracy(test_iter,model,True,labels)

        print('epoch %d, lr %.6f,loss %.4f, train acc %.3f, val acc %.3f,val weighted f1 %.3f, val best_weighted f1 %.3f,test best_acc %.3f,test best_weighted f1 %.3f,time %.1f sec'
              % (epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'],train_l_sum / batch_count, train_acc_sum / n, val_acc,val_f1, best_f1_val,best_acc_test,best_f1_test,time.time() - start))



def evaluate_accuracy(data_iter, net,flag=False,labels=None):
    #计算模型在验证集上的相关指标 多分类我们使用 weighed average f1-score

    acc_sum, n = 0.0, 0
    net.eval()  # 评估模式, 这会关闭dropout
    y_pred_total = []
    y_total = []
    with torch.no_grad():
        for X, y in data_iter:
            #acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            #n += y.shape[0]
            y_pred = net(X.to(device)).argmax(dim=1).cpu().numpy()
            y_pred_total.append(y_pred)
            y_total.append(y.numpy())

    y_pred = np.concatenate(y_pred_total)
    y_label = np.concatenate(y_total)
    weighted_f1 = f1_score(y_label,y_pred,average='weighted')
    accuracy = accuracy_score(y_label,y_pred)
    if flag: #当在测试集上验证时 flag设置为True  额外打印分类报告
        print(classification_report(y_label,y_pred,digits=4,target_names = labels))
        cm = confusion_matrix(y_label,y_pred)
        print(cm)
    net.train()  # 改回训练模式

    return weighted_f1,accuracy


def predict(**kwargs):
    # 根据命令行参数更新配置 否则使用默认配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备:", device)
    opt.parse(kwargs)
    text = opt.text #待分类文本

    # 词典大小
    with open('./data/vocabsize.json') as f:
        vocab_size = json.load(f)
    print(vocab_size)

    #创建指定的模型对象
    model = getattr(models, opt.model)(vocab_size, opt)

    #加载训练好的模型参数
    if device.type=='cpu': #GPU训练 CPU预测 加载参数时需要对参数进行映射
        model.load_map('./checkpoints/'+opt.model+'_best.pth',device)
    else:
        model.load('./checkpoints/' + opt.model + '_best.pth')

    #加载之前预处理过程 保存的词到索引的映射字典
    with open('./data/word2index.json') as f:
        word2index = json.load(f)

    #device = list(model.parameters())[0].device
    if opt.predict_pad: #预测时对文本进行填充（若文本长度<opt.max_len）或截断（若文本长度>opt.max_len）
        sentence = [word2index.get(word, 1) for word in jieba.lcut(text)]
        sentence = sentence[:opt.max_len] if len(sentence) > opt.max_len else sentence + [0] * (opt.max_len - len(sentence))
        sentence = torch.tensor(sentence,device=device)
    else:
        sentence = torch.tensor([word2index.get(word,1) for word in jieba.lcut(text)],device=device)
    print(sentence)
    #预测
    with torch.no_grad():
        model.eval()
        label = torch.argmax(model(sentence.view((1,-1))),dim=1)

    # 加载之前预处理过程 保存的索引到类别标签的映射字典
    with open('./data/index2labels.json') as f:
        index2labels = json.load(f)
    #输出新文本的类别标签
    print(index2labels[str(label.item())])




def help():

    '''
    打印帮助的信息： python file.py help
     '''

    print('''
   usage : python {0} <function> [--args=value,]
   <function> := train | test | help
   example: 
           python {0} train --model='TextCNN' --lr=0.01
           python {0} test --text='xxxxx'
           python {0} help
   avai    able
    args: '''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    # GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import fire
    fire.Fire()