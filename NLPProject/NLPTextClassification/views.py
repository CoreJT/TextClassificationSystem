from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
from django.views.decorators.csrf import csrf_exempt 

#from . import predict


from . import config
import os
import sys
sys.path.append( "path" )
import torch
from . import models
import torch.utils.data as Data
import time
from torch.optim import lr_scheduler
import json
import jieba
from torchsummary import summary


def hello(request):
    return HttpResponse("Hello,world")
    
@csrf_exempt
def index(request):
    print("======index======")
    
    if request.method == "POST":
        print("======POST请求======")
        text = request.POST.get("text")
        model = request.POST.get("model")
        print(text)
        print(model)
        label = textPredict(text,model);
        return render(request, "index2.html",{'result':label,"orign":text,"model":model})
    else:
        text = request.GET.get("textInputArea")
        print(text)
        return render(request, "index.html",{'result':" 等待分析","orign":""})
    
    
    
@csrf_exempt
def predict(request):
    print(request.get_full_path());
    model='FastText';
    if request.method == "POST":
        print("======POST请求======")
        text = request.POST.get("text")
        model = request.POST.get("model")
        print(text)
        print(model)
        #return render(request, "index.html",{'result':" 等待分析p"})
    else:
        print("======GET请求======")
        text = request.GET.get("text")
        model = request.GET.get("model")
        print(text)
        print(model)
        print("======index界面======")
       # return render(request, "index.html",{'result':" 等待分析g"})
    
    label = textPredict(text,model);
    return render(request, "index2.html",{'result':label,"orign":text,"model":model})
    
def textPredict(text,modelName):
    print(modelName)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = config.DefaultConfig();
    print(text)
    #text = "在今天的NBA比赛中，詹姆斯成为了最年轻的33000分先生。"
    print(text)
    with open('NLPTextClassification/data/vocabsize.json') as f:
        vocab_size = json.load(f)
    print(vocab_size)
    #modelName=repr(modelInput);
    #model = getattr(models, opt.model)(vocab_size, opt)
    model = getattr(models, modelName)(vocab_size, opt)

    if device.type=='cpu':
        #model.load_map('NLPTextClassification/checkpoints/'+opt.model+'_best.pth',device)
        model.load_map('NLPTextClassification/checkpoints/'+modelName+'_best.pth',device)
    else:
        #model.load('NLPTextClassification/checkpoints/' + opt.model + '_best.pth')
        model.load('NLPTextClassification/checkpoints/' + modelName + '_best.pth')
    with open('NLPTextClassification/data/word2index.json') as f:
        word2index = json.load(f)

    #device = list(model.parameters())[0].device
    sentence = torch.tensor([word2index.get(word,1) for word in jieba.lcut(text)],device=device)
    print(sentence)
    with torch.no_grad():
        model.eval()
        label = torch.argmax(model(sentence.view((1,-1))),dim=1)

    with open('NLPTextClassification/data/index2labels.json') as f:
        index2labels = json.load(f)

    print(index2labels[str(label.item())])
    return index2labels[str(label.item())]
    print("right------------------------------！")
    
