import torch.nn as nn
import time
import torch

class BasicModule(nn.Module):
   '''
   封装了nn.Module，主要提供save和load两个方法
   '''

   def __init__(self,opt=None):
       super(BasicModule,self).__init__()
       self.model_name = str(type(self)) # 模型的默认名字

   def load(self, path):
       '''
       加载模型
       可指定路径
       '''
       self.load_state_dict(torch.load(path))

   def load_map(self, path,device): #如果在GPU上训练 在CPU上加载 可以调用这个函数
       '''
       加载模型
       可指定路径
       '''
       self.load_state_dict(torch.load(path,map_location=device))

   def save(self, name=None):
       '''
       保存模型，默认使用“模型名字_best”作为文件名，
       '''
       if name is None:
           prefix = 'checkpoints/' + self.model_name.split('.')[-2] + '_best.pth'
           #name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
       torch.save(self.state_dict(), prefix) #只保存模型的参数
       return name

   def save_multiGPU(self, name=None):  #如果使用多GPU训练，保存模型时，可以调用这个函数。
       '''
       保存模型，默认使用“模型名字_best”作为文件名，
       '''
       if name is None:
           prefix = 'checkpoints/' + self.model_name.split('.')[-2] + '_best.pth'
           # name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
       torch.save(self.module.state_dict(), prefix)  # 只保存模型的参数
       return name

