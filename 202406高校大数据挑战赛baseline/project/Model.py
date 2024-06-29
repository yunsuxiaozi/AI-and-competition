import torch#深度学习库,pytorch
import torch.nn as nn#neural network,神经网络
import torch.nn.functional as F#神经网络函数库
import torch.optim as optim#一个实现了各种优化算法的库

class BaselineModel(nn.Module):
    def __init__(self,):
        super(BaselineModel,self).__init__()
        self.conv=nn.Sequential(
                  #1*24*36->16*24*36
                  nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1),
                  nn.BatchNorm2d(16),
                  #16*24*36->16*12*18
                  nn.MaxPool2d(kernel_size=2,stride=2),
                  nn.GELU(),
                  #16*12*18->32*12*18
                  nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
                  nn.BatchNorm2d(32),
                  #32*12*18->64*12*18
                  nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
                  nn.BatchNorm2d(64),
                  #64*12*18->64*6*9
                  nn.MaxPool2d(kernel_size=2,stride=2),
                  nn.GELU(),
                  #64*6*9->128*6*9
                  nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=1,padding=2),
                  nn.BatchNorm2d(128),
                  #128*6*9->128*3*4
                  nn.MaxPool2d(kernel_size=2,stride=2),
                  nn.GELU(),
        )
        self.head=nn.Sequential(
                nn.Linear(128*3*4,128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Linear(128,256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Linear(256,1)
        )
        
    def forward(self,x):
        x=self.conv(x)
        x=x.reshape(x.shape[0],-1)
        return self.head(x)
