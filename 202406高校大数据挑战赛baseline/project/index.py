import os
import numpy as np#矩阵运算与科学计算的库
from sklearn.model_selection import train_test_split
import torch#深度学习库,pytorch
import torch.nn as nn#neural network,神经网络

from Model import BaselineModel

def predict(feats,model_name='temp.pth',batch_size=128,train_mean=0,train_std=1):
    #测试数据
    x=60#60个观测站点
    model = BaselineModel()
    model.load_state_dict(torch.load(model_name))
    model.eval()  # 将模型设置为评估模式
    #预测24个小时的用的是71*x个站点(i:i-24)时刻的36个特征
    data=[]
    for i in range(24):
        data.append(feats[:,i:i-24,].reshape(feats.shape[0],-1))
    data=np.array(data)#24个小时,71*x个站点 【i:i-24时刻*36个特征】
    #【24小时*71*x个站点】*【i:i-24时刻*36个特征】
    data=data.reshape(-1,data.shape[-1])
    data=(data-train_mean)/train_std
    print(f"input.shape:{data.shape}")
    #test_preds=【24小时*71*x个站点】*【预测值】
    test_preds=np.zeros(len(data))
    for idx in range(0,len(data),batch_size):
        data1=torch.Tensor(data[idx:idx+batch_size]).reshape(-1,1,24,36)
        test_preds[idx:idx+batch_size]=model(data1).detach().numpy().reshape(-1)
    test_preds=test_preds.reshape(24,71,x,-1)
    #71个周,24小时,x个站点的预测值
    test_preds=test_preds.transpose(1,0,2,3)
    return test_preds

def invoke(input_dir):
    date='0629_2'
    np.random.seed(2024)
    #测试数据
    x=60#60个观测站点
    #71个不连续的周,56(每3个小时测一次),观测4个特征,9个观测方位,x个站点
    cenn_data=np.load(os.path.join(input_dir,'cenn_data.npy')).mean(axis=-2,keepdims=True)#真实情况是np.load加载的
    print(f"cenn_data.shape:{cenn_data.shape}")
    #将3个小时变成1个小时  (71,168,4,9,x)
    cenn_data_hour=np.repeat(cenn_data, 3, axis=1)

    cenn_data_hour=cenn_data_hour.transpose(0,4,1,2,3)#71*x*168*4*9
    cenn_data_hour=cenn_data_hour.reshape(71*x,168,4)
    #cenn/temp_lookback.npy 71个不连续的周  1个小时一次  x站上一周的温度
    temp_lookback = np.load(os.path.join(input_dir,'temp_lookback.npy'))
    print(f"temp_lookback.shape:{temp_lookback.shape}")
    temp_lookback=temp_lookback.transpose(0,2,1,3)#71,x,168,1
    temp_lookback=temp_lookback.reshape(71*x,168,1)
    #cenn/wind_lookback.npy 71个不连续的周  1个小时一次  x站上一周的风速
    wind_lookback = np.load(os.path.join(input_dir,'wind_lookback.npy'))
    print(f"wind_lookback.shape:{wind_lookback.shape}")
    wind_lookback=wind_lookback.transpose(0,2,1,3)#71,x,168,1
    wind_lookback=wind_lookback.reshape(71*x,168,1)
    #71*x个站点,168小时,38个特征
    total_feats=np.concatenate((cenn_data_hour,temp_lookback,wind_lookback),axis=-1)
    # 保存到 project 当中
    save_path = os.path.join('/home/mw','project')
    train_mean=np.load(os.path.join(save_path,'train_mean.npy'))
    train_std=np.load(os.path.join(save_path,'train_std.npy'))
    temp_predict=predict(feats=total_feats,model_name=os.path.join(save_path,f'{date}temp.pth'),batch_size=128,train_mean=train_mean,train_std=train_std)
    wind_predict=predict(feats=total_feats,model_name=os.path.join(save_path,f'{date}wind.pth'),batch_size=128,train_mean=train_mean,train_std=train_std)    
    np.save(os.path.join(save_path,'temp_predict.npy'),temp_predict)
    np.save(os.path.join(save_path,'wind_predict.npy'),wind_predict)