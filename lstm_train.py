import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import method
from model import LSTMNet
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
file_path = './data.csv'
data = pd.read_csv(file_path)
feature = ['airTemp', 'humidity', 'insolation', 'windSpeed']
# 设定输入特征
input_feature = ['airTemp', 'humidity', 'insolation', 'windSpeed']
input_feature_num = 4
# 设定目标特征
target_feature = ['PvOutput']

# # 删除功率为空的数据组
# data = data.dropna(subset=['Power'])
#
# # NAN值赋0
# data = data.fillna(0)
# data[data < 0] = 0
#
# # 设定样本数目
# data = data[:8640]
#
# # 归一化
# scaler = MinMaxScaler()
# data[feature] = scaler.fit_transform(data[feature].to_numpy())

# 数据集分配
train_x, train_y = method.create_dataset(data, target_feature, input_feature)

lstm = LSTMNet(input_size=input_feature_num)
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.05)
loss_func = nn.MSELoss()
epochs = 80
print(lstm)
print('Start training...')

a = []
for e in range(epochs):
    # 前向传播
    y_pred = lstm(train_x)
    y_pred = torch.squeeze(y_pred)
    loss = loss_func(y_pred, train_y)
    a.append(loss.item())
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 20 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(e, loss.item()))

x = np.linspace(0, 80, 80)
plt.plot(x, a, 'b', label='loss')
plt.title('LSTM训练loss')
plt.legend()
plt.show()

plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
plt.plot(train_y.detach().numpy(), 'b', label='y_train')
plt.title('LSTM训练')
plt.legend()
plt.show()

plt.plot(abs(train_y.detach().numpy() - y_pred.detach().numpy()), 'b', label='loss')
plt.title('LSTM训练')
plt.legend()
plt.show()

print('Model saving...')

MODEL_PATH = 'model_lstm.pth'

torch.save(lstm, MODEL_PATH)

print('Model saved')
