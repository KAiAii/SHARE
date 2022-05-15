import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import method
from model import BiLSTMNet
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

# 设置因素特征数据集
data_x = data[input_feature]
# 设置目标特征数据集
data_y = data[target_feature]
# data.to_csv('./dataset.csv')

train_x = data_x  # [:8640]
train_y = data_y  # [:8640]


#########
# 画图查看特征元素的曲线
########
# i = 1
# p = np.arange(1, 8, 1)
# plt.figure(figsize=(10, 10))
# for i in p:
#     plt.subplot(len(p), 1, i)
#     plt.plot(data.values[:, i])
#     plt.title(data.columns[i], y=0.5, loc='right')
#     i += 1
#
# plt.show()
#
# 格式转为numpy
train_x = torch.from_numpy(train_x.to_numpy()).float()
train_y = torch.squeeze(torch.from_numpy(train_y.to_numpy()).float())
# x转tensor
train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
# 导入网络模型
bi_lstm = BiLSTMNet(input_size=input_feature_num)
# bi_lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=2, bidirectional=True, batch_first=True)
# bi_lstm = RNN_BI(input_dim=input_feature_num)

optimizer = torch.optim.Adam(bi_lstm.parameters(), lr=0.01)
loss_func = nn.MSELoss()
epochs = 80
print(bi_lstm)
print('Start training...')

a = []
for e in range(epochs):

    # 前向传播
    y_pred = bi_lstm(train_x)
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
plt.title('Bi_LSTM训练loss')
plt.legend()
plt.show()

plt.plot(y_pred.detach().numpy(), 'r', label='y_pred')
plt.plot(train_y.detach().numpy(), 'b', label='y_train')
plt.legend()
plt.show()

plt.plot(abs(train_y.detach().numpy() - y_pred.detach().numpy()), 'b', label='loss')
plt.title('LSTM训练')
plt.legend()
plt.show()
print('Model saving...')

MODEL_PATH = 'model_bi_lstm.pth'

torch.save(bi_lstm, MODEL_PATH)

print('Model saved')
