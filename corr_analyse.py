import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = './data.csv'
data = pd.read_csv(file_path)
# data.index = pd.to_datetime(data.Timestamp)
data = data.dropna(subset=['PvOutput'])
# print(data.index)
print(data.corr()['PvOutput'])
sns.heatmap(data.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
plt.show()