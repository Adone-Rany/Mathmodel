# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import division

import numpy as np  



from sklearn.model_selection import train_test_split  
import matplotlib.pyplot as plt  


# 初始化数据和标签列表
data = []
labels = []

# 从文件中读取数据
with open("data.txt") as ifile:
    for line in ifile:
        tokens = line.strip().split('\t')
        if tokens != ['']:
            # 将特征和标签分开
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])

# 将数据转换为 NumPy 数组
x = np.array(data)
y = np.double(labels)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)


# 为训练集数据随机引入缺失值（NaN）
s = x_train.shape  # 获取训练集的形状（行数和列数）
for k in range(0, 1000):
    # 随机选择一个样本（行）和一个特征（列），将其值设为 NaN（缺失值）
    x_train[np.random.randint(s[0]), np.random.randint(s[1])] = np.nan

# 为测试集数据随机引入缺失值（NaN）
s = x_test.shape  # 获取测试集的形状
for k in range(0, 10):
    # 同样随机选择样本和特征，设置缺失值，但数量远少于训练集
    x_test[np.random.randint(s[0]), np.random.randint(s[1])] = np.nan

''''''''  
#xgboost可以自动处理缺失数据集
''''' xgboost '''  

#先安装xgboost包pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xgboost
import xgboost as xgb


# 注释说明：XGBoost 能够自动处理含有缺失值的数据，无需手动填充或删除

# 定义 XGBoost 模型的参数
params = {
    'booster': 'gbtree',               # 使用基于树的模型（gbtree）
    'objective': 'reg:gamma',          # 回归任务，使用 gamma 分布损失函数（可能适合特定数据分布）
    'gamma': 0.01,                     # 控制节点分裂的最小损失减少量，值越大越保守
    'max_depth': 6,                    # 树的最大深度，控制模型复杂度
    'lambda': 3,                       # L2 正则化系数，防止过拟合
    'subsample': 0.8,                  # 每次训练使用样本的比率，防止过拟合
    'colsample_bytree': 0.8,           # 每棵树使用的特征比例，防止过拟合
    'min_child_weight': 3,             # 子节点样本权重和的最小值，控制分裂
    'eta': 0.1,                        # 学习率，控制每一步的权重更新幅度
    'seed': 1000,                      # 随机种子，保证结果可复现
    'nthread': 4,                      # 使用的线程数
}
# 将训练数据转换为 XGBoost 的 DMatrix 格式，支持缺失值处理
dtrain = xgb.DMatrix(x_train, y_train)

num_rounds = 800  # 设置迭代次数（树的数量）
model = xgb.train(params, dtrain, num_rounds)  # 训练模型

# 对训练集进行预测
d_train = xgb.DMatrix(x_train)
y_predict_train = model.predict(d_train)  # 得到训练集预测结果

# 对测试集进行预测
d_test = xgb.DMatrix(x_test)
y_predict = model.predict(d_test)  # 得到测试集预测结果

''''' RMSE_train '''
# 计算训练集上的 RMSE（均方根误差）
N = y_train.size
X = range(0, N)
Y1 = np.double(y_train)           # 真实值
Y2 = np.double(y_predict_train)   # 预测值
GError = np.abs(Y1 - Y2)  # 计算绝对误差
RMSE = np.sqrt(np.dot(GError, GError) / N)  # RMSE 公式计算

print("RMSE_train=", RMSE)  # 输出训练集误差


''''' RMSE_test '''
N = y_test.size
X = range(0, N)
Y1 = np.double(y_test)         # 真实值
Y2 = np.double(y_predict)      # 预测值
GError = np.abs(Y1 - Y2)  # 计算绝对误差
RMSE = np.sqrt(np.dot(GError, GError) / N)  # RMSE 公式计算

print("RMSE_test=", RMSE)  # 输出测试集误差

# 绘图部分
# fig1: 绘制误差曲线（预测值与真实值之间的绝对误差）
fig1 = plt.figure('fig1')
plt.plot(X, GError)  # 误差随样本变化的折线图
# plt.scatter(X, Y1, marker='o', c='b')  # 可选：绘制真实值散点图
# plt.scatter(X, Y2, marker='o', c='r')  # 可选：绘制预测值散点图

# fig2: 绘制真实值与预测值对比图
fig2 = plt.figure('fig2')
plt.plot(X, Y1, 'b', lw=1)  # 真实值，蓝色实线
plt.plot(X, Y2, 'r+')       # 预测值，红色加号点
plt.show()  # 显示图形
