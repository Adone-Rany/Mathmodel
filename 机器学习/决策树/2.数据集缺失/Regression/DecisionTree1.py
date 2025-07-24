# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import division
import numpy as np
from sklearn import tree as skt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing  
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

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
# 用平均值填补缺失值（NaN）
# 注意：计算平均值时只能使用训练数据，不能使用测试数据，以防止数据泄露

s = x_train.shape  # 获取训练集的形状（行数、列数）
mean_train = []    # 用于存储每个特征的平均值

# 遍历每一个特征维度（列）
for k in range(0, s[1]):
    lx = x_train[:, k]  # 取出第 k 列的特征数据

    # 计算该特征列的平均值（忽略 NaN）
    mean_train.append(np.mean(lx[~np.isnan(lx)]))

    # 将该列中的 NaN 值替换为该列的平均值
    lx[np.isnan(lx)] = mean_train[-1]
    x_train[:, k] = lx  # 更新训练集该列数据

    # 对测试集的第 k 列也进行缺失值填充，使用训练集的平均值
    ls = x_test[:, k]
    ls[np.isnan(ls)] = mean_train[-1]
    x_test[:, k] = ls  # 更新测试集该列数据

''''' DecisionTree '''

# 创建决策树回归模型
DecisionTree = skt.DecisionTreeRegressor()

# 使用填补后的训练数据训练模型
DecisionTree.fit(x_train, y_train)

# 在训练集上进行预测
y_predict_train = DecisionTree.predict(x_train)


# 计算训练集上的 RMSE（均方根误差）
N = y_train.size
X = range(0, N)
Y1 = np.double(y_train)           # 真实值
Y2 = np.double(y_predict_train)   # 预测值
GError = np.abs(Y1 - Y2)  # 计算绝对误差
RMSE = np.sqrt(np.dot(GError, GError) / N)  # RMSE 公式计算

print("RMSE_train=", RMSE)  # 输出训练集误差


# 在测试集上进行预测
y_predict = DecisionTree.predict(x_test)

# 计算测试集上的 RMSE
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