# -*- coding: utf-8 -*-
# !/usr/bin/python

from __future__ import division

import numpy as np
from sklearn.model_selection import train_test_split

# 初始化两个空列表，用于存储数据和对应的标签
data = []
labels = []

# 打开文件 data.txt，读取数据
with open("data.txt") as ifile:
    for line in ifile:
        # 去除行首尾空白字符，并按制表符 \t 分割
        tokens = line.strip().split('\t')
        # 确保不是空行
        if tokens != ['']:
            # 将前 k 个元素转为浮点数作为特征，最后一个元素作为标签
            data.append([float(tk) for tk in tokens[:-1]])  # 特征向量
            labels.append(tokens[-1])  # 类别标签（字符串）

# 将特征数据转换为 NumPy 数组，便于后续处理
x = np.array(data)
# 将标签转换为 NumPy 数组
y = np.array(labels)
# 将数据集划分为训练集和测试集，测试集占比 1%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

''''' RandomForest '''
import sklearn.ensemble as ske

RandomForest = ske.RandomForestRegressor(n_estimators=100)
RandomForest.fit(x_train, y_train)
y_predict = RandomForest.predict(x_train)
''''' RandomForest '''
# 计算训练集上的 RMSE
N = y_train.size
X = range(0, N)
Y1 = np.double(y_train)
Y2 = np.double(y_predict)

GError = np.abs(Y1 - Y2)
RMSE = np.sqrt(np.dot(GError, GError) / N)
print("RMSE_train=", RMSE)

# 计算测试集上的 RMSE
y_predict = RandomForest.predict(x_test)

N = y_test.size
X = range(0, N)
Y1 = np.double(y_test)
Y2 = np.double(y_predict)

GError = np.abs(Y1 - Y2)
RMSE = np.sqrt(np.dot(GError, GError) / N)

print("RMSE_test=", RMSE)

import matplotlib.pyplot as plt

fig1 = plt.figure('fig1')
plt.plot(X, GError)
# plt.scatter(X,Y1,marker='o',c='b')
# plt.scatter(X,Y2,marker='o',c='r')


fig2 = plt.figure('fig2')
plt.plot(X, Y1, 'b', lw=1)
plt.plot(X, Y2, 'r+')
