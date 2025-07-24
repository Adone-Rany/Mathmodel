# -*- coding: utf-8 -*-
# !/usr/bin/python

from __future__ import division

import numpy as np
from sklearn.model_selection import train_test_split

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

''''' AdaBoost '''
import sklearn.ensemble as ske

AdaBoost = ske.AdaBoostRegressor()
AdaBoost.fit(x_train, y_train)
# 在训练集上进行预测
y_predict = AdaBoost.predict(x_train)
''''' AdaBoost '''
# 计算训练集上的 RMSE
N = y_train.size
X = range(0, N)
Y1 = np.double(y_train)
Y2 = np.double(y_predict)

GError = np.abs(Y1 - Y2)
RMSE = np.sqrt(np.dot(GError, GError) / N)
print("RMSE_train=", RMSE)

# 在测试集上进行预测
y_predict = AdaBoost.predict(x_test)
# 计算测试集上的 RMSE
N = y_test.size
X = range(0, N)
Y1 = np.double(y_test)
Y2 = np.double(y_predict)

GError = np.abs(Y1 - Y2)
RMSE = np.sqrt(np.dot(GError, GError) / N)

print("RMSE_test=", RMSE)

import matplotlib.pyplot as plt
# 绘制误差图
fig1 = plt.figure('fig1')
plt.plot(X, GError)
# plt.scatter(X,Y1,marker='o',c='b')
# plt.scatter(X,Y2,marker='o',c='r')

# 绘制实际值和预测值对比图
fig2 = plt.figure('fig2')
plt.plot(X, Y1, 'b', lw=1)
plt.plot(X, Y2, 'r+')
plt.show()