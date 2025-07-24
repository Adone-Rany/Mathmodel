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
# 将数据集划分为训练集和测试集，测试集占比 10%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

''''' gbrt '''
import sklearn.ensemble as ske

# 创建 Gradient Boosting 回归模型对象
# loss: 损失函数类型，'squared_error' 表示使用均方误差（MSE）作为损失函数
# learning_rate: 学习率，控制每棵树对最终结果的贡献程度，值越小越需要更多树来提升性能
# n_estimators: 弱分类器（决策树）的数量，即迭代次数
# subsample: 每次训练使用的样本比例，1.0 表示使用全部样本（不进行采样）
# min_samples_split: 内部节点再划分所需的最小样本数
# min_samples_leaf: 叶子节点所需的最小样本数
# max_depth: 每棵决策树的最大深度，控制模型复杂度，防止过拟合
# init: 可选的初始估计器，若为 None 则使用默认的常数估计器
# random_state: 控制随机性，用于数据划分、特征选择等（None 表示不固定随机种子）
# max_features: 每次分裂时考虑的最大特征数量，None 表示使用所有特征
# alpha: 当使用 'huber' 或 'quantile' 损失函数时，用于控制异常值的分位数
# verbose: 控制日志输出的详细程度，0 表示不输出日志信息
# max_leaf_nodes: 可选参数，限制树的最大叶子节点数，用于控制模型复杂度
# warm_start: 若为 True，则重用前一次训练的模型作为起点，节省重复训练时间
gbrt = ske.GradientBoostingRegressor(
    loss='squared_error',
    learning_rate=0.1,
    n_estimators=100,
    subsample=1,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False)

# 使用训练集训练模型
# x_train: 训练数据特征
# y_train: 训练数据标签（目标值）
gbrt.fit(x_train, y_train)

# 使用训练好的模型对训练集进行预测
# y_predict: 预测输出结果（连续值）
y_predict = gbrt.predict(x_train)

''''' gbrt '''
# 计算训练集上的 RMSE
N = y_train.size
X = range(0, N)
Y1 = np.double(y_train)
Y2 = np.double(y_predict)

GError = np.abs(Y1 - Y2)
RMSE = np.sqrt(np.dot(GError, GError) / N)
print("RMSE_train=", RMSE)

# 在测试集上进行预测
y_predict = gbrt.predict(x_test)

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