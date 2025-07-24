# -*- coding: utf-8 -*-
# !/usr/bin/python
# LightGBM 是由微软开发的一种基于决策树的 梯度提升框架（Gradient Boosting Decision Tree, GBDT），具有训练速度快、内存占用低、准确率高的特点。

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

''''' lightgbm '''
# 先安装lightgbm包 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lightgbm


import lightgbm as lgb

# 将训练数据转换为 LightGBM 的 Dataset 格式，这是 LightGBM 的高效数据结构
# x_train: 特征数据（numpy array）
# label=y_train: 标签数据（目标值）
train_data = lgb.Dataset(x_train, label=y_train)

# 下面是几个可选的参数配置示例（已被注释掉），可以根据需要进行切换或调整
# 示例1：简单回归模型，设置叶子数和目标函数
# param = {'num_leaves': 10, 'num_trees': 50, 'objective': 'regression'}

# 示例2：更复杂的回归模型，控制树深度、学习率、分箱数等
# param = {'num_leaves':150, 'objective':'regression','max_depth':7,'learning_rate':.05,'max_bin':200}

# 当前使用的参数配置：指定任务为回归任务
# objective: 指定任务类型为 'regression'，即回归任务
# verbose: 控制输出信息，-1 表示不输出训练日志
param = {'objective': 'regression', 'verbose': -1}

# 可选：指定评估指标（metric），例如 AUC、对数损失等（用于训练过程中监控）
# param['metric'] = ['auc', 'binary_logloss']

# 使用指定参数训练 LightGBM 模型
# train_data: 训练数据集（LightGBM Dataset 格式）
# gbm: 训练完成后的模型对象
gbm = lgb.train(param, train_data)

# 使用训练好的模型对训练集进行预测
# 返回的是每个样本的预测值（连续值，用于回归任务）
y_predict = gbm.predict(x_train)


''''' lightgbm '''
# 计算训练集上的 RMSE
N = y_train.size
X = range(0, N)
Y1 = np.double(y_train)
Y2 = np.double(y_predict)

GError = np.abs(Y1 - Y2)
RMSE = np.sqrt(np.dot(GError, GError) / N)
print("RMSE_train=", RMSE)

# 在测试集上进行预测
y_predict = gbm.predict(x_test)

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
