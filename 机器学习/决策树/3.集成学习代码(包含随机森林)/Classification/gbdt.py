# -*- coding: utf-8 -*-
# !/usr/bin/python
# GBDT（Gradient Boosting Decision Tree，梯度提升决策树）是一种强大的集成学习算法，广泛应用于分类和回归任务。它的核心思想是通过迭代构建多个弱决策树模型，并通过梯度下降优化损失函数，逐步修正模型的残差（预测误差）。

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
y = np.array(labels).astype(int)
# 将数据集划分为训练集和测试集，测试集占比 10%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

''''' gbdt '''
import sklearn.ensemble as ske

# 创建 GBDT 分类器对象
# learning_rate: 学习率，控制每一步的权重更新幅度，较小的值需要更多的弱学习器来补偿
# n_estimators: 弱分类器的最大迭代次数（即决策树的数量）
# subsample: 用于训练每个弱学习器的样本比例（subsample=1 表示使用全部样本）
# min_samples_split: 内部节点再划分所需最小样本数
# min_samples_leaf: 叶子节点最少样本数
# max_depth: 每棵决策树的最大深度，值越小越不容易过拟合
# init: 可选的初始估计器，若为 None 则使用默认的初始估计
# random_state: 控制随机性，用于数据划分和随机选择特征等
# max_features: 每次分裂时考虑的最大特征数量，None 表示使用所有特征
# verbose: 控制日志输出的详细程度，0 表示不输出日志信息
# max_leaf_nodes: 最大叶子节点数，用于限制树的生长
# warm_start: 若为 True，则重用之前训练的模型作为初始化
gbdt = ske.GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=20,
    subsample=1,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,
    init=None,
    random_state=None,
    max_features=None,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False)

# 使用训练集训练 GBDT 模型
gbdt.fit(x_train, y_train)

# 使用训练好的模型对测试集进行预测
y_predict = gbdt.predict(x_test)


''''' gbdt '''

''''' accuracy_train '''
# 对训练集进行预测
y_predict_train = gbdt.predict(x_train)
# 将真实标签和预测标签转换为浮点类型，便于后续比较
Y1 = np.double(y_train)
Y2 = np.double(y_predict_train)

# 初始化计数器
Size = 0
# 遍历所有训练样本，统计预测正确的数量
for k in range(0, Y1.size):
    if (Y1[k] == Y2[k]):
        Size += 1

# 计算训练集准确率（百分比）
accuracy = Size / Y1.size * 100
print('accuracy_train', accuracy)

''''' accuracy_test '''
# 将测试集的真实标签和预测标签转换为浮点类型
Y1 = np.double(y_test)
Y2 = np.double(y_predict)

# accuracy =   np.size(find(predictLabel == testLabel))/np.size(testLabel)*100
# 初始化计数器
Size = 0
# 遍历所有测试样本，统计预测正确的数量
for k in range(0, Y1.size):
    if (Y1[k] == Y2[k]):
        Size += 1

# 计算测试集准确率（百分比）
accuracy = Size / Y1.size * 100
print('accuracy_test', accuracy)

''''''''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建一个绘图窗口，设置大小
fig = plt.figure(figsize=plt.figaspect(0.5))

# 获取预测标签的最大值和最小值，用于生成颜色
maxy = int(max(y_predict))
miny = int(min(y_predict))
# 根据类别数量生成随机颜色
K = maxy - miny + 1
c = np.random.random([K, 3])

# 创建第一个 3D 子图，用于显示真实标签的样本分布
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('real')  # 设置标题为“真实标签”
ax.view_init(30, 60)  # 设置视角

# 遍历测试集，根据真实标签绘制 3D 散点图
for k in range(0, Y1.size):
    # 根据标签值选择颜色
    co = (c[np.mod(int(Y1[k]), 4), 0], c[np.mod(int(Y1[k]), 4), 1], c[np.mod(int(Y1[k]), 4), 2])
    ax.scatter(x_test[k, 0], x_test[k, 1], x_test[k, 2], color=co)

# 创建第二个 3D 子图，用于显示决策树预测的标签分布
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('DecisionTree')  # 设置标题为“决策树预测”
ax.view_init(30, 60)  # 设置视角

# 遍历测试集，根据预测标签绘制 3D 散点图
for k in range(0, Y2.size):
    # 根据预测标签值选择颜色
    co = (c[np.mod(int(Y2[k]), 4), 0], c[np.mod(int(Y2[k]), 4), 1], c[np.mod(int(Y2[k]), 4), 2])
    ax.scatter(x_test[k, 0], x_test[k, 1], x_test[k, 2], color=co)

# 显示绘图结果
plt.show()