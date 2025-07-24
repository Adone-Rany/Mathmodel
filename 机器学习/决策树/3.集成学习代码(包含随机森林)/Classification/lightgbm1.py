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
y = np.array(labels).astype(int)
# 将数据集划分为训练集和测试集，测试集占比 10%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

''''' lightgbm '''
# 先安装 lightgbm 包（若尚未安装）
# 可使用命令：pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lightgbm

# 将标签从 1-based 转换为 0-based（LightGBM 的多分类任务通常使用 0-based 标签）
y_train = np.double(y_train) - 1
y_test = np.double(y_test) - 1

# 获取类别数量，用于设置 num_class 参数
k1 = int(max(y_train) + 1)  # +1 是因为标签已从 0 开始编号

# 导入 LightGBM 模块
import lightgbm as lgb

# 将训练数据转换为 LightGBM 的 Dataset 格式，这是 LightGBM 的高效数据结构
train_data = lgb.Dataset(x_train, label=y_train)

# 设置模型参数（param）
# objective: 任务目标，这里为多分类任务 'multiclass'
# num_class: 类别数量，必须指定
# verbose: 控制输出信息，-1 表示不输出训练日志
param = {'objective': 'multiclass', 'num_class': k1, 'verbose': -1}

# 可选参数示例（注释掉的）：
# param = {'num_leaves': 10, 'num_trees': 50, 'objective': 'regression'}
# param = {'num_leaves':150, 'objective':'regression','max_depth':7,'learning_rate':.05,'max_bin':200}
# param = {'objective': 'binary'}  # 二分类任务
# param['metric'] = ['auc', 'binary_logloss']  # 指定评估指标

# 训练 LightGBM 模型
# train_data: 训练数据集
# param: 模型参数
gbm = lgb.train(param, train_data)

# 使用训练好的模型对测试集进行预测
# predict 返回的是每个样本在各个类别上的预测概率或得分
y_predict = gbm.predict(x_test)

# 将预测结果转换为类别标签
# 对于每个样本，取概率最高的类别作为预测结果
# 即：找出每行中最大值的索引作为预测类别
y_predict = [list(x).index(max(x)) for x in y_predict]

''''' lightgbm '''

''''' accuracy_train '''
# 对训练集进行预测
y_predict_train = gbm.predict(x_train)

# 同样将预测结果转为类别标签
y_predict_train = [list(x).index(max(x)) for x in y_predict_train]

# 将真实训练标签和预测标签转为浮点类型，便于比较
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

# from mpl_toolkits.mplot3d import Axes3D
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
