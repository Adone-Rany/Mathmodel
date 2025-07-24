# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import division
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt    
#先安装xgboost包 pip install -i https://pypi.tuna.tsinghua.edu.cn/simple xgboost
import xgboost as xgb

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


# 计算训练集准确率
Y1 = np.double(y_train)  # 真实标签
Y2 = np.rint(y_predict_train)  # 对预测值四舍五入取整

Size = 0
for k in range(0, Y1.size):
    if Y1[k] == Y2[k]:
        Size += 1

accuracy = Size / Y1.size * 100  # 计算准确率百分比
print('accuracy_train', accuracy)


# 计算测试集准确率
Y1 = np.double(y_test)  # 真实标签
Y2 = np.rint(y_predict)  # 对预测值四舍五入取整

Size = 0
for k in range(0, Y1.size):
    if Y1[k] == Y2[k]:
        Size += 1

accuracy = Size / Y1.size * 100  # 计算准确率百分比
print('accuracy_test', accuracy)


# 绘图部分：3D 散点图展示真实标签与预测标签的对比

fig = plt.figure(figsize=plt.figaspect(0.5))  # 创建一个图形窗口，设置宽高比

# 获取预测值中的最大和最小类别，用于生成随机颜色
maxy = int(max(y_predict))
miny = int(min(y_predict))
K = maxy - miny + 1
c = np.random.random([K, 3])  # 生成 K 种随机颜色用于绘图

# 第一个子图：展示真实标签的 3D 散点图
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('real')  # 图标题
ax.view_init(30, 60)  # 设置视角角度

for k in range(0, Y1.size):
    # 根据真实标签选择颜色（取模避免越界）
    co = (c[np.mod(int(Y1[k]), 4), 0], c[np.mod(int(Y1[k]), 4), 1], c[np.mod(int(Y1[k]), 4), 2])
    ax.scatter(x_test[k, 0], x_test[k, 1], x_test[k, 2], color=co)  # 绘制散点

# 第二个子图：展示预测标签的 3D 散点图
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Xgboost')  # 图标题
ax.view_init(30, 60)  # 设置视角角度

for k in range(0, Y2.size):
    # 根据预测标签选择颜色
    co = (c[np.mod(int(Y2[k]), 4), 0], c[np.mod(int(Y2[k]), 4), 1], c[np.mod(int(Y2[k]), 4), 2])
    ax.scatter(x_test[k, 0], x_test[k, 1], x_test[k, 2], color=co)  # 绘制散点

plt.show()  # 显示图形
