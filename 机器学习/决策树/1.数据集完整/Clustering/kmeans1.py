# -*- coding: utf-8 -*-
# !/usr/bin/python

from __future__ import division

import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# import  scipy.cluster.vq as km
import collections as co

data = []
labels = []

with open("data.txt") as ifile:
    for line in ifile:
        tokens = line.strip().split('\t')
        if tokens != ['']:
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
x = np.array(data)
y = np.array(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

# 获取训练集中最大的类别标签值，作为聚类数 k1
k1 = int(max(y_train))

''''' kmeans '''
# 导入 KMeans 聚类模型
from sklearn.cluster import KMeans

# 创建 KMeans 聚类器，聚类数量为 k1，随机种子为 9
kmeans = KMeans(n_clusters=k1, random_state=9)

# 对训练集进行聚类，返回每个样本所属的聚类标签，是从 0 开始的整数，一直到 n_clusters - 1
y_predict = kmeans.fit_predict(x_train)

# 获取聚类中心点坐标
res = kmeans.cluster_centers_
''''' kmeans '''


# 将真实标签减 1，使其从 0 开始（适用于聚类标签对齐）
Y1 = np.double(y_train) - 1

# 将预测标签偏移 k1，用于后续的类别匹配
Y2 = y_predict + k1

# 使用 collections.Counter 来统计每个聚类中出现最多的类别
con = co.Counter()

# 遍历每个聚类（从 k1 到 2*k1 - 1）
for k in range(k1, 2 * k1):
    # 提取真实标签中属于当前聚类的样本，找出 Y2 中等于 k 的位置，然后从 Y1 中取出这些位置的值
    s = Y1[Y2 == k]
    # 统计该聚类中最常见的类别
    con.update(s)
    # 获取该聚类中出现次数最多的类别
    s1 = con.most_common(1)[0][0]
    # 将该聚类的所有预测标签替换为最匹配的真实类别标签
    Y2[Y2 == k] = s1
    # 清空计数器以便下一轮使用
    con.clear()
    # 打印当前聚类最匹配的真实类别
    print(s1)

# 初始化计数器，用于统计预测正确的样本数量
Size = 0
# 遍历所有训练样本
for k in range(0, Y1.size):
    # 如果预测标签与真实标签一致，则计数器加一
    if (Y1[k] == Y2[k]):
        Size += 1

# 计算准确率（百分比）
accuracy = Size / Y1.size * 100
# 打印准确率
print(accuracy)


# 创建 Y_1 的副本，用于后续绘图颜色映射
Y_1 = Y2 * 1

# 将 Y1（真实标签）转换为整数类型并赋值给 Y_1
for k in range(0, Y1.size):
    Y_1[k] = int(Y1[k])

# colors = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1],[0.8,0.2,0.1] )[i] for i in y_predict])

# 生成随机颜色数组，用于不同类别的可视化
# 每个类别对应一个 RGB 颜色，共 k1 个类别，每个颜色是 3 个 0~1 之间的随机数
c = np.random.random([k1, 3])

# 创建第一个子图（左侧），用于显示真实标签的样本分布
plt.subplot(1, 2, 1)
# 为每个样本分配颜色：根据 Y_1 中的类别编号，从颜色数组 c 中取出对应颜色
colors = ([c[i] for i in Y_1])
# 绘制散点图，仅使用前两个特征（第 0 列和第 1 列）作为 X 和 Y 轴
plt.scatter(x_train[:, 0], x_train[:, 1], c=colors)

# 创建第二个子图（右侧），用于显示 KMeans 聚类后的标签分布
plt.subplot(1, 2, 2)
# 同样根据 Y2 中的类别编号，从颜色数组 c 中取出对应颜色
colors = ([c[i] for i in Y2])
# 绘制散点图，仅使用前两个特征（第 0 列和第 1 列）作为 X 和 Y 轴
plt.scatter(x_train[:, 0], x_train[:, 1], c=colors)

# 在聚类中心绘制一个大圆圈（空心）表示聚类中心位置
plt.scatter(res[:, 0], res[:, 1], marker='o', s=500, linewidths=1, c='none')
# 在聚类中心再绘制一个“X”标记，更清晰地标出中心点
plt.scatter(res[:, 0], res[:, 1], marker='x', s=500, linewidths=1)

# 显示图像
plt.show()


# maxy=int(max(y_predict))
# miny=int(min(y_predict))
# K=maxy-miny+1
# c=np.random.random([K,3])
#
# Y1=np.double(y_test)
# Y2=np.double(y_predict)
#
# fig = plt.figure(figsize=plt.figaspect(0.5))

#
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('real')
# ax.view_init(30, 60)
#
# for k in range(0,Y1.size):
#    co=(c[np.mod(int(Y1[k]),4),0],c[np.mod(int(Y1[k]),4),1],c[np.mod(int(Y1[k]),4),2])
#    ax.scatter(x_test[k,0],x_test[k,1],x_test[k,2],color=co)
#
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('svm')
# ax.view_init(30, 60)
#
#
# for k in range(0,Y2.size):
#    co=(c[np.mod(int(Y2[k]),4),0],c[np.mod(int(Y2[k]),4),1],c[np.mod(int(Y2[k]),4),2])
#    ax.scatter(x_test[k,0],x_test[k,1],x_test[k,2],color=co)
#
##accuracy =   np.size(find(predictLabel == testLabel))/np.size(testLabel)*100
# Size=0
# for k in range(0,Y1.size):
#    if(Y1[k]==Y2[k]):
#        Size+=1
#        
# accuracy =   Size/Y1.size*100
#
# print accuracy
