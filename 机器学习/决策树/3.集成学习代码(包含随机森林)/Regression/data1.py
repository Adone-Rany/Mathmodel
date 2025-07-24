# -*- coding: utf-8 -*-
#!/usr/bin/python

from __future__ import division
import numpy as np  


# 定义一个非线性函数，用于生成回归目标值 z
def fun(y, k, c):
    """
    输入：
        y: 一个长度为 k 的输入向量（样本特征）
        k: 特征维度
        c: 系数数组，长度为 k+1
    输出：
        z: 根据多项式和系数生成的非线性组合值（带噪声）
           z=11+y[0]*1+y[1]*2+y[2]*3+np.random.random()*0.1
    """

    # 初始值为常数项 c[0]，并添加随机噪声
    z = c[0] + np.random.random() * 0.1

    # 一次项：c[1] * y[0] + c[2] * y[1] + ... + c[k] * y[k-1]
    for s in range(0, k):
        z += c[s + 1] * y[s]

    # 二次项：c[1] * y[0]^2 + c[2] * y[1]^2 + ... + c[k] * y[k-1]^2
    for s in range(0, k):
        z += c[s + 1] * y[s] ** 2

    # 交叉项：c[1] * y[0]*y[1] + c[2] * y[1]*y[2] + ... + c[k-1] * y[k-2]*y[k-1]
    for s in range(0, k - 1):
        z += c[s + 1] * y[s] * y[s + 1]

    # 可选的更高阶项（当前被注释掉）
    #    for s in range(0,k-2):
    #        z+=c[s]*y[s]*y[s+1]*y[s+2]**2
    #
    #    for s in range(0,k-3):
    #        z+=c[s+1]*y[s]*y[s+1]*y[s+2]*y[s+3]

    return z


# 每类生成 4000 个样本
N = 4000
# 每个样本有 4 个特征（维度）
k = 4
# 类别数（这里只生成一类，用于回归）
J = 1

# 协方差矩阵：单位矩阵，表示特征之间不相关
sigma = np.eye(k)

# 初始化特征向量 y（长度为 k）
y = np.zeros(k)

# 打开 data.txt 文件用于写入数据（会清空已有内容）
f = open('data.txt', 'w')

# 生成 k+1 个随机系数，c[0] 是常数项，c[1]~c[k] 是特征系数
c = np.random.random(k + 1)
# 将常数项 c[0] 放大，增加偏移量影响
c[0] *= 30

# 循环生成每个类别的数据（这里只循环一次）
for j in range(0, J):
    # 随机生成均值向量 mu，范围 [0, 30]
    mu = np.random.random(k) * 30

    # 生成 N 个服从多维正态分布的样本 A
    A = np.random.multivariate_normal(mu, sigma, N)

    # 遍历每个样本 n
    for n in A:
        mm = 0
        # 写入特征值到文件，同时保存到 y 数组中
        for s in n:
            f.write(str(s) + '\t')
            y[mm] = s
            mm = mm + 1

        # 调用函数 fun 计算目标值 z
        z = fun(y, k, c)

        # 写入目标值 z 到文件，并换行
        f.write(str(z))
        f.write('\r\n')  # Windows 风格换行符

# 关闭文件
f.close()
