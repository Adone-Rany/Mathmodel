# -*- coding: utf-8 -*-
#!/usr/bin/python

# 这段代码的主要目的是生成一个用于多分类任务的合成数据集，并将其保存到文件 data.txt 中

from __future__ import division
import numpy as np

# 每类生成 2000 个样本
N = 2000
# 每个样本有 9 个特征（维度）
k = 9
# 总共有 4 个类别
J = 4

sigma = np.eye(k)
# 执行这一句会清空原有代码
f = open('data.txt', 'w')

for j in range(0, J):
    # 随机生成一个均值向量mu，长度为k，值在[0, k]之间
    mu = np.random.random(k) * k
    # 生成N个服从多维正态分布的样本
    A = np.random.multivariate_normal(mu, sigma, N)

    for n in A:
        for s in n:
            f.write(str(s) + '\t')
        f.write(str(j + 1))
        f.write('\r\n')  # \r\n为换行符

f.close()
