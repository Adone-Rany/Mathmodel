#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    把step2和step3合并，不保存模型，直接输出结果
    
"""

import os

import numpy as np

import datetime

import pickle as pk


def load_file(file_path: str):
    '''
    #读取字典数据
    '''
    try:
        file = open(file_path, 'rb')
        data_dict_load = pk.load(file)
        file.close()
        return data_dict_load
    except Exception as e:
        print("!!! save mat file %s with Error: %s" % (file_path, e))
        return False


def save_file(file_path: str, data: dict):
    '''
    #存储字典数据
    '''
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file = open(file_path, 'wb')
        pk.dump(data, file)
        file.close()
        return True
    except Exception as e:
        print("!!! save mat file %s with Error: %s" % (file_path, e))
        return False


# ==========================================================================================


path_dict = os.path.join("./data/", 'step1', 'data_sets.pkl')  # 数据集存储路径

data_sets = load_file(path_dict)

data_set_names = ['data_set_1to2_10',
                  'data_set_1to2_40',
                  'data_set_1to2_100',
                  'data_set_1to2_200',
                  'data_set_1to3_10',
                  'data_set_1to3_40',
                  'data_set_1to3_100',
                  'data_set_1to3_200', ]

models_names = ['model_1to2_10',
                'model_1to2_40',
                'model_1to2_100',
                'model_1to2_200',
                'model_1to3_10',
                'model_1to3_40',
                'model_1to3_100',
                'model_1to3_200', ]

import class_model

# 调用model类存储模型
models = class_model.model_v1()
models.data_dict['model_name'] = 'models_dict'
models.data_dict['info'] = {'data_set_names': data_set_names, 'models_names': models_names, }

models.data_dict['model'] = {}  # 把所有模型存到该字典中

for k in range(8):
    data_set_inputs = data_sets[data_set_names[k]]['inputs_train']
    data_set_labels = data_sets[data_set_names[k]]['labels_train'][:, 0]

    model_ML = class_model.Model_MachineLearning()
    model_ML.train(data_set_inputs, data_set_labels)  # 模型训练

    models.data_dict['model'][models_names[k]] = model_ML

# path_models = os.path.join("./data/", 'step2', 'models_dict.pkl') # 模型存储路径
# models.savepkl(path_models)


# =================================================================================

# 初始化四个列表，用于存储测试结果的各项指标
outputs_list = []        # 存储模型预测输出值
abs_error_list = []      # 存储绝对误差
labels_list = []         # 存储真实标签值
relative_error_list = [] # 存储相对误差

# 遍历8个不同的数据集和对应的模型
for k in range(8):
    # 从模型字典中获取第k个训练好的模型
    model_ML = models.data_dict['model'][models_names[k]]  # 载入模型

    # 获取第k个数据集的测试输入和测试标签
    inputs_test = data_sets[data_set_names[k]]['inputs_test']  # 载入数据
    labels_test = data_sets[data_set_names[k]]['labels_test']

    # 使用模型对测试数据进行预测
    outputs = model_ML.predict(inputs_test)

    # 计算预测值与真实值之间的绝对误差
    abs_error = abs(outputs - labels_test[:, 0])

    # 计算相对误差（绝对误差除以真实值）
    relative_error = abs((outputs - labels_test[:, 0]) / labels_test[:, 0])

    # 将各项结果添加到对应列表中
    labels_list.append(labels_test[:, 0])      # 添加真实标签
    outputs_list.append(outputs)               # 添加预测输出
    abs_error_list.append(abs_error)           # 添加绝对误差
    relative_error_list.append(relative_error) # 添加相对误差

# 定义Excel表头列表，对应8种不同深度和年份的土壤湿度数据
head_list = ['2020-10cm', '2020-40cm', '2020-100cm', '2020-200cm',
             '2021-10cm', '2021-40cm', '2021-100cm', '2021-200cm']

# 导入xlwt库用于创建和操作Excel文件
import xlwt

# 创建一个新的Excel工作簿，设置编码为utf-8
book = xlwt.Workbook(encoding="utf-8")

# 添加第一个工作表，用于存储真实标签值
sheet1 = book.add_sheet('labels')
# 写入表头
for i, col in enumerate(head_list):
    sheet1.write(0, i, col)
# 写入每列的真实标签数据
for i, row in enumerate(labels_list):
    for j, col in enumerate(row):
        sheet1.write(j + 1, i, col)

# 添加第二个工作表，用于存储模型预测值
sheet2 = book.add_sheet('outputs')
# 写入表头
for i, col in enumerate(head_list):
    sheet2.write(0, i, col)
# 写入每列的预测输出数据
for i, row in enumerate(outputs_list):
    for j, col in enumerate(row):
        sheet2.write(j + 1, i, col)

# 添加第三个工作表，用于存储绝对误差
sheet3 = book.add_sheet('abs_error')
# 写入表头
for i, col in enumerate(head_list):
    sheet3.write(0, i, col)
# 写入每列的绝对误差数据
for i, row in enumerate(abs_error_list):
    for j, col in enumerate(row):
        sheet3.write(j + 1, i, col)

# 添加第四个工作表，用于存储相对误差
sheet4 = book.add_sheet('relatives_error')
# 写入表头
for i, col in enumerate(head_list):
    sheet4.write(0, i, col)
# 写入每列的相对误差数据
for i, row in enumerate(relative_error_list):
    for j, col in enumerate(row):
        sheet4.write(j + 1, i, col)

# 定义Excel文件的保存路径
path_book = os.path.join("./data/", 'step3', 'reluts.xls')  # 模型存储路径
# 确保保存路径的目录存在，如果不存在则创建
os.makedirs(os.path.dirname(path_book), exist_ok=True)
# 将工作簿保存到指定路径
book.save(path_book)
