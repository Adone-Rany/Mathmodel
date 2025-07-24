#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    读取step1数据集和step2的模型，将模型应用到测试集中，
    生成结果保存到./data/step3中
    
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

path_models = os.path.join("./data/", 'step2', 'models_dict.pkl')  # 模型存储路径
models.loadpkl(path_models)  # 载入模型

data_set_names = ['data_set_1to2_10',
                  'data_set_1to2_40',
                  'data_set_1to2_100',
                  'data_set_1to2_200',
                  'data_set_1to3_10',
                  'data_set_1to3_40',
                  'data_set_1to3_100',
                  'data_set_1to3_200', ]
data_set_names = models.data_dict['info']['data_set_names']  # 和上面的列表一样，为方便看就不删了

models_names = ['model_1to2_10',
                'model_1to2_40',
                'model_1to2_100',
                'model_1to2_200',
                'model_1to3_10',
                'model_1to3_40',
                'model_1to3_100',
                'model_1to3_200', ]
models_names = models.data_dict['info']['models_names']  # 和上面的列表一样，为方便看就不删了

outputs_list = []
abs_error_list = []
labels_list = []
relative_error_list = []

for k in range(8):
    model_ML = models.data_dict['model'][models_names[k]]  # 载入模型

    inputs_test = data_sets[data_set_names[k]]['inputs_test']  # 载入数据
    labels_test = data_sets[data_set_names[k]]['labels_test']

    outputs = model_ML.predict(inputs_test)

    abs_error = abs(outputs - labels_test[:, 0])

    relative_error = abs((outputs - labels_test[:, 0]) / labels_test[:, 0])

    labels_list.append(labels_test[:, 0])
    outputs_list.append(outputs)
    abs_error_list.append(abs_error)
    relative_error_list.append(relative_error)

head_list = ['2020-10cm', '2020-40cm', '2020-100cm', '2020-200cm', '2021-10cm', '2021-40cm', '2021-100cm',
             '2021-200cm', ]

import xlwt

book = xlwt.Workbook(encoding="utf-8")

# 添加一个sheet页
sheet1 = book.add_sheet('labels')
# 将列表数据写入sheet页
for i, col in enumerate(head_list):
    sheet1.write(0, i, col)
for i, row in enumerate(labels_list):
    for j, col in enumerate(row):
        sheet1.write(j + 1, i, col)

# 添加一个sheet页
sheet2 = book.add_sheet('outputs')
# 将列表数据写入sheet页
for i, col in enumerate(head_list):
    sheet2.write(0, i, col)
for i, row in enumerate(outputs_list):
    for j, col in enumerate(row):
        sheet2.write(j + 1, i, col)

# 添加一个sheet页
sheet3 = book.add_sheet('abs_error')
# 将列表数据写入sheet页
for i, col in enumerate(head_list):
    sheet3.write(0, i, col)
for i, row in enumerate(abs_error_list):
    for j, col in enumerate(row):
        sheet3.write(j + 1, i, col)

# 添加一个sheet页
sheet4 = book.add_sheet('relatives_error')
# 将列表数据写入sheet页
for i, col in enumerate(head_list):
    sheet4.write(0, i, col)
for i, row in enumerate(relative_error_list):
    for j, col in enumerate(row):
        sheet4.write(j + 1, i, col)

path_book = os.path.join("./data/", 'step3', 'reluts.xls')  # 模型存储路径
# 保存到文件
os.makedirs(os.path.dirname(path_book), exist_ok=True)
book.save(path_book)
