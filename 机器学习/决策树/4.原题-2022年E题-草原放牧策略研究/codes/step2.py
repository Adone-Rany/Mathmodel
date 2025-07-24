#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    将机器学习算法(以随机森林为例)应用到step1生成的数据集中
    生成模型，保存到./data/step2里
    
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

path_models = os.path.join("./data/", 'step2', 'models_dict.pkl')  # 模型存储路径
models.savepkl(path_models)
