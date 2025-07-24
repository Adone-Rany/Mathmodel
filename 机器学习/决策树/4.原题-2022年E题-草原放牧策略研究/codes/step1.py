#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    把原始数据处理成机器学习数据集
    保存为.pkl类型, 存到./data/step1中
    
"""

import os

import numpy as np

import datetime

import xlrd
import pickle as pk


def xls_to_list(file_path):
    """
    将Excel文件读取为列表格式的数据
    参数:
        file_path: Excel文件路径
    返回:
        包含Excel数据的二维列表
    """
    # 打开Excel工作簿
    workbook = xlrd.open_workbook(file_path)

    # 选择第一个工作表（索引为0）
    worksheet = workbook.sheet_by_index(0)

    # 初始化数据列表
    data = []

    # 遍历工作表中的每一行
    for row_idx in range(worksheet.nrows):
        # 获取当前行的所有单元格值
        row_data = worksheet.row_values(row_idx)
        # 将行数据添加到总数据列表中
        data.append(row_data)

    # 返回完整的数据列表
    return data


def make_data_set(features, soil_moisture, span, deep):
    """
    根据特征数据和标签数据生成机器学习数据集
    参数:
        features: 特征数据列表
        soil_moisture: 土壤湿度标签数据（包含时间信息）
        span: 预测跨度（'1to2'表示预测下一年，'1to3'表示预测后两年）
        deep: 土壤深度（'10','40','100','200'表示不同深度）
    返回:
        inputs, labels, inputs_train, labels_train, inputs_test, labels_test:
        分别为完整数据集、训练集和测试集的输入和标签
    """

    # 初始化数据列表
    data = []

    # 定义土壤深度到索引的映射关系
    deep2index = {'10': 0, '40': 1, '100': 2, '200': 3}

    # 定义预测跨度到年数的映射关系
    span2year = {'1to2': 1, '1to3': 2}

    # 遍历所有特征数据
    for line in features:
        # 计算目标时间：原始时间 + 预测跨度年数
        date_label = datetime.datetime(
            year=int(line[-1]) + span2year[span],  # 年份加上预测跨度
            month=int(line[-2]),  # 月份保持不变
            day=1  # 日期设为1号
        )

        # 检查目标时间是否在土壤湿度数据中存在
        if date_label in soil_moisture:
            # 找到对应时间在土壤湿度数据中的索引位置
            index = np.where(date_label == soil_moisture)

            # 复制当前特征行数据
            line_copy = line * 1

            # 将特征数据转换为浮点数（除了最后一个时间字段）
            line_copy = [float(item) for item in line_copy[:-1]]

            # 将最后一个特征替换为对应深度的土壤湿度值
            line_copy[-1] = float(soil_moisture[index[0][0], deep2index[deep]])

            # 在末尾添加时间标签
            line_copy.append(date_label)

            # 将处理后的数据添加到数据列表中
            data.append(line_copy)

    # 将数据列表转换为NumPy数组
    data = np.array(data)

    # 划分训练集和测试集
    data_train = data[24:, :]  # 2019年及之前的數據作為訓練集

    # 根据预测跨度确定测试集范围
    # 对于1to2: 2020年的数据；对于1to3: 2021年的数据
    data_test = data[(24 - span2year[span] * 12):(36 - span2year[span] * 12), :]

    # 分离输入特征和标签
    inputs = data[:, :-2]  # 所有特征（除去最后两个字段）
    labels = data[:, -2:]  # 最后两个字段（土壤湿度值和时间标签）

    # 分离训练集的输入和标签
    inputs_train = data_train[:, :-2]
    labels_train = data_train[:, -2:]

    # 分离测试集的输入和标签
    inputs_test = data_test[:, :-2]
    labels_test = data_test[:, -2:]

    # 返回所有数据集
    return inputs, labels, inputs_train, labels_train, inputs_test, labels_test


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

# 定义三个Excel文件路径
file3 = "./data/raw_data/附件3、土壤湿度2022—2012年.xls"   # 土壤湿度数据（作为标签）
file4 = "./data/raw_data/附件4、土壤蒸发量2012—2022年.xls"  # 土壤蒸发量数据（作为特征）

# 读取数据并跳过前4行（2022年）
data3_label = xls_to_list(file3)[4:]  # 土壤湿度数据，作为预测目标（标签）
data4 = xls_to_list(file4)[4:]        # 土壤蒸发量数据，作为输入特征

# 处理气象数据（附件8）
path8 = "./data/raw_data/附件8、锡林郭勒盟气候2012-2022/"  # 气候数据目录路径
data8 = []  # 存储合并后的气象信息，作为特征

# 获取目录下所有文件名，并去掉最后一个（2022年）
file_list = os.listdir(path8)[:-1]

# 将文件列表倒序排列（按时间倒序）
file_list_reverse = [file_list[i] for i in range(len(file_list) - 1, -1, -1)]

# 遍历所有气象数据文件并读取内容
for file_name in file_list_reverse:
    file8 = os.path.join(path8, file_name)
    # 读取每个文件的内容，跳过第一行标题行
    data8.extend(xls_to_list(file8)[1:])

# 数据预处理阶段
features = []       # 存储所有特征数据
soil_moisture = []  # 存储土壤湿度标签数据

# 遍历所有数据样本，进行特征和标签的整合
for k in range(len(data4)):
    # 构造特征向量：包括蒸发量的部分字段 + 气象数据的多个字段 + 蒸发量的时间信息
    # 去掉了积雪深度和瞬时风等字段
    data = data4[k][4:6] + data8[k][6:-9] + data8[k][-8:-2] + data4[k][0:2]
    features.append(data)  # 添加到特征列表

    # 构造土壤湿度标签数据：包括湿度数据 + 时间信息
    data = data3_label[k][4:] + data3_label[k][0:2]
    # 将时间信息转换为datetime对象并添加到数据末尾
    data.append(datetime.datetime(year=int(data[-1]), month=int(data[-2]), day=1))
    soil_moisture.append(data)  # 添加到标签列表

# 将土壤湿度数据转换为NumPy数组以便处理
soil_moisture = np.array(soil_moisture)

# 生成多种组合的数据集（不同的预测跨度和土壤深度），存为字典类型数据

# 生成跨度为1年（用第一年预测第二年），深度为10cm的数据集
inputs, labels, inputs_train, labels_train, inputs_test, labels_test = make_data_set(
    features, soil_moisture, span='1to2', deep='10')
data_set_1to2_10 = {
    'inputs': inputs, 'labels': labels,
    'inputs_train': inputs_train, 'labels_train': labels_train,
    'inputs_test': inputs_test, 'labels_test': labels_test,
}

# 生成跨度为1年（用第一年预测第二年），深度为40cm的数据集
inputs, labels, inputs_train, labels_train, inputs_test, labels_test = make_data_set(
    features, soil_moisture, span='1to2', deep='40')
data_set_1to2_40 = {
    'inputs': inputs, 'labels': labels,
    'inputs_train': inputs_train, 'labels_train': labels_train,
    'inputs_test': inputs_test, 'labels_test': labels_test,
}

# 生成跨度为1年（用第一年预测第二年），深度为100cm的数据集
inputs, labels, inputs_train, labels_train, inputs_test, labels_test = make_data_set(
    features, soil_moisture, span='1to2', deep='100')
data_set_1to2_100 = {
    'inputs': inputs, 'labels': labels,
    'inputs_train': inputs_train, 'labels_train': labels_train,
    'inputs_test': inputs_test, 'labels_test': labels_test,
}

# 生成跨度为1年（用第一年预测第二年），深度为200cm的数据集
inputs, labels, inputs_train, labels_train, inputs_test, labels_test = make_data_set(
    features, soil_moisture, span='1to2', deep='200')
data_set_1to2_200 = {
    'inputs': inputs, 'labels': labels,
    'inputs_train': inputs_train, 'labels_train': labels_train,
    'inputs_test': inputs_test, 'labels_test': labels_test,
}

# 生成跨度为2年（用第一年预测第三年），深度为10cm的数据集
inputs, labels, inputs_train, labels_train, inputs_test, labels_test = make_data_set(
    features, soil_moisture, span='1to3', deep='10')
data_set_1to3_10 = {
    'inputs': inputs, 'labels': labels,
    'inputs_train': inputs_train, 'labels_train': labels_train,
    'inputs_test': inputs_test, 'labels_test': labels_test,
}

# 生成跨度为2年（用第一年预测第三年），深度为40cm的数据集
inputs, labels, inputs_train, labels_train, inputs_test, labels_test = make_data_set(
    features, soil_moisture, span='1to3', deep='40')
data_set_1to3_40 = {
    'inputs': inputs, 'labels': labels,
    'inputs_train': inputs_train, 'labels_train': labels_train,
    'inputs_test': inputs_test, 'labels_test': labels_test,
}

# 生成跨度为2年（用第一年预测第三年），深度为100cm的数据集
inputs, labels, inputs_train, labels_train, inputs_test, labels_test = make_data_set(
    features, soil_moisture, span='1to3', deep='100')
data_set_1to3_100 = {
    'inputs': inputs, 'labels': labels,
    'inputs_train': inputs_train, 'labels_train': labels_train,
    'inputs_test': inputs_test, 'labels_test': labels_test,
}

# 生成跨度为2年（用第一年预测第三年），深度为200cm的数据集
inputs, labels, inputs_train, labels_train, inputs_test, labels_test = make_data_set(
    features, soil_moisture, span='1to3', deep='200')
data_set_1to3_200 = {
    'inputs': inputs, 'labels': labels,
    'inputs_train': inputs_train, 'labels_train': labels_train,
    'inputs_test': inputs_test, 'labels_test': labels_test,
}

# 将所有生成的数据集组合成一个大的字典
data_sets = {
    'data_set_1to2_10': data_set_1to2_10,
    'data_set_1to2_40': data_set_1to2_40,
    'data_set_1to2_100': data_set_1to2_100,
    'data_set_1to2_200': data_set_1to2_200,
    'data_set_1to3_10': data_set_1to3_10,
    'data_set_1to3_40': data_set_1to3_40,
    'data_set_1to3_100': data_set_1to3_100,
    'data_set_1to3_200': data_set_1to3_200,
}

# 定义保存路径
path_dict = os.path.join("./data/", 'step1', 'data_sets.pkl')  # 数据集存储路径

# 调用函数保存所有数据集到指定路径
save_file(path_dict, data_sets)  # 存储dict数据, 到path_dict 路径
