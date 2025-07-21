import numpy as np

# -----------1.数据标准化-----------
# 进行正向和逆向标准化的函数
def normalize_data(data):
    """
    对每一列进行标准化处理。对于正向指标，使用正向标准化公式， 8对于逆向指标，使用逆向标准化公式。 9"""
    normalized_data = np.zeros_like(data, dtype=float)
    n, m = data.shape  # n为样本数量，m为指标数量

    for j in range(m):
        column_data = data[:, j]
        if np.all(column_data == np.max(column_data)):  # 如果列数据都是相同值
            normalized_data[:, j] = 1
        elif j == 0:  # 假设第1列为正向指标
            normalized_data[:, j] = (column_data - np.min(column_data)) / (np.max(column_data) - np.min(column_data))
        else:  # 其他列为逆向指标
            normalized_data[:, j] = (np.max(column_data) - column_data) / (np.max(column_data) - np.min(column_data))
    return normalized_data


# -----------2.计算比例值-----------
def calculate_proportions(normalized_data):
    """
    计算比例值矩阵
    """
    proportions = normalized_data / np.sum(normalized_data, axis=0)
    return proportions


# -----------3.计算信息熵-----------
def calculate_entropy(proportions):
    """ 计算信息熵值
    """
    m, n = proportions.shape
    k = 1 / np.log(m)  # 根据熵权法公式
    entropy = np.zeros(n)
    for j in range(n):
        column_data = proportions[:, j]
        # 计算该列的信息熵
        entropy[j] = -k * np.sum(column_data * np.log(column_data + 1e-9))  # 防止log(0)的错误
    return entropy


# -----------4.计算权重-----------
def calculate_weights(entropy):
    """
    根据熵值计算权重
    """
    d = 1 - entropy  # 计算效用值
    weights = d / np.sum(d)  # 计算每个指标的权重
    return weights


# -----------5.数据输入部分-----------
# 假设有3个策略，3个指标的数据
data = np.array([
    [100, 4, 500],  # 策略1：市场需求，竞争态势，产品成本
    [150, 2, 450],  # 策略2
    [120, 3, 550]  # 策略3
])

# -----------6.计算标准化数据-----------
normalized_data = normalize_data(data)
print("标准化后的数据：")
print(normalized_data)

# -----------7.计算比例值-----------
proportions = calculate_proportions(normalized_data)
print("\n比例值矩阵：")
print(proportions)

# -----------8.计算信息熵-----------
entropy = calculate_entropy(proportions)
print("\n信息熵：")
print(entropy)

# -----------9.计算权重-----------
weights = calculate_weights(entropy)
print("\n指标权重：")
print(weights)

# -----------10.输出最终结果-----------
# 显示权重并做排序
strategies = ["策略1", "策略2", "策略3"]
ranking = sorted(zip(strategies, weights), key=lambda x: x[1], reverse=True)
print("\n权重排序结果：")
for i, (strategy, weight) in enumerate(ranking, 1):
    print(f"{i}.{strategy}:{weight:.4f}")
