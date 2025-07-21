import numpy as np


# -----------1.数据标准化-----------
def normalize_data(data, criteria_types):
    """
    对数据进行标准化处理。
    :param data:原始数据矩阵，每行代表一个方案，每列代表一个指标。
    :param criteria_types:指标类型列表，1表示正向指标，-1表示逆向指标。
    :return:标准化后的数据矩阵。
    """
    normalized_data = np.zeros_like(data, dtype=float)
    n, m = data.shape  # n为方案数量，m为指标数量

    for j in range(m):
        column_data = data[:, j]
        if criteria_types[j] == 1:  # 正向指标
            normalized_data[:, j] = (column_data - np.min(column_data)) / (np.max(column_data) - np.min(column_data))
        else:  # 逆向指标
            normalized_data[:, j] = (np.max(column_data) - column_data) / (np.max(column_data) - np.min(column_data))

    return normalized_data


# -----------2.加权标准化-----------
def weighted_normalization(normalized_data, weights):
    """
    对标准化后的数据进行加权处理。
    aram normalized_data:标准化后的数据矩阵。
    :param weights:各指标的权重向量。
    :return:加权标准化后的数据矩阵。
    """
    return normalized_data * weights


# -----------3.计算理想解和负理想解-----------
def calculate_ideal_solutions(weighted_data):
    """
    计算理想解和负理想解。
    :param weighted_data:加权标准化后的数据矩阵。
    :return:理想解和负理想解。
    """
    ideal_solution = np.max(weighted_data, axis=0)  # 理想解
    negative_ideal_solution = np.min(weighted_data, axis=0)  # 负理想解
    return ideal_solution, negative_ideal_solution


# -----------4.计算距离-----------
def calculate_distances(weighted_data, ideal_solution, negative_ideal_solution):
    """
    计算各方案与理想解和负理想解的距离。
    :param weighted_data:加权标准化后的数据矩阵。
    :param ideal_solution:理想解。
    :param negative_ideal_solution:负理想解。
    :return:各方案与理想解和负理想解的距离。
    """
    n = weighted_data.shape[0]  # 方案数量
    d_plus = np.zeros(n)  # 与理想解的距离
    d_minus = np.zeros(n)  # 与负理想解的距离

    for i in range(n):
        d_plus[i] = np.sqrt(np.sum((weighted_data[i] - ideal_solution) ** 2))
        d_minus[i] = np.sqrt(np.sum((weighted_data[i] - negative_ideal_solution) ** 2))

    return d_plus, d_minus


# -----------5.计算贴近度-----------
def calculate_closeness(d_plus, d_minus):
    """
    计算各方案的贴近度。
    :param d_plus:各方案与理想解的距离。
    :param d_minus:各方案与负理想解的距离。
    :return:各方案的贴近度。
    """
    return d_minus / (d_plus + d_minus)

# -----------6.数据输入部分-----------
# 原始数据矩阵
data = np.array([
    [100, 8, 8],  # 供应商A
    [150, 7, 12],  # 供应商B
    [120, 9, 10]  # 供应商C
])

# 指标类型：1表示正向指标，-1表示逆向指标
criteria_types = [-1, 1, -1]
# 指标权重
weights = np.array([0.4, 0.3, 0.3])

# -----------7.计算标准化数据-----------
normalized_data = normalize_data(data, criteria_types)
print("标准化后的数据：")
print(normalized_data)

# -----------8.计算加权标准化数据-----------
weighted_data = weighted_normalization(normalized_data, weights)
print("\n加权标准化后的数据：")
print(weighted_data)

# -----------9.计算理想解和负理想解-----------
ideal_solution, negative_ideal_solution = calculate_ideal_solutions(weighted_data)
print("\n理想解：", ideal_solution)
print("负理想解：", negative_ideal_solution)

# -----------10.计算距离-----------
d_plus, d_minus = calculate_distances(weighted_data, ideal_solution, negative_ideal_solution)
print("\n与理想解的距离：", d_plus)
print("与负理想解的距离：", d_minus)

# -----------11.计算贴近度-----------
closeness = calculate_closeness(d_plus, d_minus)
print("\n各方案的贴近度：", closeness)

# -----------12.排序与输出结果-----------
suppliers = ["供应商A", "供应商B", "供应商C"]
ranking = sorted(zip(suppliers, closeness), key=lambda x: x[1], reverse=True)
print("\n推荐排序：")
for i, (supplier, score) in enumerate(ranking, 1):
    print(f"{i}.{supplier}:{score:.4f}")
