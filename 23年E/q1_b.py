# LightGBM 模型训练，效果一般
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


# === 读取数据 ===
file1 = "表1-患者列表及临床信息.xlsx"
file2 = "表2-患者影像信息血肿及水肿的体积及位置.xlsx"
file3 = "表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx"
file_q1a = "q1_a2.xlsx"  # 来自问题a的结果

df1 = pd.read_excel(file1, sheet_name="患者信息")  # 包含ID, 流水号, E-W字段
df2 = pd.read_excel(file2, sheet_name="Data")  # 包含流水号, C-X字段
df3 = pd.read_excel(file3, sheet_name="Hemo")  # 包含流水号, C-AG字段
labels = pd.read_excel(file_q1a)  # 包含 患者id, 是否扩张

# === 数据合并 ===
# 先用流水号把表1与表2、表3连起来
data = df1.merge(df2, left_on="入院首次影像检查流水号", right_on="首次检查流水号", how="left")
data = data.merge(df3, left_on="入院首次影像检查流水号", right_on="流水号", how="left")

# 保留 sub001–sub160
data = data[data["患者id"].str.startswith("sub")]

# 合并标签
data = data.merge(labels[["患者id", "是否扩张"]], left_on="患者id", right_on="患者id", how="left")


# === 特征预处理：性别和血压 ===
def preprocess_features(dataframe):
    """处理性别和血压特征"""
    df = dataframe.copy()

    # 处理性别特征
    if '性别' in df.columns:
        # 将'男'映射为1，'女'映射为0
        gender_map = {'男': 1, '女': 0}
        df['性别'] = df['性别'].map(gender_map)
        print("性别数据处理完成：'男' -> 1, '女' -> 0")

    # 处理血压特征
    if '血压' in df.columns:
        # 将血压数据拆分为收缩压和舒张压
        df[['收缩压', '舒张压']] = df['血压'].astype(str).str.split('/', expand=True).apply(pd.to_numeric,
                                                                                            errors='coerce')

        # 可选：移除原始血压列
        df = df.drop(columns=['血压'])

        print("血压数据处理完成，已拆分为'收缩压'和'舒张压'两列。")

    return df


data = preprocess_features(data)

# === 构造特征矩阵 ===
# 取表1 E–W、表2 C–X、表3 C–AG
feature_cols = list(df1.columns[4:23]) + list(df2.columns[2:24]) + list(df3.columns[2:34])

# 添加处理后的血压特征
# 这里的'血压'列在预处理后已被移除，所以我们添加新生成的列
if '收缩压' in data.columns and '舒张压' in data.columns:
    # 移除原始'血压'列，并加入新的收缩压和舒张压
    if '血压' in feature_cols:
        feature_cols.remove('血压')
    feature_cols.append('收缩压')
    feature_cols.append('舒张压')

# 移除原始'性别'列，因为已经被替换成数字
if '性别' in feature_cols:
    feature_cols.remove('性别')
    feature_cols.append('性别')

X = data[feature_cols]
y = data["是否扩张"]


# === 数据类型处理 ===
# 转换所有object类型为数值类型
def preprocess_categorical_features(X_data):
    """处理剩余的分类特征"""
    X_processed = X_data.copy()
    object_cols = X_processed.select_dtypes(include=['object']).columns
    for col in object_cols:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')

    return X_processed


X = preprocess_categorical_features(X)

# === 训练集和预测集 ===
train_data = data[data["患者id"].isin([f"sub{i:03d}" for i in range(1, 101)])]
test_data = data[data["患者id"].isin([f"sub{i:03d}" for i in range(102, 161)])]

X_train = X.loc[train_data.index]
y_train = train_data["是否扩张"]
X_test = X.loc[test_data.index]

# === LightGBM 模型训练 ===
# 删除NaN值以避免训练错误
# 注意：这种简单删除NaN的方式可能导致信息丢失，实际比赛中可尝试均值填充等方法。
train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
X_train_clean = X_train[train_mask]
y_train_clean = y_train[train_mask]

train_dataset = lgb.Dataset(X_train_clean, label=y_train_clean)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}

model = lgb.train(params, train_dataset, num_boost_round=200)

# === 训练集评估 ===
train_probs = model.predict(X_train_clean)
train_predictions = (train_probs > 0.5).astype(int)  # 将概率转换为类别预测
train_auc = roc_auc_score(y_train_clean, train_probs)
train_accuracy = accuracy_score(y_train_clean, train_predictions)
print(f"训练集AUC: {train_auc:.4f}")
print(f"训练集准确度: {train_accuracy:.4f}")

# === 预测概率 ===
# 清理测试数据中的NaN值
test_mask = ~X_test.isnull().any(axis=1)
X_test_clean = X_test[test_mask]
test_data_clean = test_data.loc[X_test_clean.index]

probs = model.predict(X_test_clean)
predictions = (probs > 0.5).astype(int)  # 将概率转换为类别预测
test_data_clean["预测值"] = predictions
test_data_clean["扩张预测概率"] = probs.round(4)

# === 测试集评估 ===
if "是否扩张" in test_data_clean.columns:
    test_accuracy = accuracy_score(test_data_clean["是否扩张"], test_data_clean["预测值"])
    test_auc = roc_auc_score(test_data_clean["是否扩张"], test_data_clean["扩张预测概率"])
    print(f"测试集AUC: {test_auc:.4f}")
    print(f"测试集准确度: {test_accuracy:.4f}")

# === 保存结果 ===
# out = test_data_clean[["患者id", "扩张预测概率"]]
# out.to_excel("q1_b.xlsx", sheet_name="q1_b", index=False)
# print("预测结果已保存到 q1_b.xlsx")