import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# === 读取数据 ===
file1 = "表1-患者列表及临床信息.xlsx"
file2 = "表2-患者影像信息血肿及水肿的体积及位置.xlsx"
file3 = "表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx"
file_q1a = "q1_a2.xlsx"  # 来自问题a的结果

df1 = pd.read_excel(file1, sheet_name="患者信息")
df2 = pd.read_excel(file2, sheet_name="Data")
df3 = pd.read_excel(file3, sheet_name="Hemo")
labels = pd.read_excel(file_q1a)

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

    if '性别' in df.columns:
        gender_map = {'男': 1, '女': 0}
        df['性别'] = df['性别'].map(gender_map)
        print("性别数据处理完成：'男' -> 1, '女' -> 0")

    if '血压' in df.columns:
        df[['收缩压', '舒张压']] = df['血压'].astype(str).str.split('/', expand=True).apply(pd.to_numeric,
                                                                                            errors='coerce')
        df = df.drop(columns=['血压'])
        print("血压数据处理完成，已拆分为'收缩压'和'舒张压'两列。")
    return df


data = preprocess_features(data)

# === 构造特征矩阵 ===
feature_cols = list(df1.columns[4:23]) + list(df2.columns[2:24]) + list(df3.columns[2:34])

# 移除原始特征列并添加处理后的新列
if '血压' in feature_cols:
    feature_cols.remove('血压')
if '性别' in feature_cols:
    feature_cols.remove('性别')

feature_cols.append('收缩压')
feature_cols.append('舒张压')
feature_cols.append('性别')

# 确保特征列在数据中存在
feature_cols = [col for col in feature_cols if col in data.columns]

X = data[feature_cols]
y = data["是否扩张"]


# === 数据类型处理和填充 ===
def preprocess_categorical_features(X_data):
    """处理剩余的分类特征并填充NaN值"""
    X_processed = X_data.copy()

    # 转换所有object类型为数值类型
    object_cols = X_processed.select_dtypes(include=['object']).columns
    for col in object_cols:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')

    # 使用均值填充NaN值
    X_processed = X_processed.fillna(X_processed.mean())

    return X_processed


X = preprocess_categorical_features(X)

# === 训练集和预测集 ===
train_data_full = data[data["患者id"].str.startswith("sub") & (data["患者id"].str[3:].astype(int) <= 100)]
test_data_full = data[data["患者id"].str.startswith("sub") &
                     (data["患者id"].str[3:].astype(int) > 100) &
                     (data["患者id"].str[3:].astype(int) <= 160)]

X_train = X.loc[train_data_full.index]
y_train = train_data_full["是否扩张"]
X_test = X.loc[test_data_full.index]

# === 训练随机森林模型 ===
# 删除NaN值以避免训练错误
train_mask = ~(X_train.isnull().any(axis=1) | y_train.isnull())
X_train_clean = X_train[train_mask]
y_train_clean = y_train[train_mask]

# 初始化并训练随机森林模型
rf_model = RandomForestClassifier(
    n_estimators=250,  # 树的数量
    max_depth=6,  # 树的最大深度，控制过拟合
    min_samples_leaf=4,  # 叶子节点最少样本数，控制过拟合
    random_state=42,
    class_weight='balanced'  # 自动平衡类别权重
)

print("\n正在使用全部训练集训练随机森林模型...")
rf_model.fit(X_train_clean, y_train_clean)
print("模型训练完成。")

# === 评估模型性能 ===
# 使用训练好的模型在训练集上进行预测，评估AUC和准确度
train_probs = rf_model.predict_proba(X_train_clean)[:, 1]
train_predictions = rf_model.predict(X_train_clean)
train_auc = roc_auc_score(y_train_clean, train_probs)
train_accuracy = accuracy_score(y_train_clean, train_predictions)
print(f"训练集AUC: {train_auc:.4f}")
print(f"训练集准确度: {train_accuracy:.4f}")

# === 对测试集进行预测 ===
# 清理测试数据中的NaN值
test_mask = ~X_test.isnull().any(axis=1)
X_test_clean = X_test[test_mask]
test_data_clean = test_data_full.loc[X_test_clean.index]

# 进行预测
probs = rf_model.predict_proba(X_test_clean)[:, 1]
predictions = rf_model.predict(X_test_clean)  # 添加预测类别
test_data_clean["预测值"] = predictions
test_data_clean["扩张预测概率"] = probs.round(4)

# 计算测试集准确度（如果有真实标签）
if "是否扩张" in test_data_clean.columns:
    test_accuracy = accuracy_score(test_data_clean["是否扩张"], test_data_clean["预测值"])
    print(f"测试集准确度: {test_accuracy:.4f}")

out = test_data_clean[["患者id", "是否扩张", "扩张预测概率", "预测值"]]
out.to_excel("q1_b.xlsx", sheet_name="q1_b", index=False)
print("预测结果已保存到 q1_b.xlsx")
