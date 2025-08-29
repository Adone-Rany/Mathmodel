import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# === 读取数据 ===
file1 = "表1-患者列表及临床信息.xlsx"
file2 = "表2-患者影像信息血肿及水肿的体积及位置.xlsx"
file3 = "表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx"
file_q1a = "q1_a2.xlsx"

df1 = pd.read_excel(file1, sheet_name="患者信息")
df2 = pd.read_excel(file2, sheet_name="Data")
df3 = pd.read_excel(file3, sheet_name="Hemo")
labels = pd.read_excel(file_q1a)

# === 数据合并 ===
data = df1.merge(df2, left_on="入院首次影像检查流水号", right_on="首次检查流水号", how="left")
data = data.merge(df3, left_on="入院首次影像检查流水号", right_on="流水号", how="left")
data = data[data["患者id"].str.startswith("sub")]
data = data.merge(labels[["患者id", "是否扩张"]], left_on="患者id", right_on="患者id", how="left")


# === 特征预处理：性别和血压 ===
def preprocess_features(dataframe):
    """处理性别和血压特征"""
    df = dataframe.copy()

    if '性别' in df.columns:
        gender_map = {'男': 1, '女': 0}
        df['性别'] = df['性别'].map(gender_map)

    if '血压' in df.columns:
        df[['收缩压', '舒张压']] = df['血压'].astype(str).str.split('/', expand=True).apply(pd.to_numeric,
                                                                                            errors='coerce')
        df = df.drop(columns=['血压'])

    return df


data = preprocess_features(data)

# === 构造特征矩阵 ===
feature_cols = list(df1.columns[4:23]) + list(df2.columns[2:24]) + list(df3.columns[2:34])

if '血压' in feature_cols:
    feature_cols.remove('血压')
if '性别' in feature_cols:
    feature_cols.remove('性别')

feature_cols.extend(['收缩压', '舒张压', '性别'])
feature_cols = [col for col in feature_cols if col in data.columns]

X = data[feature_cols]
y = data["是否扩张"]


# === 改进的数据预处理 ===
def preprocess_categorical_features(X_data):
    """改进的数据预处理"""
    X_processed = X_data.copy()

    # 转换object类型为数值
    object_cols = X_processed.select_dtypes(include=['object']).columns
    for col in object_cols:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')

    # 使用中位数填充NaN值（对异常值更鲁棒）
    for col in X_processed.columns:
        if X_processed[col].isnull().sum() > 0:
            X_processed[col].fillna(X_processed[col].median(), inplace=True)

    return X_processed


X = preprocess_categorical_features(X)


# === 特征选择 ===
def select_features(X_data, y_data, k=20):
    """选择最重要的k个特征"""
    selector = SelectKBest(score_func=f_classif, k=min(k, X_data.shape[1]))
    X_selected = selector.fit_transform(X_data, y_data)
    selected_features = X_data.columns[selector.get_support()]
    return X_selected, selected_features, selector


# === 数据标准化 ===
scaler = StandardScaler()

# === 训练集和测试集划分 ===
train_mask = data["患者id"].str.startswith("sub") & (data["患者id"].str[3:].astype(int) <= 100)
test_mask = data["患者id"].str.startswith("sub") & (data["患者id"].str[3:].astype(int) > 100) & (
            data["患者id"].str[3:].astype(int) <= 160)

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]
X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

# 移除NaN值
train_mask_clean = ~(X_train.isnull().any(axis=1) | y_train.isnull())
X_train_clean = X_train[train_mask_clean]
y_train_clean = y_train[train_mask_clean]

test_mask_clean = ~X_test.isnull().any(axis=1)
X_test_clean = X_test[test_mask_clean]
y_test_clean = y_test[test_mask_clean]

# === 处理类别不平衡 ===
print("处理类别不平衡...")
smote = SMOTE(random_state=42, sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_clean, y_train_clean)

# === 特征选择和标准化 ===
print("进行特征选择...")
X_train_selected, selected_features, selector = select_features(X_train_resampled, y_train_resampled, k=15)
X_test_selected = selector.transform(X_test_clean)

print(f"选择的特征: {list(selected_features)}")

# 标准化数据
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# === 超参数调优 ===
print("进行超参数调优...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train_resampled)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# === 使用最佳参数训练模型 ===
best_rf = grid_search.best_estimator_

# === 交叉验证评估 ===
cv_scores = cross_val_score(best_rf, X_train_scaled, y_train_resampled, cv=5, scoring='roc_auc')
print(f"交叉验证AUC分数: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# === 训练最终模型 ===
best_rf.fit(X_train_scaled, y_train_resampled)

# === 评估模型性能 ===
# 训练集评估
train_probs = best_rf.predict_proba(X_train_scaled)[:, 1]
train_predictions = best_rf.predict(X_train_scaled)
train_auc = roc_auc_score(y_train_resampled, train_probs)
train_accuracy = accuracy_score(y_train_resampled, train_predictions)

print(f"\n=== 模型性能评估 ===")
print(f"训练集AUC: {train_auc:.4f}")
print(f"训练集准确度: {train_accuracy:.4f}")

# 测试集评估
test_probs = best_rf.predict_proba(X_test_scaled)[:, 1]
test_predictions = best_rf.predict(X_test_scaled)
test_auc = roc_auc_score(y_test_clean, test_probs)
test_accuracy = accuracy_score(y_test_clean, test_predictions)

print(f"测试集AUC: {test_auc:.4f}")
print(f"测试集准确度: {test_accuracy:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_test_clean, test_predictions)
print(f"混淆矩阵:\n{cm}")

# === 特征重要性分析 ===
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特征重要性排名:")
print(feature_importance.head(10))

# === 对测试集进行预测 ===
test_data_clean = data.loc[X_test_clean.index]
test_data_clean["预测值"] = test_predictions
test_data_clean["扩张预测概率"] = test_probs.round(4)

out = test_data_clean[["患者id", "是否扩张", "扩张预测概率", "预测值"]]
out.to_excel("q1_b_optimized.xlsx", sheet_name="q1_b", index=False)
print("\n优化后的预测结果已保存到 q1_b_optimized.xlsx")