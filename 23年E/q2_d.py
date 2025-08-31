import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# === 读取数据 ===
file = "数据处理q2_d.xlsx"
df = pd.read_excel(file)

# === 注意：体积单位是 10^-3 ml，需要换算成 ml ===
df["HM_volume_ml"] = df["HM_volume"] / 1000
df["ED_volume_ml"] = df["ED_volume"] / 1000

# === 划分区间 ===
def categorize_hm(x):
    if x < 30:
        return "小血肿(<30ml)"
    elif x <= 60:
        return "中血肿(30-60ml)"
    else:
        return "大血肿(>60ml)"

def categorize_ed(x):
    if x < 20:
        return "轻水肿(<20ml)"
    elif x <= 50:
        return "中水肿(20-50ml)"
    else:
        return "重水肿(>50ml)"

df["血肿分组"] = df["HM_volume_ml"].apply(categorize_hm)
df["水肿分组"] = df["ED_volume_ml"].apply(categorize_ed)
df["联合分组"] = df["血肿分组"] + "-" + df["水肿分组"]

# 只分析前100例患者 sub001–sub100
df["患者编号"] = df["患者id"].str.extract(r"sub(\d+)").astype(int)
df_100 = df[df["患者编号"] <= 100]

# === 治疗手段列 ===
treatments = ["脑室引流", "止血治疗", "降颅压治疗", "降压治疗",
              "镇静、镇痛治疗", "止吐护胃", "营养神经"]

# === 统计各分组治疗情况 ===
treatment_counts = df_100.groupby("联合分组")[treatments].sum().fillna(0)

# === 热力图（百分比形式） ===
# === 每组样本数 ===
group_counts = df_100.groupby("联合分组").size()

# === 热力图（百分比：该治疗手段在该区间样本中使用的比例） ===
treatment_counts_percent = treatment_counts.div(group_counts, axis=0) * 100

plt.figure(figsize=(10, 6))
sns.heatmap(treatment_counts_percent, annot=True, fmt=".1f", cmap="YlOrRd")
plt.title("前100例患者不同分组治疗手段使用比例（%）")
plt.ylabel("联合分组")
plt.xlabel("治疗手段")
plt.tight_layout()
plt.show()


# === 堆叠柱状图 ===
treatment_counts.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
plt.title("前100例患者不同分组的治疗手段堆叠柱状图")
plt.ylabel("人数")
plt.xlabel("联合分组")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# === 扇形图（每个区间一个扇形图） ===
n_groups = len(treatment_counts)
ncols = 3
nrows = (n_groups + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))
axes = axes.flatten()

for i, (group, row) in enumerate(treatment_counts.iterrows()):
    axes[i].pie(row.values, labels=row.index, autopct="%1.1f%%", startangle=140,
                colors=sns.color_palette("Set3"))
    axes[i].set_title(f"{group} 治疗手段分布")

# 去掉多余子图
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# === 九个区间样本数量柱状图 ===
group_counts = df_100["联合分组"].value_counts().reindex([
    "小血肿(<30ml)-轻水肿(<20ml)",
    "小血肿(<30ml)-中水肿(20-50ml)",
    "小血肿(<30ml)-重水肿(>50ml)",
    "中血肿(30-60ml)-轻水肿(<20ml)",
    "中血肿(30-60ml)-中水肿(20-50ml)",
    "中血肿(30-60ml)-重水肿(>50ml)",
    "大血肿(>60ml)-轻水肿(<20ml)",
    "大血肿(>60ml)-中水肿(20-50ml)",
    "大血肿(>60ml)-重水肿(>50ml)"
], fill_value=0)

plt.figure(figsize=(12, 6))
sns.barplot(x=group_counts.index, y=group_counts.values, palette="Set2")
plt.title("前100例患者九个区间样本数量分布")
plt.ylabel("样本数")
plt.xlabel("联合分组")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
