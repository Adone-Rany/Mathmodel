import pandas as pd
import numpy as np

# 读取整理好的数据
file = "数据处理q1_a.xlsx"
df = pd.read_excel(file)

results = []

for i in range(1, 161):  # sub001 ~ sub100
    pid = f"sub{i:03d}"
    row = df[df["患者id"] == pid]
    if row.empty:
        continue
    row = row.iloc[0]

    # 计算发病时间
    onset_to_first = row["发病到首次影像检查时间间隔"]  # 小时
    first_exam_time = pd.to_datetime(row["入院首次检查时间点"])
    onset_time = first_exam_time - pd.to_timedelta(onset_to_first, unit="h")

    # 首次血肿体积
    first_volume = row["HM_volume0"]

    expand_flag = 0
    expand_time = np.nan

    # 遍历随访检查
    for k in range(1, 9):  # 随访1~8
        time_col = f"随访{k}时间点"
        vol_col = f"HM_volume{k}"

        if pd.isna(row[time_col]) or pd.isna(row[vol_col]):
            continue

        follow_time = pd.to_datetime(row[time_col])
        follow_volume = row[vol_col]

        # 相对发病时间（小时）
        interval_h = (follow_time - onset_time).total_seconds() / 3600

        if interval_h <= 48:
            abs_inc = follow_volume - first_volume
            rel_inc = (follow_volume - first_volume) / first_volume if first_volume > 0 else 0
            if abs_inc >= 6000 or rel_inc >= 0.33:
                expand_flag = 1
                expand_time = interval_h
                break  # 记录首次发生扩张的时间

    results.append([pid, expand_flag, expand_time])

# 保存结果到 q1.xlsx
df_out = pd.DataFrame(results, columns=["患者id", "是否扩张", "扩张时间间隔(h)"])
df_out.to_excel("q1_a2.xlsx", sheet_name="q1", index=False)
print("结果已保存到 q1_a2.xlsx")
