import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimSun']
rcParams['font.family'] = 'sans-serif'


class GM11:
    def __init__(self, data):
        self.data = data
        self.n = len(data)

    def fit(self):
        # 1. 累加生成
        self.ago = np.cumsum(self.data)

        # 2. 构造矩阵B和Y
        B = np.zeros((self.n - 1, 2))
        for i in range(self.n - 1):
            B[i, 0] = -0.5 * (self.ago[i] + self.ago[i + 1])
            B[i, 1] = 1
        Y = self.data[1:].reshape(-1, 1)

        # 3. 计算参数a和b
        self.a, self.b = np.linalg.inv(B.T @ B) @ B.T @ Y
        self.a, self.b = self.a[0], self.b[0]

        # 4. 计算拟合值
        self.fit_values = np.zeros_like(self.data)
        self.fit_values[0] = self.data[0]
        for k in range(1, self.n):
            self.fit_values[k] = (1 - np.exp(self.a)) * (self.data[0] - self.b / self.a) * np.exp(-self.a * (k))

        # 5. 计算残差
        self.residuals = self.data - self.fit_values

        # 6. 计算后验差比C值
        self._calculate_c()

        return self

    def _calculate_c(self):
        """计算后验差比C值"""
        # 原始数据标准差
        S1 = np.std(self.data, ddof=1)

        # 残差标准差
        S2 = np.std(self.residuals, ddof=1)

        # 后验差比
        self.C = S2 / S1

        # 精度等级判定
        if self.C <= 0.35:
            self.grade = "优"
        elif self.C <= 0.5:
            self.grade = "合格"
        elif self.C <= 0.65:
            self.grade = "勉强合格"
        else:
            self.grade = "不合格"

    def predict(self, steps):
        """预测未来值"""
        pred = []
        for k in range(self.n, self.n + steps):
            pred.append((1 - np.exp(self.a)) * (self.data[0] - self.b / self.a) * np.exp(-self.a * (k)))
        return np.array(pred)


# 建模与预测
def gm11_energy_forecast():
    # 历史数据（2010-2020年能源消费量，万吨标准煤）
    energy_hist = np.array([23539.31, 26860.03, 27999.22, 28203.10, 28170.51,
                            29033.61, 29947.98, 30669.89, 31373.13, 32227.51, 31438.00])

    # 1. 建立GM(1,1)模型
    model = GM11(energy_hist).fit()

    # 2. 预测2021-2060年（40年）
    future_years = np.arange(2021, 2061)
    pred_energy = model.predict(40)

    # 3. 计算误差指标
    mae = np.mean(np.abs(model.residuals))
    rmse = np.sqrt(np.mean(model.residuals ** 2))

    # 4. 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(2010, 2021), energy_hist, 'bo-', label='历史数据')
    plt.plot(np.arange(2010, 2021), model.fit_values, 'rs--', label='拟合值')
    plt.plot(future_years, pred_energy, 'g--', label='预测值')

    plt.title('基于GM(1,1)模型的能源消费量预测（2021-2060）', fontsize=14)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('能源消费量（万吨标准煤）', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(np.arange(2010, 2061, 5))

    # 5. 输出预测表格和模型参数
    results = pd.DataFrame({
        '年份': future_years,
        '预测能源消费量': np.round(pred_energy, 2)
    })

    print("\n=== 模型参数 ===")
    print(f"发展系数 a = {model.a:.6f}")
    print(f"灰色作用量 b = {model.b:.2f}")

    print("\n=== 精度检验 ===")
    print(f"后验差比 C = {model.C:.4f} （精度等级：{model.grade}）")
    print(f"平均绝对误差(MAE) = {mae:.2f}")
    print(f"均方根误差(RMSE) = {rmse:.2f}")

    print("\n=== 2021-2060年预测结果（每10年）===")
    print(results.iloc[::])

    plt.tight_layout()
    plt.show()


# 执行预测
gm11_energy_forecast()