import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 日本語フォント設定（Windows向け）
matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False

# ダミーデータの読み込み
df = pd.read_csv('dummy_esg_data.csv', encoding='utf-8')

# KPIごとに分布を描画
kpis = df['KPI名'].unique()
n_kpi = len(kpis)
cols = 3
rows = (n_kpi + cols - 1) // cols

plt.figure(figsize=(cols*6, rows*4))
for i, kpi in enumerate(kpis, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(df[df['KPI名'] == kpi]['値'], bins=15, kde=True)
    plt.title(kpi)
    plt.xlabel('値')
    plt.ylabel('件数')
plt.tight_layout()
plt.show()
