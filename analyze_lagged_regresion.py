# -*- coding: utf-8 -*-
# 回帰分析スクリプト

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import os
from tqdm import tqdm

# 日本語フォント設定（Windows向け）
matplotlib.rcParams['font.family'] = 'Meiryo'
matplotlib.rcParams['axes.unicode_minus'] = False

# 1. データ読み込み
df = pd.read_csv('dummy_data.csv', encoding='utf-8-sig')
df = df.set_index('年度')

# 2. PBR, ROE以外のKPI列名リスト
kpi_cols = [col for col in df.columns if col not in ['PBR', 'ROE']]

results = []

# 3. 各KPI・遅延浸透効果tごとに回帰分析
for kpi in kpi_cols:
    for t in range(0, 21):
        # KPIをt年だけシフト
        kpi_shifted = df[kpi].shift(t)
        # 欠損を除外
        reg_df = pd.DataFrame({
            'log_PBR': np.log(df['PBR']),
            'log_ROE': np.log(df['ROE']),
            'log_KPI': np.log(kpi_shifted)
        }).dropna()
        if len(reg_df) < 5:  # データ数が少なすぎる場合はスキップ
            continue
        X = reg_df[['log_ROE', 'log_KPI']]
        X = sm.add_constant(X)
        y = reg_df['log_PBR']
        model = sm.OLS(y, X).fit()
        # beta_2はlog_KPIの係数
        beta_2 = model.params['log_KPI']
        t_value = model.tvalues['log_KPI']
        p_value = model.pvalues['log_KPI']
        adj_r2 = model.rsquared_adj
        results.append({
            'KPI': kpi,
            'lag_t': t,
            'beta_2': beta_2,
            't_value': t_value,
            'p_value': p_value,
            'adj_r2': adj_r2
        })

# 4. 結果をDataFrame化
results_df = pd.DataFrame(results)

# 5. KPIごとに最もp値が低いtのみ抽出し、p値昇順で並べる
best_df = results_df.loc[results_df.groupby('KPI')['p_value'].idxmin()].sort_values('p_value')

# 保存
results_df.to_csv('all_lagged_regression_results.csv', index=False, encoding='utf-8-sig')
best_df.to_csv('best_lagged_regression_results.csv', index=False, encoding='utf-8-sig')

print('全組み合わせ結果: all_lagged_regression_results.csv')
print('KPIごと最良遅延効果: best_lagged_regression_results.csv')

# 6. KPI・lag_tごとに散布図を画像ファイルで出力
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

for kpi in tqdm(kpi_cols, desc='KPI'):
    # 最良lag_tのp値を取得
    best_row = best_df[best_df['KPI'] == kpi]
    if best_row.empty or best_row['p_value'].values[0] >= 0.05:
        continue  # p値が0.05以上なら描画しない

    # lag_t=0と最良lag_t（p値最小）だけ描画
    lag_ts = [0]
    best_lag_t = int(best_row['lag_t'].values[0])
    if best_lag_t != 0:
        lag_ts.append(best_lag_t)

    for t in lag_ts:
        kpi_shifted = df[kpi].shift(t)
        reg_df = pd.DataFrame({
            'log_PBR': np.log(df['PBR']),
            'log_ROE': np.log(df['ROE']),
            'log_KPI': np.log(kpi_shifted)
        }).dropna()
        if len(reg_df) < 5:
            continue
        # 回帰結果を取得
        row = results_df[(results_df['KPI'] == kpi) & (results_df['lag_t'] == t)]
        if row.empty:
            continue
        beta_2 = row['beta_2'].values[0]
        p_value = row['p_value'].values[0]
        # 散布図作成
        plt.figure(figsize=(6, 5))
        plt.scatter(reg_df['log_KPI'], reg_df['log_PBR'], alpha=0.7)
        plt.xlabel(f'log({kpi})')
        plt.ylabel('log(PBR)')
        plt.title(f'{kpi} (lag_t={t})')
        plt.figtext(0.15, 0.85, f'lag_t={t}\nbeta_2={beta_2:.3g}\np_value={p_value:.3g}', fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        plt.tight_layout()
        # ファイル名は日本語対応のためにエンコード
        safe_kpi = kpi.replace('/', '_').replace('（', '_').replace('）', '_').replace('%', 'pct').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'scatter_{safe_kpi}_lag{t}.png'), dpi=150)
        plt.close()

# 7. 最良lag_tのp値が0.05以下のKPIについて、PBR・KPI（全ての遅延t年分）を折れ線グラフで出力
for _, row in tqdm(best_df[best_df['p_value'] < 0.05].iterrows(), total=best_df[best_df['p_value'] < 0.05].shape[0], desc='Line plots'):
    kpi = row['KPI']
    lag_t = int(row['lag_t'])
    years = df.index.values
    pbr = df['PBR']

    # 各lagごとにPBRとKPI(lag_t)を1枚の画像にして保存
    colors = plt.cm.viridis(np.linspace(0, 1, 21))
    color_pbr = 'tab:blue'
    safe_kpi = kpi.replace('/', '_').replace('（', '_').replace('）', '_').replace('%', 'pct').replace(' ', '_')
    for t in range(0, 21):
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.set_xlabel('年度')
        ax1.set_ylabel('PBR', color=color_pbr)
        l1, = ax1.plot(years, pbr, label='PBR', marker='^', color=color_pbr)
        ax1.tick_params(axis='y', labelcolor=color_pbr)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax2 = ax1.twinx()
        kpi_shifted = df[kpi].shift(t)
        ax2.set_ylabel(f'{kpi}(lag_t={t})', color=colors[t])
        l2, = ax2.plot(years, kpi_shifted, label=f'{kpi}(lag_t={t})', marker='o', color=colors[t], alpha=0.8, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=colors[t])

        plt.title(f'{kpi} (lag_t={t}) とPBRの推移')
        fig.tight_layout()
        ax1.legend([l1, l2], ['PBR', f'{kpi}(lag_t={t})'], loc='upper left', bbox_to_anchor=(0, 1))
        plt.savefig(os.path.join(output_dir, f'line_{safe_kpi}_lag{t}.png'), dpi=150)
        plt.close()
