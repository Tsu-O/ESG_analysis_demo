# -*- coding: utf-8 -*-
# ダミーデータ生成スクリプト

import pandas as pd
import numpy as np

# 設定ファイルの読み込み
df_config = pd.read_csv('data_generate_config.csv', encoding='utf-8')

# 年度リスト（2001～2020年度の20年分）
years = list(range(2001, 2021))

# PBRのみ抽出
pbr_row = df_config[df_config['KPI'] == 'PBR'].iloc[0]
# ROEのみ抽出
roe_row = df_config[df_config['KPI'] == 'ROE'].iloc[0]

# ダミーデータ格納用リスト
data = []

# PBRのダミーデータ生成
min_val = float(pbr_row['最小'])
max_val = float(pbr_row['最大'])
log_min = np.log(min_val)
log_max = np.log(max_val)
log_mean = (log_min + log_max) / 2
log_std = (log_max - log_min) / 6  # 99.7%が範囲内になるように
for year in years:
    log_value = np.random.normal(loc=log_mean, scale=log_std)
    log_value = np.clip(log_value, log_min, log_max)
    value = np.exp(log_value)
    data.append({
        'KPI': pbr_row['KPI'],
        '年度': year,
        '値': value,
        '単位': pbr_row['単位']
    })

# ROEのダミーデータ生成
min_val = float(roe_row['最小'])
max_val = float(roe_row['最大'])
log_min = np.log(min_val)
log_max = np.log(max_val)
log_mean = (log_min + log_max) / 2
log_std = (log_max - log_min) / 6  # 99.7%が範囲内になるように
for year in years:
    log_value = np.random.normal(loc=log_mean, scale=log_std)
    log_value = np.clip(log_value, log_min, log_max)
    value = np.exp(log_value)
    data.append({
        'KPI': roe_row['KPI'],
        '年度': year,
        '値': value,
        '単位': roe_row['単位']
    })

# ある1つのESG関連のKPIであるESGKPIに対して、
# log(PBR_i) = alpha + beta_1 * log(ROE_i) + beta_2 * log(ESGKPI_{i-t}) + gamma_{i-t}
# が成立するように、ROE_iやESGKPI_{i-t}を生成
# ただし、iは年度、tは過去の年度を表す

# alpha: 定数項。ROEでもESGでも説明できないPBR上昇の影響要素
alpha = 0.1
# beta_1: ROEとPBRの関係性の強さを表す係数
beta_1 = 0.5

# beta_2 * log(ESGKPI_{i-t}) + gamma_{i-t} = log(PBR_i) - alpha - beta_1 * log(ROE_i) := sub_iを生成
sub = []
for year in years:
    # 該当年度のPBRを取得
    pbr_value = next(item for item in data if item['KPI'] == 'PBR' and item['年度'] == year)['値']
    # 該当年度のROEを取得
    roe_value = next(item for item in data if item['KPI'] == 'ROE' and item['年度'] == year)['値']
    # subを計算
    sub_value = np.log(pbr_value) - alpha - beta_1 * np.log(roe_value)
    sub.append({
        '年度': year,
        '値': sub_value
    })

# ESG KPIを抽出
esg_kpi = df_config[(df_config['分類'] == 'E') | (df_config['分類'] == 'S') | (df_config['分類'] == 'G')]
# ESG KPI別にダミーデータ生成
# まず、
# sub = beta_2 * log(x) + gamma, gamma ~ N( mean(sub) - beta_2 * mean(log(x)), sigma^2) ...(1)
# mean(log(x)) and mean(sub) are given
# としたときのlog(x)を生成した後、遅延浸透効果tを考慮してx_{i} -> ESGKPI_{i-t}としてESG KPIのダミーデータとする

for idx, esg_kpi_row in esg_kpi.iterrows():
    OFFSET = 1
    # log(x)の平均は、data_generate_configで設定した最小値と最大値の中間値にする
    # ただし、99.7%が範囲内になるようにする
    min_val = float(esg_kpi_row['最小'])
    max_val = float(esg_kpi_row['最大'])
    log_min = np.log(min_val) if min_val > 0 else np.log(min_val + OFFSET)
    log_max = np.log(max_val) if min_val > 0 else np.log(max_val + OFFSET)
    log_mean = (log_min + log_max) / 2
    log_std = (log_max - log_min) / 6  # 99.7%が範囲内になるように

    # 遅延浸透効果tを設定
    t = int(esg_kpi_row['遅延浸透効果'])
    # この時、回帰分析で使用されるesg_kpiは2000年度~2020-t年度
    # 合わせて使用されるsubは2000+t年度から2020年度
    sub_mean = np.mean([item['値'] for item in sub if item['年度'] >= 2000 + t])

    # beta_2(回帰係数の推定値)を設定
    beta_2 = float(esg_kpi_row['回帰係数'])
    # t値を設定
    t_value = float(esg_kpi_row['t値'])
    # t値からSE(beta_2): 回帰係数の推定値の標準偏差を設定
    se_beta_2 = beta_2 / t_value
    
    # SE(beta_2) = \sqrt( sigma^2 / (n * var(x)) ) = \sqrt( sigma^2 / (n * std(x)^2))
    # -> sigma^2 = SE(beta_2)^2 * n * var(x) = SE(beta_2)^2 * n * std(x)^2
    # -> sigma = SE(beta_2) * sqrt(n * var(x)) = SE(beta_2) * sqrt(n) * std(x) ...(2)
    # ただし、nは観測数、var(x)はxの分散、std(x)はxの標準偏差
    n = len(years) - t_value
    sigma = se_beta_2 * np.sqrt(n) * log_std #(2)式より

    for year in years:
        # (1)式のgammmaを生成。ただし、log(x)の平均がlog_meanになるようにする
        gamma = np.random.normal(loc= sub_mean - beta_2 * log_mean, scale=sigma)
        # subを取得
        sub_value = next(item for item in sub if item['年度'] == year)['値']
        # log(x)を計算
        log_x = (sub_value - gamma) / beta_2
        # xを計算
        value = np.exp(log_x)
        # tだけ年度を遅らせる
        data.append({
            'KPI': esg_kpi_row['KPI'],
            '年度': year - t,
            '値': value,
            '単位': esg_kpi_row['単位']
        })

# DataFrame化してCSV出力
result = pd.DataFrame(data)
# 1列目を年度、2列目以降をKPIの値にする
result = result.pivot(index='年度', columns='KPI', values='値')
result.to_csv('dummy_data.csv', index=True, encoding='utf-8-sig')
print('dummy_data.csvを出力しました。')
