# ESG分析デモ（柳モデル）

## 概要
本リポジトリは、ESG（環境・社会・ガバナンス）KPIと財務指標（PBR, ROE）との関係を多重回帰分析で検証する「柳モデル」分析デモです。ダミーデータ生成から回帰分析・可視化まで一連の流れを自動化しています。

---

## ディレクトリ構成

- `generate_dummy_data.py` … ダミーデータ生成スクリプト
- `data_generate_config.csv` … 各KPIの設定ファイル
- `dummy_data.csv` … 生成されたダミーデータ
- `analyze_lagged_regresion.py` … 遅延効果付き回帰分析・可視化スクリプト
- `all_lagged_regression_results.csv` … 全KPI・遅延tの回帰結果
- `best_lagged_regression_results.csv` … KPIごと最良遅延tの回帰結果
- `plots/` … 散布図・折れ線グラフ画像出力先
- `visualize_dummy_data.py` … ダミーデータ分布の可視化
- `requirements.txt` … 必要なPythonパッケージ

---

## セットアップ手順

### 1. Python仮想環境の作成・有効化

#### PowerShellの場合
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### コマンドプロンプト（cmd.exe）の場合
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### 2. 必要パッケージのインストール

#### PowerShell・cmd共通
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## ダミーデータ生成

- `generate_dummy_data.py` を実行すると、`data_generate_config.csv`の設定に基づき、
  - PBR, ROE, ESG各KPIの20年分ダミーデータ（ln正規分布）
  - ESG KPIは遅延浸透効果・回帰係数等を考慮
- 出力: `dummy_data.csv`（年度×KPIの値）

#### PowerShell
```powershell
.venv\Scripts\python.exe generate_dummy_data.py
```
#### cmd
```cmd
.venv\Scripts\python.exe generate_dummy_data.py
```

---

## ダミーデータの分布可視化

- `visualize_dummy_data.py` を実行すると、KPIごとの値の分布ヒストグラムが表示されます。

#### PowerShell
```powershell
.venv\Scripts\python.exe visualize_dummy_data.py
```
#### cmd
```cmd
.venv\Scripts\python.exe visualize_dummy_data.py
```

---

## 回帰分析・遅延効果検証

- `analyze_lagged_regresion.py` を実行すると、
  - 各KPI・遅延t（0～20年）ごとに
    - log(PBR) = α + β₁ log(ROE) + β₂ log(KPI_{i-t}) + γ_{i-t}
    - 回帰分析（β₂のt値・p値・自由度修正済み決定係数）
  - 全組み合わせ結果: `all_lagged_regression_results.csv`
  - KPIごと最良遅延tのみ: `best_lagged_regression_results.csv`
  - p値が0.05未満のKPIについて、
    - 散布図（log(KPI) vs log(PBR)）
    - PBR・KPI（遅延前/後）の折れ線グラフ
    - 画像は `plots/` フォルダに出力

#### PowerShell
```powershell
.venv\Scripts\python.exe analyze_lagged_regresion.py
```
#### cmd
```cmd
.venv\Scripts\python.exe analyze_lagged_regresion.py
```

---

## 主要ファイルの説明

- `data_generate_config.csv` … KPI名、分類、単位、最小・最大値、遅延浸透効果、回帰係数などの設定
- `generate_dummy_data.py` … 上記設定に基づきダミーデータを生成
- `analyze_lagged_regresion.py` … 遅延効果を考慮した多重回帰分析・可視化
- `visualize_dummy_data.py` … ダミーデータの分布をKPIごとに可視化
- `requirements.txt` … 必要なPythonパッケージ一覧

---

## 注意事項
- Windows環境・Meiryoフォント前提（matplotlibの日本語対応）
- データやパラメータはダミーです。実データでの運用時は適宜修正してください。
- GitHub等でSSH接続する場合は公開鍵の登録が必要です。

---

## 参考
- 柳モデル: ESG指標と財務指標の関係を時系列・遅延効果も含めて多重回帰で検証する分析手法

---

## 実行例

#### PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe generate_dummy_data.py
.venv\Scripts\python.exe visualize_dummy_data.py
.venv\Scripts\python.exe analyze_lagged_regresion.py
```

#### cmd
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe generate_dummy_data.py
.venv\Scripts\python.exe visualize_dummy_data.py
.venv\Scripts\python.exe analyze_lagged_regresion.py
```
