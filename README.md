## 目次

1. [コンペ概要](#1-コンペ概要)
2. [環境](#環境)
3. [ディレクトリ構成](#ディレクトリ構成)
4. [学びと課題](#学びと課題)
5. [実験結果](#実験結果)

## 1. コンペ概要
- **コンペ名**: [Kaggle Playground Series - S5E6](https://www.kaggle.com/competitions/playground-series-s5e6)
- **タスク**: 多クラス分類
- **評価指標**: Map@3
- **最終スコア**: 0.35138
- **最終順位**: 707位 / 2648チーム中
- **コンペ終了後の改善結果**: 0.36485(510位相当)

## 2. 環境
### 使用言語・主要ライブラリ

| パッケージ名          | バージョン  |
|----------------------|------------|
| Python               | 3.10       |
| pandas               | 2.2.3      |
| numpy                | 2.1.3      |
| scikit-learn         | 1.6.1      |
| XGBoost              | 3.0.2      |
| LightGBM             | 4.6.0      |
| optuna               | 4.4.0      |
| matplotlib           | 3.10.0     |
| seaborn              | 0.13.2     |


### 環境変数
| 変数名  | 説明       | 例            |
|--------|-------------------|---------------|
| `OPTUNA_STORAGE_URL` | Optuna が使用する PostgreSQL の接続 URL  | `postgresql://user:pass@localhost:5432/db`  |

## 3. ディレクトリ構成

```text
.
├── README.md
├── summary.md
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_pre.ipynb
│   ├── 03_xgb_analysis.ipynb
│   ├── 04_xgb_tuning.ipynb
│   ├── 05_lgb_analysis.ipynb
│   ├── 06_lgb_tuning.ipynb
│   └── 07_ensemble.ipynb
└── src
    ├── __init__.py
    ├── models
    │   ├── __init__.py
    │   ├── lgbm
    │   │   ├── __init__.py
    │   │   ├── lgbm_cv_trainer.py        ← LGBMのCV学習用
    │   │   ├── lgbm_ensemble.py          ← LGBMのアンサンブル処理
    │   │   └── lgbm_optuna_optimizer.py  ← Optunaでのパラメータ探索
    │   └── xgb
    │       ├── __init__.py               
    │       ├── xgb_cv_trainer.py         ← XGBのCV学習用
    │       ├── xgb_ensemble.py           ← XGBのアンサンブル処理
    │       └── xgb_optuna_optimizer.py   ← Optunaでのパラメータ探索
    ├── pipeline
    │   ├── __init__.py
    │   ├── fe_gbdt_v4.py  ← コンペ終了後に改善したFEファイル
    │   ├── fe_gbdt_v5.py  ← 提出に使ったFEファイル
    │   └── prepro.py
    └── utils
        ├── __init__.py
        ├── create_sub.py
        ├── eda.py         ← EDAの可視化関数をまとめたファイル
        ├── map_k.py
        ├── optuna_visualizer.py
        └── weight_optimizer.py
```

## 4. 学びと課題

コンペを通して学んだ点や今後の改善点などを以下のMarkdownにまとめました。

 [学びと課題](./summary.md)

## 5. 実験結果
Optunaによるハイパーパラメータ探索の詳細結果は、以下のNotionページでご覧いただけます。

 [Notionリンク](https://www.notion.so/Fertilizer-port-folio-22efeb435b0180c18267fb6f5f373307?source=copy_link)