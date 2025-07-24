import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def feature_engineering(train_data, test_data):
    """
    特徴量エンジニアリングを行う関数

    Parameters
    ----------
    train_data : pd.DataFrame
        前処理済みの学習用データ
    test_data : pd.DataFrame
        前処理済みのテスト用データ

    Returns
    -------
    tr_df : pd.DataFrame
        特徴量エンジニアリング済みの学習用データ
    test_df : pd.DataFrame
        特徴エンジニアリング済みのテスト用データ

    Notes
    -----
    - 特徴量エンジニアリングはせず
    """
    # 全データを結合（train + original + test）
    all_data = pd.concat(
        [train_data, test_data], ignore_index=True
    )

    # === 1) カテゴリー変数をlabel encoding ===
    cat_cols = ["Sex"]
    cat_le_df = pd.DataFrame(index=all_data.index)

    for c in cat_cols:
        le = LabelEncoder()
        cat_le_df[c] = le.fit_transform(all_data[c])
    cat_le_df = cat_le_df.astype("str")

    # === dfを結合 ===
    num_df = all_data.select_dtypes(
        include=np.number
    ).drop("target", axis=1)
    df_feat = pd.concat([num_df, cat_le_df], axis=1)

    # === データを分割 ===
    tr_df = df_feat.iloc[:len(train_data)].copy()
    test_df = df_feat.iloc[len(train_data):]

    # === targetを追加 ===
    tr_df["target"] = train_data["target"]

    return tr_df, test_df