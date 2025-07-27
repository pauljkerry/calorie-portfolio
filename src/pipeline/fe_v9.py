import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


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
    - MLP用
    """
    # 全データを結合（train + original + test）
    all_data = pd.concat(
        [train_data, test_data], ignore_index=True
    )

    # === 1) カテゴリー変数をone hot encoding ===
    cat_cols = ["Sex"]
    encoder = OneHotEncoder(drop='first', sparse_output=False)

    ohe_array = encoder.fit_transform(all_data[cat_cols])
    cat_ohe_df = pd.DataFrame(
        ohe_array,
        columns=encoder.get_feature_names_out(cat_cols),
        index=all_data.index
    )

    # === 2) 数値変数を標準化
    num_df = all_data.select_dtypes(
        include=np.number
    ).drop("target", axis=1)

    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(num_df)
    num_scaled_df = pd.DataFrame(
        scaled_array,
        columns=num_df.columns,
        index=all_data.index
    )

    # === dfを結合 ===
    df_feat = pd.concat([num_scaled_df, cat_ohe_df], axis=1)

    # === データを分割 ===
    tr_df = df_feat.iloc[:len(train_data)].copy()
    test_df = df_feat.iloc[len(train_data):]

    # === targetを追加 ===
    tr_df["target"] = train_data["target"]

    return tr_df, test_df