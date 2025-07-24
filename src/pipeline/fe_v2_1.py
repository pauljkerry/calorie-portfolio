import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from itertools import combinations


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
    - logreg用
    - 交互作用2ペア
    - 残差が外れ値 or not の3値分類
    """
    # 全データを結合（train + original + test）
    all_data = pd.concat(
        [train_data, test_data], ignore_index=True
    )

    # === 1) カテゴリー変数をOne hot encoding ===
    cat_cols = ["Sex"]

    encoder = OneHotEncoder(
        sparse_output=False, dtype=int, handle_unknown='ignore',
        drop="first"
    )
    ohe_array = encoder.fit_transform(all_data[cat_cols])
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=encoder.get_feature_names_out(cat_cols),
        index=all_data.index
    )

    # === 2) 交互作用を追加
    inter_df = pd.DataFrame(index=all_data.index)
    num_df1 = all_data.select_dtypes(
        include=np.number
    ).drop("target", axis=1)
    num_df2 = pd.concat([ohe_df, num_df1], axis=1)

    for col1, col2 in combinations(num_df2.columns, 2):
        col_name = f"{col1}_{col2}"
        inter_df[col_name] = num_df2[col1] * num_df2[col2]

    # === 2) targetをoofとの残差をbin分割したものにする ===
    oof = np.load("../artifacts/oof/single/oof_single_3.npy")
    residual = train_data["target"].values - oof

    bins = [-np.inf, -0.1, 0.1, np.inf]
    residual_bins = np.digitize(residual, bins[1:])

    # === 3) 数値変数を標準化
    num_df3 = pd.concat([inter_df, num_df1], axis=1)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(num_df3)

    scaled_df = pd.DataFrame(
        scaled_array, columns=num_df3.columns, index=all_data.index
    )

    # === dfを結合 ===
    df_feat = pd.concat([
        scaled_df,
        ohe_df
    ], axis=1)

    # === データを分割 ===
    tr_df = df_feat.iloc[:len(train_data)].copy()
    test_df = df_feat.iloc[len(train_data):]

    # === targetを追加 ===
    tr_df["target"] = residual_bins

    return tr_df, test_df