import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
import time
from src.utils.print_duration import print_duration


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
    - 数値変数のカテゴリ化
    - embedding処理
    - mlp用
    """
    # 全データを結合（train + original + test）
    all_data = pd.concat(
        [train_data, test_data], ignore_index=True
    )

    # === 1) 数値変数のカテゴリ化
    num_df = all_data.select_dtypes(
        include=np.number
    ).drop("target", axis=1)
    bin_df = num_df.astype("str")
    cat_df = all_data.select_dtypes(include=["category", "object"])
    merged_df = pd.concat([bin_df, cat_df.astype("str")], axis=1)

    # === 2) 3変数交互作用を追加 ===
    inter_df2 = pd.DataFrame(index=all_data.index)
    inter_df3 = pd.DataFrame(index=all_data.index)

    for c1, c2 in combinations(merged_df.columns, 2):
        new_col_name = f"{c1}_{c2}"
        inter_df2[new_col_name] = merged_df[c1] + "_" + merged_df[c2]

    for c1, c2, c3 in combinations(merged_df.columns, 3):
        new_col_name = f"{c1}_{c2}_{c3}"
        inter_df3[new_col_name] = (merged_df[c1] + "_" + merged_df[c2] +
                                   merged_df[c3])

    inter_df = pd.concat([inter_df2, inter_df3], axis=1)

    # === 3) one hot encoding
    start = time.time()
    print("OHE Starting...")

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_array = ohe.fit_transform(inter_df)
    ohe_cols = ohe.get_feature_names_out(inter_df.columns)
    inter_ohe_df = pd.DataFrame(
        ohe_array, columns=ohe_cols, index=inter_df.index
    )

    end = time.time()
    print_duration(start, end)

    # === dfを結合 ===
    df_feat = pd.concat([num_df, inter_ohe_df], axis=1)

    # === データを分割 ===
    tr_df = df_feat.iloc[:len(train_data)].copy()
    test_df = df_feat.iloc[len(train_data):]

    # === targetを追加 ===
    tr_df["target"] = train_data["target"]

    return tr_df, test_df