from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
import pandas as pd
import numpy as np
from scipy import sparse
import time


def feature_engineering(train_data, test_data, weight=0.4):
    start = time.time()
    original_data = pd.read_csv("../artifacts/features/original_data.csv")

    # 結合
    all_data = pd.concat([train_data, original_data, test_data], ignore_index=True)

    # 数値→文字列化（カテゴリ化）
    num_cols = all_data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    all_data[num_cols] = all_data[num_cols].astype(str)

    cat_cols = ["Soil", "Crop"]
    interaction_cols = []

    # 交互作用列の生成（文字列結合で新しいカテゴリに）
    cols = list(num_cols) + cat_cols
    for c1, c2 in combinations(cols, 2):
        col_name = f"{c1}_{c2}"
        all_data[col_name] = all_data[c1] + "_" + all_data[c2]
        interaction_cols.append(col_name)

    # OneHotEncoder準備（すべての列をfit）
    enc = OneHotEncoder(sparse=True, handle_unknown='ignore')
    ohe_matrix = enc.fit_transform(all_data[cat_cols + num_cols + interaction_cols])

    # tr/test分割
    tr_len = len(train_data)
    org_len = len(original_data)

    X_train = ohe_matrix[:tr_len + org_len]
    X_test = ohe_matrix[tr_len + org_len:]

    # target / weight 合成
    target = pd.concat([
        train_data["target"],
        original_data["target"]
    ], axis=0).reset_index(drop=True)

    # DataFrameとしては戻せないのでscipy.sparseとして返す
    return {
        "X_train": X_train,       # scipy.sparse.csr_matrix
        "X_test": X_test,         # scipy.sparse.csr_matrix
        "target": target,         # pandas.Series
        "weights": weights        # np.ndarray
    }