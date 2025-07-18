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
    weight : float
        originalデータのweight

    Returns
    -------
    tr_df : pd.DataFrame
        特徴量エンジニアリング済みの学習用データ
    test_df : pd.DataFrame
        特徴エンジニアリング済みのテスト用データ

    Notes
    -----
    - logreg用
    """
    # 全データを結合（train + original + test）
    all_data = pd.concat(
        [train_data, test_data], ignore_index=True
    )

    # === 1) カテゴリー変数をOne hot encoding ===
    cat_cols = ["Sex"]
    cat_all_df = pd.DataFrame(index=all_data.index)

    for c in cat_cols:
        encoder = OneHotEncoder(
            sparse_output=False, dtype=int, handle_unknown='ignore'
        )
        ohe_array = encoder.fit_transform(all_data[[c]])
        ohe_df = pd.DataFrame(
            ohe_array,
            columns=[f"{c}_{cat}" for cat in encoder.categories_[0]],
            index=all_data.index
        )
        cat_all_df = pd.concat([cat_all_df, ohe_df], axis=1)

    # === 2) targetをoofとの残差をbin分割したものにする ===
    oof = np.load("../artifacts/oof/single/oof_single_3.npy")
    residual = train_data["target"].values - oof

    bins = [-np.inf, -0.1, 0.1, np.inf]
    residual_bins = np.digitize(residual, bins[1:])

    # === 3) 数値変数を標準化
    num_df = all_data.select_dtypes(
        include=np.number
    ).drop("target", axis=1)
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(num_df)

    scaled_df = pd.DataFrame(
        scaled_array, columns=num_df.columns, index=all_data.index
    )

    # === dfを結合 ===
    df_feat = pd.concat([scaled_df, cat_all_df], axis=1)

    # === データを分割 ===
    tr_df = df_feat.iloc[:len(train_data)].copy()
    test_df = df_feat.iloc[len(train_data):]

    # === target と weight を追加 ===
    tr_df["target"] = residual_bins
    tr_df["weight"] = 1

    return tr_df, test_df