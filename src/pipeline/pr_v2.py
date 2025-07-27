import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocessing(train_data, test_data):
    """
    生データの前処理を行う関数。

    Parameters
    ----------
    train_data : pd.DataFrame
        学習用の生データ。
    test_data : pd.DataFrame
        テスト用の生データ

    Returns
    -------
    train_data : pd.DataFrame
        前処理済みの学習用データ。
    test_data : pd.DataFrame
        前処理済みのテスト用データ。

    Notes
    -----
    - id列を削除
    - 特徴量名をリネーム
    - 主成分分析で相関係数が高いもの同士を圧縮
    """
    all_data = pd.concat([train_data, test_data])
    all_data = all_data.drop("id", axis=1)

    all_data = all_data.rename(columns={
        "Calories": "target"
    })

    all_data["target"] = np.log1p(all_data["target"])

    # 主成分分析による次元圧縮
    pca_cols1 = ["Duration", "Heart_Rate", "Body_Temp"]
    pca_cols2 = ["Height", "Weight"]

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    X_1 = scaler1.fit_transform(all_data[pca_cols1])
    X_2 = scaler2.fit_transform(all_data[pca_cols2])

    pca1 = PCA(n_components=0.95)
    pca2 = PCA(n_components=0.95)

    X_pca1 = pca1.fit_transform(X_1)
    X_pca2 = pca2.fit_transform(X_2)

    # 圧縮された主成分をDataFrameに変換
    pca_df1 = pd.DataFrame(X_pca1, columns=[f"pca1_{i}" for i in range(X_pca1.shape[1])])
    pca_df2 = pd.DataFrame(X_pca2, columns=[f"pca2_{i}" for i in range(X_pca2.shape[1])])

    # 元のカラムは削除
    all_data = all_data.drop(columns=pca_cols1 + pca_cols2)

    # PCAカラム追加
    all_data = pd.concat([all_data.reset_index(drop=True), pca_df1, pca_df2], axis=1)

    tr_df = all_data.iloc[:len(train_data)]
    test_df = all_data.iloc[len(train_data):]

    return tr_df, test_df