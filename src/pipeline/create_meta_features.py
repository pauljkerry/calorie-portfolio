import pandas as pd
import numpy as np


def create_meta_features(ID_list, fe_version=None):
    """
    stackingのためのメタ特徴量を作成する

    Parameters
    ----------
    ID_list : list
        使用する予測値のID
    fe_version : str, default None
        特徴量エンジニアリングのversion名

    Returns
    -------
    tr_df : pd.DataFrame
        メタ特徴量の学習用データ
    test_df : pd.DataFrame
        メタ特徴量のテスト用データ

    Notes
    -----
    - 特徴量エンジニアリングはせず
    """
    # 予測値の読み込みと結合
    tr_list = []
    test_list = []

    for ID in ID_list:
        array_tr = np.load(f"../artifacts/oof/single/oof_single_{ID}.npy")
        array_test = np.load(
            f"../artifacts/test_preds/single/test_single_{ID}.npy"
        )
        tr_list.append(pd.DataFrame(array_tr, columns=[f"pred_{ID}"]))
        test_list.append(pd.DataFrame(array_test, columns=[f"pred_{ID}"]))

    tr_df = pd.concat(tr_list, axis=1)
    test_df = pd.concat(test_list, axis=1)

    # 元データの特徴量エンジニアリングを追加する場合
    if fe_version is not None:
        tr_fe = pd.read_parquet(
            f"../artifacts/features/tr_df{fe_version}.parquet"
        )
        test_fe = pd.read_parquet(
            f"../artifacts/features/test_df{fe_version}.parquet"
        )
        tr_df = pd.concat([tr_fe, tr_df], axis=1)
        test_df = pd.concat([test_fe, test_df], axis=1)

    # targetの追加
    train_data = pd.read_csv("../artifacts/prepro/train_data1.csv")
    tr_df["target"] = train_data["target"]

    return tr_df, test_df