import pandas as pd
import numpy as np


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
    """
    all_data = pd.concat([train_data, test_data])
    all_data = all_data.drop("id", axis=1)

    all_data = all_data.rename(columns={
        "Calories": "target"
    })

    all_data["target"] = np.log1p(all_data["target"])

    tr_df = all_data.iloc[:len(train_data)]
    test_df = all_data.iloc[len(train_data):]

    return tr_df, test_df