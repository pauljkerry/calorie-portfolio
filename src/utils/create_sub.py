import numpy as np
import pandas as pd


def create_sub(test_proba, path="../output/sub_vn.csv"):
    """
    Kaggleの提出用のフォーマットに整形する関数。

    Parameters
    ----------
    test_proba : np.ndarray
        各ラベルについての予測値の配列。
    path : str
        保存先のpath
    """
    preds = np.expm1(test_proba)
    sub_df = pd.DataFrame({
        "id": np.arange(750000, 750000 + len(test_proba)),
        "Calories": preds
    })
    sub_df.to_csv(path, index=False)
    print(f"Saved model successfully to {path}!")