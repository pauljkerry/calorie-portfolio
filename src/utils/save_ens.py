import numpy as np


def save_ensemble_prediction(test_preds_list, weights_dict, ID):
    """
    アンサンブル予測を計算して保存する。

    Parameters
    ----------
    test_preds_list : list of np.ndarray
        各モデルのテストデータ予測値のリスト
    weights_dict : dict
        比率のパラメータ
    ID : str
        保存ファイルに使用する識別子
    """
    # 重みを順に取り出して正規化
    weights = np.array([weights_dict[f"raw_w{i}"]
                        for i in range(len(test_preds_list))])
    weights /= weights.sum()

    # 加重平均を計算
    ensemble_pred = np.sum([w * pred for w, pred
                            in zip(weights, test_preds_list)], axis=0)

    save_path = f"../artifacts/preds/ens/ens_{ID}.npy"

    np.save(save_path, ensemble_pred)
    print("Ensemble preds saved successfully")