from src.models.lgbm.lgbm_cv_trainer import LGBMCVTrainer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def ensemble_with_seeds(
    tr_df, test_df, params, cat_cols, seeds, ID, n_splits=5
):
    """
    複数の乱数シードを用いたアンサンブル学習を実行する関数。

    Parameters
    ----------
    tr_df : pd.DataFrame
        学習用データ。
    test_df : pd.DataFrame
        テスト用データ。
    params : dict
        ハイパーパラメータ。
    cat_cols : list
        カテゴリ変数名のリスト。
    seeds : list
        使用する乱数シードのリスト。
    ID : str
        予測結果を保存する際の識別子。
    n_splits : int, default 5
        CVの分割数。

    Returns
    -------
    avg_oof : np.ndarray
        全シードで平均されたOOF予測。
    avg_test : np.ndarray
        全シードで平均されたテスト予測。
    """
    label_encoder = joblib.load("../artifacts/label_encoder.pkl")
    n_classes = len(label_encoder.classes_)
    oof_sum = np.zeros((len(tr_df), n_classes))
    test_sum = np.zeros((len(test_df), n_classes))

    # 結果を保存するリスト
    scores_history = []

    for i, seed in enumerate(seeds):
        trainer = LGBMCVTrainer(
            params=params, n_splits=n_splits, seed=seed,
            cat_cols=cat_cols)
        oof_preds, test_preds = trainer.fit(
            tr_df.copy(), test_df.copy())

        oof_sum += oof_preds
        test_sum += test_preds

        if i == 0:
            oof_path = (
                f"../artifacts/oof/"
                f"single/oof_single_{ID}.npy"
            )
        else:
            oof_path = (
                f"../artifacts/oof/"
                f"repeated/oof_repeated_{i+1}_{ID}.npy"
            )
        np.save(oof_path, oof_sum)

        if i == 0:
            test_path = (
                f"../artifacts/test_preds/"
                f"single/test_single_{ID}.npy"
                )
        else:
            test_path = (
                f"../artifacts/"
                f"test_preds/repeated/test_repeated_{i+1}_{ID}.npy"
            )
        np.save(test_path, test_sum)

        # oofスコアのアンサンブル結果のモニタリング
        avg_oof = oof_sum / (i + 1)
        y_true = label_encoder.transform(tr_df["target"].to_numpy())
        map3_score = map_k(y_true, avg_oof)

        # 結果をリストに保存
        scores_history.append({
            'seed_count': i + 1,
            'map3_score': map3_score
        })

        print(f"[{i+1} seeds] Ensemble MAP@3: {map3_score:.5f}")

    # 可視化
    plot_ensemble_progress(scores_history)

    # 最終の test 平均
    avg_test = test_sum / len(seeds)
    return avg_oof, avg_test


def plot_ensemble_progress(scores_history):
    """
    アンサンブルにおけるMAP@3スコアの推移を可視化する関数。

    Parameters
    ----------
    scores_history : list of dict
        seed_count'と'map3_score'を記録した辞書のリスト。
    """
    df = pd.DataFrame(scores_history)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='seed_count', y='map3_score', marker='o')
    plt.title('Ensemble MAP@3 Score Progress')
    plt.xlabel('Number of Seeds')
    plt.ylabel('MAP@3 Score')
    plt.grid(True, alpha=0.3)
    plt.show()