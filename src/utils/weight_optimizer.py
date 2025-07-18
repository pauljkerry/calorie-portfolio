import numpy as np
import pandas as pd
import optuna
import joblib
from src.utils.map_k import map_k


def create_objective(oof1, oof2, oof3):
    """
    oofの最適な重みを探索する関数

    Parameters
    ----------
    oof1 : np.ndarray
        各ラベルの予測確率をまとめた配列。
    oof2 : np.ndarray
        各ラベルの予測確率をまとめた配列。
    oof3 : np.ndarray
        各ラベルの予測確率をまとめた配列。

    Returns
    -------
    objective : function
        optunaで使用する目的関数。
    """
    label_encoder = (
        joblib.load("../artifacts/label_encoder.pkl")
    )
    train_data = pd.read_csv("../artifacts/features/tr_df4.csv")
    target = train_data["target"].to_numpy()
    y_true = label_encoder.transform(target)

    def objective(trial):
        # 3つの重みを[0, 1]でサンプリング
        w1 = trial.suggest_float("w1", 0.0, 0.5)
        w2 = trial.suggest_float("w2", 0.0, 0.5)
        w3 = 1.0 - w1 - w2

        if w3 < 0:
            return -np.inf

        # アンサンブル予測の作成
        y_pred = w1 * oof1 + w2 * oof2 + w3 * oof3

        # 評価指標の計算（できれば負方向にする）
        score = map_k(y_true, y_pred)
        return score
    return objective


def run_optuna_search(
    objective, n_trials=50, n_jobs=1, study_name="weight_study",
    storage=None, initial_params: dict = None, sampler=None
):
    """
    Optunaによるハイパーパラメータ探索を実行する関数。

    Parameters
    ----------
    objective : function
        Optunaの目的関数。
    n_trials : int, default 50
        試行回数。
    n_jobs : int, default 1
        並列実行数。
    study_name : str or None, default "weight_study"
        StudyName。
    storage : str or None, default None
        保存先URL。
    initial_params : dict or None, default None
        初期の試行パラメータ。
    sampler : optuna.samplers.BaseSampler or None, default TPESampler
        使用するSampler。

    Returns
    -------
    study : optuna.Study
        探索結果のStudyオブジェクト。
    """
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler or optuna.samplers.TPESampler()
    )

    if initial_params is not None:
        study.enqueue_trial(initial_params)

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True
    )

    return study