import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error


def create_objective(oof_list):
    """
    Optunaの目的関数（objective）を生成する関数。

    Parameters
    ----------
    oof_list : list of np.ndarray
        各モデルの予測値（oof）をまとめたリスト

    Returns
    -------
    objective : function
        optunaで使う目的関数
    """
    train_data = pd.read_parquet("../artifacts/prepro/train_data1.parquet")
    y_true = train_data["target"].to_numpy()
    n_models = len(oof_list)

    def objective(trial):
        raw_weights = [trial.suggest_float(f"raw_w{i}", 0.0, 1.0)
                       for i in range(n_models)]
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]  # 正規化

        y_pred = sum(w * oof for w, oof in zip(weights, oof_list))
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return rmse

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
        direction="minimize",
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