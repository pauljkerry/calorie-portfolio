import optuna
from src.models.xgb.xgb_cv_trainer import XGBCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    early_stopping_rounds=200,
    n_jobs=1,
    tree_method="gpu_hist",
    cat_cols=None
):
    """
    Optunaの目的関数（objective）を生成する関数。

    Parameters
    ----------
    tr_df : pd.DataFrame
        訓練データ。
    n_splits : int, default 5
        cv分割数。
    early_stopping_rounds : int, default 200
        EarlyStoppingのラウンド数。
    n_jobs : int, default 1
        xgb並列数。
    tree_mehod: str, default "gpu_hist"
        使用するtree_method。
    cat_cols : list, default None
        カテゴリ変数名のリスト。

    Returns
    -------
    function
        Optunaで使用する目的関数。
    """
    def objective(trial):
        params = {
            "learning_rate": 0.02,
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_weight": trial.suggest_float("min_child_weight",
                                                    5, 100),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 0.85),
            "subsample": trial.suggest_float("subsample",
                                             0.5, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha",
                                             1e-4, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda",
                                              1e-4, 10.0, log=True),
            "tree_method": tree_method,
            "n_jobs": n_jobs
        }

        trainer = XGBCVTrainer(
            params=params, n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            cat_cols=cat_cols
        )

        trainer.fit_one_fold(tr_df, fold=0)

        best_iteration = trainer.fold_models[0].model.best_iteration
        trial.set_user_attr("best_iteration", best_iteration)

        return trainer.fold_scores[0]
    return objective


def run_optuna_search(
    objective, n_trials=50, n_jobs=1, study_name="xgb_study",
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
    study_name : str or None, default "xgb_study"
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