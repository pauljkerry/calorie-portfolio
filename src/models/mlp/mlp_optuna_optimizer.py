import optuna
import torch.nn as nn
from src.models.mlp.mlp_cv_trainer import MLPCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    epochs=100,
    early_stopping_rounds=10,
    use_gpu=True,
    activation="ReLU"
):
    """
    Optunaの目的関数（objective）を生成する関数。

    Parameters
    ----------
    tr_df : pd.DataFrame
        訓練データ。
    n_splits : int, default 5
        cv分割数。
    epochs : int, default 100
        エポック数。
    early_stopping_rounds : int, default 200
        早期停止ラウンド数。
    use_gpu : bool, default True
        Trueの場合はGPUが使用可能であれば使用する。
    activation : nn.Module, default ReLU
        活性化関数のクラス

    Returns
    -------
    function
        Optunaで使用する目的関数。
    """
    def objective(trial):
        trainer = MLPCVTrainer()
        trainer.n_splits = n_splits
        trainer.epochs = epochs
        trainer.early_stopping_rounds = early_stopping_rounds
        trainer.use_gpu = use_gpu
        trainer.batch_size = trial.suggest_int(
            "batch_size", 512, 1024, step=32
        )
        trainer.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        trainer.dropout_rate = round(trial.suggest_float(
            "dropout_rate", 0.1, 0.3, step=0.01), 2)

        activation_str = trial.suggest_categorical(
            "activation", [activation])
        activation_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "LeakyRelu": nn.LeakyReLU
        }
        trainer.activation = activation_map[activation_str]()

        dim1 = trial.suggest_int("hidden_dim1", 96, 512, step=32)
        dim2 = trial.suggest_int("hidden_dim2", 64, dim1, step=32)
        dim3 = trial.suggest_int("hidden_dim3", 32, dim2, step=32)
        trainer.hidden_dims = [dim1, dim2, dim3]

        trainer.fit_one_fold(tr_df, fold=0)
        best_rounds = trainer.fold_models[0].best_rounds
        trial.set_user_attr("best_round", best_rounds)

        return trainer.fold_scores[0]
    return objective


def run_optuna_search(
    objective, n_trials=50, n_jobs=1, study_name="mlp_study",
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
    study_name : str or None, default "mlp_study"
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