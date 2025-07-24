import optuna
from src.models.cb.cb_cv_trainer import CBCVTrainer


def create_objective(
    tr_df,
    n_splits=5,
    early_stopping_rounds=200,
    n_jobs=1,
    task_type="GPU"
):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.02),
            "depth": trial.suggest_int("depth", 10, 16),
            # "rsm": trial.suggest_float("rsm", 0.2, 0.4),
            # "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "min_data_in_leaf": trial.suggest_float(
                "min_data_in_leaf", 10, 100),
            "l2_leaf_reg": trial.suggest_float(
                "l2_leaf_reg", 1e-2, 20.0
            ),
            "bagging_temperature": trial.suggest_float(
                "bagging_temperature", 1e-2, 1.0
            ),
            "random_strength": trial.suggest_int(
                "random_strength", 1, 50
            ),
            "border_count": trial.suggest_int(
                "border_count", 64, 255
            ),
            "task_type": task_type,
            "early_stopping_rounds": early_stopping_rounds,
        }

        trainer = CBCVTrainer(
            params=params,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds
        )

        trainer.fit_one_fold(tr_df, fold=0)

        return trainer.fold_scores[0]
    return objective


def run_optuna_search(
    objective, n_trials=50, n_jobs=1, study_name="cb_study",
    storage=None, initial_params: dict = None, sampler=None
):
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

# random_sampler = optuna.samplers.RandomSampler()