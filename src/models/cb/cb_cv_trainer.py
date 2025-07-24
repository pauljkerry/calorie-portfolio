from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import time
from src.utils.print_duration import print_duration


class CBCVTrainer:
    def __init__(self, params=None, n_splits=5,
                 early_stopping_rounds=100, seed=42):
        self.params = params or {}
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.fold_models = []
        self.fold_scores = []
        self.seed = seed
        self.oof_score = None

    def get_default_params(self):
        default_params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "learning_rate": 0.1,
            "depth": 6,
            "iterations": 10000,
            "min_data_in_leaf": 1,
            "l2_leaf_reg": 3.0,
            "bagging_temperature": 1,
            "random_strength": 10,
            "border_count": 128,
            "grow_policy": "SymmetricTree",
            "random_seed": self.seed,
            "verbose": 100,
            "task_type": "GPU",  # or CPU
            "early_stopping_rounds": 100,
            "allow_writing_files": False,
            "verbose": 100  # ログ出力周期
        }
        return default_params

    def fit(self, tr_df, test_df):
        tr_df = tr_df.copy()
        test_df = test_df.copy()

        if "weight" in tr_df.columns:
            weights = tr_df["weight"].astype("float32")
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = pd.Series(
                np.ones(len(tr_df), dtype="float32"),
                index=tr_df.index
            )

        X = tr_df.drop("target", axis=1)
        y = tr_df["target"]

        default_params = self.get_default_params()
        self.params = {**default_params, **self.params}

        cat_cols = tr_df.select_dtypes(include="object").columns.to_list()

        oof_preds = np.zeros(len(tr_df))
        test_preds = np.zeros(len(test_df))

        kf = KFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.seed)

        iteration_list = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold + 1}")
            start = time.time()

            X_tr, y_tr, w_tr = (
                X.iloc[tr_idx],
                y.iloc[tr_idx],
                weights.iloc[tr_idx]
            )
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            train_pool = Pool(
                X_tr, y_tr,
                cat_features=cat_cols,
                weight=w_tr
            )
            val_pool = Pool(
                X_val, y_val,
                cat_features=cat_cols
            )

            model = CatBoostRegressor(**self.params)

            model.fit(
                train_pool, eval_set=val_pool, use_best_model=True
            )

            val_preds = model.predict(X_val)
            oof_preds[val_idx] = val_preds

            test_preds += model.predict(test_df)

            best_iteration = model.best_iteration_
            evals_result = model.evals_result_
            train_rmse = evals_result["learn"]["RMSE"][best_iteration]
            valid_rmse = evals_result["validation"]["RMSE"][best_iteration]

            print(f"Train rmse: {train_rmse:.5f}")
            print(f"Valid rmse: {valid_rmse:.5f}")

            end = time.time()
            print_duration(start, end)

            fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
            print(f"Valid rmse: {fold_score:.5f}")

            self.fold_models.append(
                CBFoldModel(
                    model, X_val, y_val, fold
                ))
            self.fold_scores.append(fold_score)

            iteration_list.append(best_iteration)

        print("\n=== CV 結果 ===")
        print(f"Fold scores: {self.fold_scores}")
        print(
            f"Mean: {np.mean(self.fold_scores):.5f}, "
            f"Std: {np.std(self.fold_scores):.5f}"
        )

        self.oof_score = np.sqrt(mean_squared_error(y, oof_preds))
        print(f" OOF score: {self.oof_score:.5f}")
        print(f"Avg best iteration: {np.mean(iteration_list)}")
        print(f"Best iterations: \n{iteration_list}")

        test_preds /= self.n_splits

        return oof_preds, test_preds

    def full_train(self, tr_df, test_df, iterations):
        tr_df = tr_df.copy()
        test_df = test_df.copy()

        if "weight" in tr_df.columns:
            weights = tr_df["weight"].astype("float32")
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = pd.Series(
                np.ones(len(tr_df), dtype="float32"),
                index=tr_df.index
            )

        X = tr_df.drop("target", axis=1)
        y = tr_df["target"].values

        default_params = self.get_default_params()
        self.params = {**default_params, **self.params}

        cat_cols = tr_df.select_dtypes(include="object").columns.to_list()

        train_pool = Pool(
            X, y, cat_features=cat_cols, weight=weights)

        start = time.time()

        model = CatBoostRegressor(**self.params)

        model.fit(
            train_pool,
            iterations=int(iterations * 1.25),
            use_best_model=True
        )

        end = time.time()
        print_duration(start, end)

        self.fold_models.append(
            CBFoldModel(
                model, None, None, None
            ))

        test_preds = model.predict_proba(test_df)
        return test_preds

    def fit_one_fold(self, tr_df, fold=0):
        tr_df = tr_df.copy()

        if "weight" in tr_df.columns:
            weights = tr_df["weight"].astype("float32")
            tr_df = tr_df.drop("weight", axis=1)
        else:
            weights = pd.Series(
                np.ones(len(tr_df), dtype="float32"),
                index=tr_df.index
            )

        X = tr_df.drop("target", axis=1)
        y = tr_df["target"]

        default_params = self.get_default_params()
        self.params = {**default_params, **self.params}

        cat_cols = tr_df.select_dtypes(include="object").columns.to_list()

        kf = KFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.seed)

        tr_idx, val_idx = list(kf.split(X, y))[fold]
        start = time.time()

        X_tr, y_tr, w_tr = (
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            weights.iloc[tr_idx]
        )
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        train_pool = Pool(
            X_tr, y_tr,
            cat_features=cat_cols,
            weight=w_tr
        )
        val_pool = Pool(
            X_val, y_val,
            cat_features=cat_cols
        )

        model = CatBoostRegressor(**self.params)

        model.fit(
            train_pool, eval_set=val_pool, use_best_model=True
        )
        best_iteration = model.best_iteration_
        evals_result = model.evals_result_
        train_rmse = evals_result["learn"]["RMSE"][best_iteration - 1]
        valid_rmse = evals_result["validation"]["RMSE"][best_iteration - 1]

        print(f"Train rmse: {train_rmse:.5f}")
        print(f"Valid rmse: {valid_rmse:.5f}")

        self.fold_models.append(CBFoldModel(
            model,
            X_val,
            y_val,
            fold
        ))
        self.fold_scores.append(valid_rmse)
        end = time.time()
        print_duration(start, end)


class CBFoldModel:
    def __init__(self, model, X_val, y_val, fold_index):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.fold_index = fold_index

    def shap_plot(self, sample=1000):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(self.X_val[:sample])
        shap.summary_plot(shap_values, self.X_val[:sample], max_display=100)

    def plot_gain_importance(self):
        importances = self.model.get_feature_importance(type="Gain")

        total_gain = importances.sum()
        importance_ratios = np.round((importances / total_gain) * 100, 2)
        df = pd.DataFrame({
            "Feature": self.model.feature_names_,
            "ImportanceRatio": importance_ratios
        }).sort_values("ImportanceRatio", ascending=False)

        fig, ax = plt.subplots(figsize=(12, max(4, len(df)*0.4)))

        sns.barplot(
            data=df,
            y="Feature",
            x="ImportanceRatio",
            orient="h",
            palette="viridis",
            hue="Feature",
            ax=ax
        )
        for container in ax.containers:
            labels = ax.bar_label(container)
            for label in labels:
                label.set_fontsize(20)

        plt.title("Feature Importance", fontsize=32)
        plt.xlabel("Importance", fontsize=28)
        plt.ylabel("Feature", fontsize=28)
        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)

        plt.tight_layout()

    def save_model(self, path="../artifacts/model/cb_vn.pkl"):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)
        return self