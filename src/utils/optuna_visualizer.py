import optuna
import optuna.visualization as vis
import pandas as pd


class OptunaVisualizer:
    """
    Optunaの探索結果を表示するクラス

    Attributes
    ----------
    study_name : str
        StudyName
    storage_path : str
        ストレージの保存先
    """

    def __init__(self, study_name, storage_path):
        self.study_name = study_name
        self.storage_path = storage_path
        self.study = optuna.load_study(
            study_name=study_name,
            storage=storage_path
        )

    def visualize_optimization(self):
        """
        探索結果を可視化する関数。

        Notes
        -----
        - パラメータ重要度
        - 最適化履歴
        - パラメータの相互関係
        """
        # パラメータ重要度
        fig1 = vis.plot_param_importances(self.study)
        fig1.show()

        # 最適化履歴
        fig2 = vis.plot_optimization_history(self.study)
        fig2.show()

        # パラメータの相互依存関係
        fig3 = vis.plot_parallel_coordinate(self.study)
        fig3.show()

    def print_trials_table(self, top_k=10):
        """
        これまでのtrialの結果の表示

        Parameters
        ----------
        top_k : int, default 10
            表示するtrialの数

        Notes
        -----
        - iteration数の表示
        - パラメータの表示
        """
        trials = [t for t in self.study.trials if t.value is not None]
        sorted_trials = sorted(trials, key=lambda t: t.value, reverse=False)

        for i, t in enumerate(sorted_trials[:top_k]):
            print(f"=== Trial {t.number} ===")
            print(f"CV Score       : {t.value:.5f}")
            print(f"Best Iteration : {t.user_attrs.get('best_iteration', 'N/A')}")
            print("Params:")
            print("params = {")
            for i, (k, v) in enumerate(t.params.items()):
                comma = "," if i < len(t.params) - 1 else ""
                print(f'    "{k}": {v}{comma}')
            print("}")

    def summarize_studies(self):
        """
        各studyの概要を表示

        Returns
        -------
        df : pd.DataFrame
            概要のdf

        Notes
        -----
        - StudyName
        - トライアル数
        - 最良のスコア
        - 最適化の方向
        - 実験開始時の日時
        """
        summaries = optuna.study.get_all_study_summaries(storage=self.storage_path)

        data = []
        for s in summaries:
            best_score = s.best_trial.value if s.best_trial else None
            data.append({
                "study_name": s.study_name,
                "n_trials": s.n_trials,
                "best_score": best_score,
                "direction": s.direction.name,
                "datetime_start": s.datetime_start.strftime("%Y-%m-%d %H:%M:%S") if s.datetime_start else None
            })

        df = pd.DataFrame(data).sort_values("best_score", ascending=False)
        return df

    def get_best_params(self):
        """
        最良のパラメータの取得

        Returns
        -------
        best_params : dict
            最良のtrialのパラメータ
        """
        return self.study.best_params