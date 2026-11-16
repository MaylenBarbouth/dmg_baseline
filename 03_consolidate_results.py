# 03_consolidate_results.py
import json
from pathlib import Path

import pandas as pd
import yaml

from src.logger_utils import atomic_write_dataframe


def main() -> None:
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    outputs_dir = Path(config["paths"]["outputs_dir"])
    shards_dir = outputs_dir / "shards"
    cons_dir = outputs_dir / "consolidated"
    cons_dir.mkdir(parents=True, exist_ok=True)

    # 1. Consolidar métricas por fold
    metric_files = sorted((shards_dir / "metrics").glob("*.json"))
    if not metric_files:
        print("Todavía no hay shards completados en outputs/shards/metrics/.")
        return

    metrics_list = []
    for mf in metric_files:
        with open(mf, "r", encoding="utf-8") as f:
            metrics_list.append(json.load(f))
    df_metrics = pd.DataFrame(metrics_list)
    atomic_write_dataframe(df_metrics, cons_dir / "master_fold_metrics.csv")
    print(
        f"✔ master_fold_metrics.csv generado ({len(df_metrics)} corridas individuales sin promediar)."
    )

    # 2. Consolidar predicciones individuales
    pred_files = sorted((shards_dir / "predictions").glob("*.parquet"))
    if pred_files:
        df_preds = pd.concat(
            [pd.read_parquet(pf) for pf in pred_files], ignore_index=True
        )
        atomic_write_dataframe(
            df_preds, cons_dir / "master_oof_predictions.parquet"
        )
        print(
            f"✔ master_oof_predictions.parquet generado ({len(df_preds)} predicciones individuales)."
        )

    # 3. Consolidar importancia de variables
    imp_files = sorted((shards_dir / "importances").glob("*.csv"))
    if imp_files:
        df_imp = pd.concat(
            [pd.read_csv(ifp) for ifp in imp_files], ignore_index=True
        )
        atomic_write_dataframe(
            df_imp, cons_dir / "master_feature_importances.csv"
        )
        print(
            f"✔ master_feature_importances.csv generado ({len(df_imp)} filas)."
        )

    # 4. Consolidar historial de trials de Optuna
    trial_files = sorted((shards_dir / "trials").glob("*.csv"))
    if trial_files:
        df_trials = pd.concat(
            [pd.read_csv(tf) for tf in trial_files], ignore_index=True
        )
        atomic_write_dataframe(
            df_trials, cons_dir / "master_optuna_trials.csv"
        )
        print(
            f"✔ master_optuna_trials.csv generado ({len(df_trials)} trials registrados)."
        )


if __name__ == "__main__":
    main()