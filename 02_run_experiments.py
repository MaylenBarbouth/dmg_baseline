# 02_run_experiments.py
import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.logger_utils import (
    atomic_write_dataframe,
    atomic_write_json,
    setup_loggers,
)
from src.metrics import (
    evaluate_predictions,
    find_optimal_thresholds_on_validation,
)
from src.optimizer import check_parameter_boundaries, optimize_and_get_oof
from src.pipelines import (
    build_estimator,
    extract_feature_importances,
    get_active_features,
)


def build_task_queue(config: Dict, filter_cohort: str, filter_val: str) -> List[Dict]:
    """Construye la lista ordenada de todas las corridas individuales (folds y semillas)."""
    cohorts = ["A_all", "B_no_antc"]
    if filter_cohort != "all":
        cohorts = [c for c in cohorts if c.startswith(filter_cohort)]

    feature_sets = list(config["columns"]["feature_sets"].keys())
    models = ["logreg", "lgbm"]
    weights = ["none", "balanced"]

    tasks = []
    for cohort in cohorts:
        for fset in feature_sets:
            for model in models:
                for w_mode in weights:
                    # 1. Tareas de Validación Cruzada 5x5
                    if filter_val in ("all", "cv"):
                        for rep in range(config["validation"]["cv_n_repeats"]):
                            for fold in range(config["validation"]["cv_n_splits"]):
                                seed = (
                                    config["validation"]["cv_base_seed"]
                                    + rep * 100
                                    + fold
                                )
                                run_id = f"{cohort}__{fset}__{model}__{w_mode}__cv__rep{rep:02d}_fold{fold:02d}"
                                tasks.append(
                                    {
                                        "run_id": run_id,
                                        "cohort": cohort,
                                        "feature_set": fset,
                                        "model_type": model,
                                        "class_weight_mode": w_mode,
                                        "val_scheme": "repeated_group_cv",
                                        "repeat_idx": rep,
                                        "fold_idx": fold,
                                        "seed": seed,
                                    }
                                )
                    # 2. Tareas de Auditoría Temporal (Out-of-Time con múltiples semillas)
                    if filter_val in ("all", "temporal"):
                        for s_idx in range(config["validation"]["temporal_n_seeds"]):
                            seed = (
                                config["validation"]["temporal_base_seed"]
                                + s_idx
                            )
                            run_id = f"{cohort}__{fset}__{model}__{w_mode}__oot__seed{s_idx:02d}"
                            tasks.append(
                                {
                                    "run_id": run_id,
                                    "cohort": cohort,
                                    "feature_set": fset,
                                    "model_type": model,
                                    "class_weight_mode": w_mode,
                                    "val_scheme": "temporal_oot",
                                    "repeat_idx": 0,
                                    "fold_idx": s_idx,
                                    "seed": seed,
                                }
                            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor de Experimentos Baseline DMG")
    parser.add_argument(
        "--cohort",
        choices=["all", "A", "B"],
        default="all",
        help="Filtrar por cohorte (A, B o all)",
    )
    parser.add_argument(
        "--val",
        choices=["all", "cv", "temporal"],
        default="all",
        help="Filtrar por esquema de validación",
    )
    args = parser.parse_args()

    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    outputs_dir = Path(config["paths"]["outputs_dir"])
    proc_dir = Path(config["paths"]["processed_dir"])
    (outputs_dir / "optuna_db").mkdir(parents=True, exist_ok=True)
    shards_dir = outputs_dir / "shards"

    logger = setup_loggers(str(outputs_dir))

    # Cargar datasets y mapas de partición
    cohorts_data = {
        "A_all": pd.read_parquet(proc_dir / "cohort_A_general.parquet"),
        "B_no_antc": pd.read_parquet(proc_dir / "cohort_B_sin_antc.parquet"),
    }
    splits_cv = pd.read_parquet(proc_dir / "splits_cv_5x5.parquet")
    splits_temp = pd.read_parquet(proc_dir / "splits_temporal.parquet")

    preg_col = config["schema"]["pregnancy_id"]
    pat_col = config["schema"]["patient_id"]
    target_col = config["schema"]["target_col"]
    date_col = config["schema"]["date_col"]

    tasks = build_task_queue(config, args.cohort, args.val)
    total_tasks = len(tasks)
    logger.info(f"Iniciando cola de ejecución: {total_tasks} corridas planificadas.")

    completed_times = []

    for idx, task in enumerate(tasks, start=1):
        run_id = task["run_id"]
        metric_shard_path = shards_dir / "metrics" / f"{run_id}.json"

        # REANUDACIÓN AUTOMÁTICA: Si el shard ya existe, se salta sin recalcular
        if metric_shard_path.exists():
            logger.info(
                f"[{idx:03d}/{total_tasks:03d}] [SKIP] {run_id} (Ya completado en disco)"
            )
            continue

        t0 = time.time()
        try:
            df = cohorts_data[task["cohort"]]
            active_cols = get_active_features(
                task["cohort"],
                task["feature_set"],
                task["model_type"],
                config,
            )

            # Obtener índices de Train y Test según el esquema
            if task["val_scheme"] == "repeated_group_cv":
                mask_split = (
                    (splits_cv["cohort"] == task["cohort"])
                    & (splits_cv["repeat_idx"] == task["repeat_idx"])
                    & (splits_cv["fold_idx"] == task["fold_idx"])
                )
                test_preg_ids = set(splits_cv.loc[mask_split, preg_col])
                is_test = df[preg_col].isin(test_preg_ids)
                df_train = df[~is_test].reset_index(drop=True)
                df_test = df[is_test].reset_index(drop=True)
            else:
                sub_temp = splits_temp[splits_temp["cohort"] == task["cohort"]]
                train_ids = set(
                    sub_temp.loc[sub_temp["role"] == "train", preg_col]
                )
                test_ids = set(
                    sub_temp.loc[sub_temp["role"] == "test", preg_col]
                )
                df_train = df[df[preg_col].isin(train_ids)].reset_index(
                    drop=True
                )
                df_test = df[df[preg_col].isin(test_ids)].reset_index(drop=True)

            # Chequeo de seguridad anti-fuga de pacientes en vivo
            leaks = len(
                set(df_train[pat_col]).intersection(set(df_test[pat_col]))
            )
            assert leaks == 0, f"Fuga detectada: {leaks} pacientes compartidas!"

            X_train = df_train[active_cols]
            y_train = df_train[target_col].values.astype(int)
            groups_train = df_train[pat_col].values

            X_test = df_test[active_cols]
            y_test = df_test[target_col].values.astype(int)

            # Estimación de ETA
            eta_str = "Calculando..."
            if completed_times:
                avg_sec = float(np.mean(completed_times[-20:]))
                rem_sec = avg_sec * (total_tasks - idx + 1)
                eta_str = f"~{rem_sec / 3600:.1f}h"

            logger.info("=" * 80)
            logger.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"PROGRESO: [ {idx} / {total_tasks} ] ({idx / total_tasks * 100:.1f}%) | ETA: {eta_str}"
            )
            logger.info(
                f"▶ CONFIG : Cohorte={task['cohort']} | Set={task['feature_set']} ({len(active_cols)} vars) | "
                f"Modelo={task['model_type'].upper()} ({task['class_weight_mode']})"
            )
            logger.info(
                f"▶ ESQUEMA: {task['val_scheme']} | Rep={task['repeat_idx']} | Fold/Seed={task['fold_idx']} (Seed: {task['seed']})"
            )
            logger.info(
                f"▶ DATOS  : Train={len(df_train)} (Pos: {y_train.sum()} [{y_train.mean()*100:.1f}%]) | "
                f"Test={len(df_test)} (Pos: {y_test.sum()} [{y_test.mean()*100:.1f}%]) | Leaks ID: {leaks}"
            )

            # 1. Optimización de Hiperparámetros en CV Interno
            (
                best_params,
                best_inner_prauc,
                oof_val_probs,
                trials_df,
            ) = optimize_and_get_oof(
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                model_type=task["model_type"],
                class_weight_mode=task["class_weight_mode"],
                run_id=run_id,
                seed=task["seed"],
                config=config,
            )

            # 2. Calibración de umbrales (Youden y Sens80) estrictamente en validación interna
            thresh_youden, thresh_sens80 = find_optimal_thresholds_on_validation(
                y_train, oof_val_probs
            )

            # 3. Entrenamiento del modelo final sobre todo el bloque Train del fold
            final_pipe = build_estimator(
                model_type=task["model_type"],
                class_weight_mode=task["class_weight_mode"],
                params=best_params,
                y_train=y_train,
                seed=task["seed"],
                n_jobs=config["optimization"]["n_jobs_per_model"],
            )
            final_pipe.fit(X_train, y_train)
            y_test_prob = final_pipe.predict_proba(X_test)[:, 1]

            # 4. Evaluación en Test y Auditoría de Bordes de Grilla
            eval_metrics = evaluate_predictions(
                y_test, y_test_prob, thresh_youden, thresh_sens80
            )
            boundary_info = check_parameter_boundaries(
                best_params,
                task["model_type"],
                tolerance=config["optimization"]["boundary_tolerance"],
            )

            elapsed = time.time() - t0
            completed_times.append(elapsed)

            # Armar registro maestro crudo del fold
            fold_record = {
                **task,
                "timestamp": datetime.now().isoformat(),
                "n_features_used": len(active_cols),
                "n_train_rows": int(len(df_train)),
                "n_train_pos": int(y_train.sum()),
                "n_train_patients": int(df_train[pat_col].nunique()),
                "n_test_rows": int(len(df_test)),
                "n_test_pos": int(y_test.sum()),
                "n_test_patients": int(df_test[pat_col].nunique()),
                "best_inner_cv_prauc": best_inner_prauc,
                **eval_metrics,
                **boundary_info,
                "best_params_json": json.dumps(best_params),
                "elapsed_sec": round(elapsed, 2),
            }

            # Predicciones individuales en Test (para curvas ROC/PR y calibración posterior)
            preds_df = pd.DataFrame(
                {
                    "run_id": run_id,
                    "cohort": task["cohort"],
                    "feature_set": task["feature_set"],
                    "model_type": task["model_type"],
                    "class_weight_mode": task["class_weight_mode"],
                    "val_scheme": task["val_scheme"],
                    "repeat_idx": task["repeat_idx"],
                    "fold_idx": task["fold_idx"],
                    preg_col: df_test[preg_col].values,
                    pat_col: df_test[pat_col].values,
                    date_col: df_test[date_col].astype(str).values,
                    "y_true": y_test,
                    "y_prob_pred": y_test_prob,
                }
            )

            # Importancia de variables
            imp_df = extract_feature_importances(
                final_pipe, active_cols, task["model_type"], run_id
            )

            # Escritura atómica de shards (el shard de métricas se escribe último como sello de completitud)
            atomic_write_dataframe(
                preds_df, shards_dir / "predictions" / f"{run_id}.parquet"
            )
            atomic_write_dataframe(
                imp_df, shards_dir / "importances" / f"{run_id}.csv"
            )
            atomic_write_dataframe(
                trials_df, shards_dir / "trials" / f"{run_id}.csv"
            )
            atomic_write_json(fold_record, metric_shard_path)

            logger.info(
                f"  └── Optuna ({config['optimization']['n_trials']} trials, {elapsed:.1f}s) | Best Inner PR-AUC: {best_inner_prauc:.4f}"
            )
            logger.info(
                f"  └── Test Eval -> PR-AUC: {eval_metrics['test_pr_auc']:.4f} | ROC-AUC: {eval_metrics['test_roc_auc']:.4f} | "
                f"Brier: {eval_metrics['test_brier_score']:.4f} | Sens(Youden): {eval_metrics['sens_youden']:.3f} | Spec(Youden): {eval_metrics['spec_youden']:.3f}"
            )
            logger.info(
                f"  └── Bordes Grilla: {boundary_info['n_params_at_boundary']} [{boundary_info['boundary_params_detail']}] | "
                f"Apagados: [{boundary_info['off_params_detail']}]"
            )

        except Exception as e:
            logger.error(
                f"FALLO CRÍTICO en {run_id}: {str(e)}\n{traceback.format_exc()}"
            )


if __name__ == "__main__":
    main()