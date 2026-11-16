# src/optimizer.py
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from src.pipelines import build_estimator

# Especificación de rangos activos para el detector de bordes de grilla
PARAM_SPECS = {
    "lgbm": {
        "n_estimators": {"low": 50, "high": 1000, "scale": "linear", "off": None},
        "learning_rate": {"low": 0.005, "high": 0.15, "scale": "log", "off": None},
        "max_depth": {"low": 3, "high": 10, "scale": "linear", "off": -1},
        "num_leaves": {"low": 7, "high": 63, "scale": "linear", "off": None},
        "min_child_samples": {"low": 10, "high": 150, "scale": "linear", "off": 1},
        "subsample": {"low": 0.5, "high": 0.99, "scale": "linear", "off": 1.0},
        "colsample_bytree": {"low": 0.3, "high": 0.99, "scale": "linear", "off": 1.0},
        "reg_alpha": {"low": 1e-4, "high": 20.0, "scale": "log", "off": 0.0},
        "reg_lambda": {"low": 1e-4, "high": 20.0, "scale": "log", "off": 0.0},
        "min_split_gain": {"low": 0.01, "high": 1.0, "scale": "linear", "off": 0.0},
    },
    "logreg": {
        "C": {"low": 1e-4, "high": 100.0, "scale": "log", "off": 1e4},
        "l1_ratio": {"low": 0.05, "high": 0.95, "scale": "linear", "off": (0.0, 1.0)},
    },
}


def sample_hyperparameters(
    trial: optuna.Trial, model_type: str
) -> Dict[str, Any]:
    """Muestrea hiperparámetros incluyendo la opción explícita de 'apagar' cada regularizador."""
    if model_type == "logreg":
        use_c_reg = trial.suggest_categorical("use_c_reg", [True, False])
        if use_c_reg:
            c_val = trial.suggest_float("C_active", 1e-4, 100.0, log=True)
        else:
            c_val = 1e4  # Penalización prácticamente apagada

        penalty_mode = trial.suggest_categorical(
            "penalty_mode", ["elasticnet", "l2_only", "l1_only"]
        )
        if penalty_mode == "l2_only":
            l1_ratio = 0.0
        elif penalty_mode == "l1_only":
            l1_ratio = 1.0
        else:
            l1_ratio = trial.suggest_float("l1_ratio_active", 0.05, 0.95)

        return {"C": c_val, "l1_ratio": l1_ratio}

    elif model_type == "lgbm":
        n_estimators = trial.suggest_int("n_estimators", 50, 1000, step=25)
        learning_rate = trial.suggest_float(
            "learning_rate", 0.005, 0.15, log=True
        )

        limit_depth = trial.suggest_categorical("limit_depth", [True, False])
        if limit_depth:
            max_depth = trial.suggest_int("max_depth_active", 3, 10)
            max_allowed_leaves = min(63, 2**max_depth)
        else:
            max_depth = -1
            max_allowed_leaves = 63

        num_leaves = trial.suggest_int("num_leaves", 7, max_allowed_leaves)

        use_min_child = trial.suggest_categorical("use_min_child", [True, False])
        min_child_samples = (
            trial.suggest_int("min_child_samples_active", 10, 150)
            if use_min_child
            else 1
        )

        use_subsample = trial.suggest_categorical("use_subsample", [True, False])
        subsample = (
            trial.suggest_float("subsample_active", 0.5, 0.99)
            if use_subsample
            else 1.0
        )

        use_colsample = trial.suggest_categorical("use_colsample", [True, False])
        colsample_bytree = (
            trial.suggest_float("colsample_bytree_active", 0.3, 0.99)
            if use_colsample
            else 1.0
        )

        use_l1 = trial.suggest_categorical("use_l1", [True, False])
        reg_alpha = (
            trial.suggest_float("reg_alpha_active", 1e-4, 20.0, log=True)
            if use_l1
            else 0.0
        )

        use_l2 = trial.suggest_categorical("use_l2", [True, False])
        reg_lambda = (
            trial.suggest_float("reg_lambda_active", 1e-4, 20.0, log=True)
            if use_l2
            else 0.0
        )

        use_split_gain = trial.suggest_categorical(
            "use_split_gain", [True, False]
        )
        min_split_gain = (
            trial.suggest_float("min_split_gain_active", 0.01, 1.0)
            if use_split_gain
            else 0.0
        )

        return {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_split_gain": min_split_gain,
        }
    else:
        raise ValueError(f"Modelo no soportado: {model_type}")


def check_parameter_boundaries(
    best_clean_params: Dict[str, Any],
    model_type: str,
    tolerance: float = 0.05,
) -> Dict[str, Any]:
    """
    Detecta si algún hiperparámetro óptimo quedó en el borde inferior/superior (5%) de la grilla,
    separándolo de aquellos que fueron deliberadamente apagados (OFF).
    """
    specs = PARAM_SPECS[model_type]
    boundary_hits = []
    off_hits = []

    for param_name, val in best_clean_params.items():
        if param_name not in specs:
            continue
        spec = specs[param_name]
        off_val = spec["off"]

        # 1. Verificar si está en estado APAGADO
        if off_val is not None:
            if isinstance(off_val, tuple) and val in off_val:
                off_hits.append(f"{param_name}=OFF({val})")
                continue
            elif val == off_val:
                off_hits.append(f"{param_name}=OFF({val})")
                continue

        # 2. Si está activo, medir distancia relativa a los bordes [low, high]
        low, high = spec["low"], spec["high"]
        # Caso especial: num_leaves acotado dinámicamente por max_depth
        if (
            param_name == "num_leaves"
            and best_clean_params.get("max_depth", -1) > 0
        ):
            high = min(high, 2 ** int(best_clean_params["max_depth"]))

        if high <= low:
            continue

        if spec["scale"] == "log":
            rel_pos = (math.log10(val) - math.log10(low)) / (
                math.log10(high) - math.log10(low)
            )
        else:
            rel_pos = (float(val) - float(low)) / (float(high) - float(low))

        if rel_pos <= tolerance:
            boundary_hits.append(f"{param_name}={val:.4g}(LOWER)")
        elif rel_pos >= (1.0 - tolerance):
            boundary_hits.append(f"{param_name}={val:.4g}(UPPER)")

    return {
        "n_params_at_boundary": len(boundary_hits),
        "boundary_flag_any": len(boundary_hits) > 0,
        "boundary_params_detail": "; ".join(boundary_hits)
        if boundary_hits
        else "NONE",
        "off_params_detail": "; ".join(off_hits) if off_hits else "NONE",
    }


def optimize_and_get_oof(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    model_type: str,
    class_weight_mode: str,
    run_id: str,
    seed: int,
    config: Dict,
) -> Tuple[Dict[str, Any], float, np.ndarray, pd.DataFrame]:
    """
    Ejecuta Optuna con persistencia SQLite (reanudable al instante si se corta).
    Devuelve:
    - best_clean_params: diccionario limpio listo para build_estimator
    - best_inner_prauc: mejor PR-AUC promedio del CV interno
    - oof_val_probs: probabilidades OOF en el CV interno para calibrar el umbral Youden
    - trials_df: historial completo de trials de este fold
    """
    n_trials = config["optimization"]["n_trials"]
    n_jobs = config["optimization"]["n_jobs_per_model"]
    inner_splits = config["validation"]["inner_cv_splits"]
    storage_url = config["paths"]["optuna_db"]

    inner_cv = StratifiedGroupKFold(
        n_splits=inner_splits, shuffle=True, random_state=seed
    )
    splits_list = list(inner_cv.split(X_train, y_train, groups_train))

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial, model_type)
        # Guardar los parámetros limpios en user_attrs del trial para recuperarlos fácil
        trial.set_user_attr("clean_params", params)

        fold_scores = []
        for tr_idx, val_idx in splits_list:
            X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

            pipe = build_estimator(
                model_type, class_weight_mode, params, y_tr, seed, n_jobs
            )
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict_proba(X_va)[:, 1]
            fold_scores.append(average_precision_score(y_va, preds))

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=run_id,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    completed_trials = len(
        [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
    )
    remaining_trials = max(0, n_trials - completed_trials)

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_clean_params = best_trial.user_attrs["clean_params"]
    best_inner_prauc = float(best_trial.value)

    # Generar predicciones OOF internas con los mejores hiperparámetros para elegir el umbral sin tocar Test
    oof_val_probs = np.zeros(len(y_train), dtype=float)
    for tr_idx, val_idx in splits_list:
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va = X_train.iloc[val_idx]
        pipe = build_estimator(
            model_type, class_weight_mode, best_clean_params, y_tr, seed, n_jobs
        )
        pipe.fit(X_tr, y_tr)
        oof_val_probs[val_idx] = pipe.predict_proba(X_va)[:, 1]

    # Exportar historial de trials
    trials_records = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            rec = {
                "run_id": run_id,
                "trial_number": t.number,
                "inner_cv_prauc": t.value,
                "duration_sec": (
                    t.datetime_complete - t.datetime_start
                ).total_seconds()
                if t.datetime_complete and t.datetime_start
                else np.nan,
            }
            rec.update(t.user_attrs.get("clean_params", {}))
            trials_records.append(rec)

    trials_df = pd.DataFrame(trials_records)
    return best_clean_params, best_inner_prauc, oof_val_probs, trials_df