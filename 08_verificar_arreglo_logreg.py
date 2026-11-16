# 08_verificar_arreglo_logreg.py
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from src.pipelines import TFQIRecalculator


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Acota (Winsoriza) cada columna continua entre los percentiles [lower_q, upper_q]
    calculados EXCLUSIVAMENTE en el conjunto de entrenamiento de cada fold (sin data leakage),
    evitando que divisiones por valores cercanos a cero (ej. SPINA_GT, NLR, LMR) disparen
    la constante de Lipschitz del solver SAGA.
    """

    def __init__(self, lower_q: float = 0.5, upper_q: float = 99.5) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: np.ndarray, y=None) -> "OutlierClipper":
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanpercentile(X_arr, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanpercentile(X_arr, self.upper_q, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


def evaluate_pipe(pipe: Pipeline, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, y_te: np.ndarray, label: str) -> None:
    pipe.fit(X_tr, y_tr)
    clf = pipe.named_steps["model"]
    probs = pipe.predict_proba(X_te)[:, 1]

    # Verificar máximo valor escalado que entra al modelo lineal
    X_Pre = X_tr.copy()
    for name, step in pipe.steps[:-1]:
        X_Pre = step.transform(X_Pre)
    max_scaled = float(np.abs(X_Pre).max())

    print(f"\n>>> {label}")
    print(f"    Iteraciones SAGA : {clf.n_iter_[0]} / {clf.max_iter} | Max valor post-escalado: {max_scaled:.2f}")
    print(f"    Intercepto       : {clf.intercept_[0]:.4f} | Prob mediana: {np.median(probs):.4f} (min={probs.min():.4f}, max={probs.max():.4f})")
    print(
        f"    Métricas Test    -> PR-AUC: {average_precision_score(y_te, probs):.4f} | "
        f"ROC-AUC: {roc_auc_score(y_te, probs):.4f} | "
        f"Brier: {brier_score_loss(y_te, probs):.4f} | "
        f"Log-Loss: {log_loss(y_te, probs):.4f}"
    )


def main() -> None:
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    proc_dir = Path(config["paths"]["processed_dir"])
    df = pd.read_parquet(proc_dir / "cohort_A_general.parquet")
    splits = pd.read_parquet(proc_dir / "splits_cv_5x5.parquet")

    preg_col = config["schema"]["pregnancy_id"]
    target_col = config["schema"]["target_col"]
    high_null = config["columns"]["high_null_drop_in_logreg"]

    mask_split = (splits["cohort"] == "A_all") & (splits["repeat_idx"] == 0) & (splits["fold_idx"] == 0)
    test_ids = set(splits.loc[mask_split, preg_col])

    df_train = df[~df[preg_col].isin(test_ids)].reset_index(drop=True)
    df_test = df[df[preg_col].isin(test_ids)].reset_index(drop=True)
    y_tr = df_train[target_col].values.astype(int)
    y_te = df_test[target_col].values.astype(int)

    cols_set1 = [c for c in config["columns"]["feature_sets"]["set_1_clinico"] if c not in high_null]
    cols_set3 = [c for c in config["columns"]["feature_sets"]["set_3_full"] if c not in high_null]

    # 1. Pipeline viejo (solo RobustScaler)
    pipe_old = Pipeline([
        ("tfqi_recalc", TFQIRecalculator()),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=2500, random_state=42)),
    ])

    # 2. Pipeline corregido (OutlierClipper in-fold [0.5%, 99.5%] + StandardScaler)
    pipe_fixed = Pipeline([
        ("tfqi_recalc", TFQIRecalculator()),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("clipper", OutlierClipper(lower_q=0.5, upper_q=99.5)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)),
    ])

    print("=" * 85)
    print(" COMPARACIÓN EN VIVO (COHORTE A - FOLD 0): PIPELINE VIEJO VS. PIPELINE CORREGIDO")
    print("=" * 85)

    evaluate_pipe(pipe_old, df_train[cols_set1], y_tr, df_test[cols_set1], y_te, "1. SET 1 CLÍNICO — Pipeline Viejo")
    evaluate_pipe(pipe_old, df_train[cols_set3], y_tr, df_test[cols_set3], y_te, "2. SET 3 FULL    — Pipeline Viejo (COLAPSADO)")
    evaluate_pipe(pipe_fixed, df_train[cols_set1], y_tr, df_test[cols_set1], y_te, "3. SET 1 CLÍNICO — Pipeline Corregido (Clipper + StandardScaler)")
    evaluate_pipe(pipe_fixed, df_train[cols_set3], y_tr, df_test[cols_set3], y_te, "4. SET 3 FULL    — Pipeline Corregido (Clipper + StandardScaler)")
    print("=" * 85)


if __name__ == "__main__":
    main()