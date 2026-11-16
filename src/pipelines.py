# src/pipelines.py
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class TFQIRecalculator(BaseEstimator, TransformerMixin):
    """
    Recalcula dinámicamente la columna TFQI dentro de cada fold ajustando las ECDFs
    de TSH1_TIR2 y T4L1_TIR2 exclusivamente sobre el conjunto de entrenamiento.
    """

    def __init__(
        self,
        tsh_col: str = "TSH1_TIR2",
        t4l_col: str = "T4L1_TIR2",
        tfqi_col: str = "TFQI",
    ) -> None:
        self.tsh_col = tsh_col
        self.t4l_col = t4l_col
        self.tfqi_col = tfqi_col

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TFQIRecalculator":
        if (
            isinstance(X, pd.DataFrame)
            and self.tfqi_col in X.columns
            and self.tsh_col in X.columns
            and self.t4l_col in X.columns
        ):
            self.active_ = True
            tsh_vals = X[self.tsh_col].dropna().to_numpy(dtype=float)
            t4l_vals = X[self.t4l_col].dropna().to_numpy(dtype=float)
            self.tsh_sorted_ = np.sort(tsh_vals)
            self.t4l_sorted_ = np.sort(t4l_vals)
        else:
            self.active_ = False
        return self

    @staticmethod
    def _empirical_cdf(
        ref_sorted: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        n = len(ref_sorted)
        if n == 0:
            return np.full_like(values, np.nan, dtype=float)

        out = np.full_like(values, np.nan, dtype=float)
        valid_mask = ~np.isnan(values)
        v = values[valid_mask]

        left = np.searchsorted(ref_sorted, v, side="left")
        right = np.searchsorted(ref_sorted, v, side="right")
        out[valid_mask] = (left + right) / (2.0 * n)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "active_", False):
            return X

        X_out = X.copy()
        f_tsh = self._empirical_cdf(
            self.tsh_sorted_, X_out[self.tsh_col].to_numpy(dtype=float)
        )
        f_t4l = self._empirical_cdf(
            self.t4l_sorted_, X_out[self.t4l_col].to_numpy(dtype=float)
        )
        X_out[self.tfqi_col] = f_tsh + f_t4l - 1.0
        return X_out


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Winsoriza cada variable entre los percentiles [lower_q, upper_q] estimados
    exclusivamente en el conjunto de entrenamiento del fold. Evita que cocientes
    con denominadores cercanos a cero (ej. SPINA_GT) disparen la constante de
    Lipschitz del solver SAGA.
    """

    def __init__(self, lower_q: float = 0.5, upper_q: float = 99.5) -> None:
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X: np.ndarray, y: Any = None) -> "OutlierClipper":
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanpercentile(X_arr, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanpercentile(X_arr, self.upper_q, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)


def get_active_features(
    cohort_name: str,
    feature_set_name: str,
    model_type: str,
    config: Dict,
) -> List[str]:
    """
    Devuelve la lista exacta de columnas para la combinación solicitada:
    - En Cohorte B (sin antecedente de DMG), quita ANTC_DMG porque es constante (0).
    - En Regresión Logística, quita IMC y TALLA (>50% nulos).
    """
    cols = list(config["columns"]["feature_sets"][feature_set_name])

    if cohort_name == "B_no_antc" and "ANTC_DMG" in cols:
        cols.remove("ANTC_DMG")

    if model_type == "logreg":
        high_null_cols = config["columns"]["high_null_drop_in_logreg"]
        cols = [c for c in cols if c not in high_null_cols]

    return cols


def build_estimator(
    model_type: str,
    class_weight_mode: str,
    params: Dict[str, Any],
    y_train: np.ndarray,
    seed: int,
    n_jobs: int,
) -> Pipeline:
    """
    Construye el Pipeline de entrenamiento:
    - Paso 0 (Ambos modelos): TFQIRecalculator in-fold.
    - LogReg: SimpleImputer(median, add_indicator=True) + OutlierClipper(0.5, 99.5) + StandardScaler + LogisticRegression(saga, elasticnet).
    - LightGBM: Manejo nativo de nulos directo en LGBMClassifier.
    """
    if model_type == "logreg":
        cw = "balanced" if class_weight_mode == "balanced" else None
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            class_weight=cw,
            max_iter=5000,
            tol=1e-3,
            random_state=seed,
            n_jobs=1,
        )
        pipe = Pipeline(
            steps=[
                ("tfqi_recalc", TFQIRecalculator()),
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True),
                ),
                ("clipper", OutlierClipper(lower_q=0.5, upper_q=99.5)),
                ("scaler", StandardScaler()),
                ("model", clf),
            ]
        )
        return pipe

    elif model_type == "lgbm":
        if class_weight_mode == "balanced":
            n_pos = max(int(np.sum(y_train == 1)), 1)
            n_neg = int(np.sum(y_train == 0))
            scale_pos_weight = float(n_neg / n_pos)
        else:
            scale_pos_weight = 1.0

        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            subsample_freq=1 if float(params["subsample"]) < 1.0 else 0,
            colsample_bytree=float(params["colsample_bytree"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            min_split_gain=float(params["min_split_gain"]),
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
            importance_type="gain",
        )
        return Pipeline(
            steps=[
                ("tfqi_recalc", TFQIRecalculator()),
                ("model", clf),
            ]
        )

    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")


def extract_feature_importances(
    fitted_pipeline: Pipeline,
    feature_names: List[str],
    model_type: str,
    run_id: str,
) -> pd.DataFrame:
    """Extrae coeficientes/Odds Ratios en LogReg o Gain/Split en LightGBM."""
    model = fitted_pipeline.named_steps["model"]

    if model_type == "logreg":
        imputer = fitted_pipeline.named_steps["imputer"]
        out_names = list(imputer.get_feature_names_out(feature_names))
        coefs = model.coef_.ravel()
        return pd.DataFrame(
            {
                "run_id": run_id,
                "model_type": "logreg",
                "feature_name": out_names,
                "coef_standardized": coefs,
                "odds_ratio_std": np.exp(np.clip(coefs, -20, 20)),
                "importance_gain": np.abs(coefs),
                "importance_split": np.nan,
            }
        )
    else:
        booster = model.booster_
        gain_imp = booster.feature_importance(importance_type="gain")
        split_imp = booster.feature_importance(importance_type="split")
        return pd.DataFrame(
            {
                "run_id": run_id,
                "model_type": "lgbm",
                "feature_name": feature_names,
                "coef_standardized": np.nan,
                "odds_ratio_std": np.nan,
                "importance_gain": gain_imp,
                "importance_split": split_imp,
            }
        )