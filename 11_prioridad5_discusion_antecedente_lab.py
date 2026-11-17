# 11_prioridad5_discusion_antecedente_lab.py
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.pipelines import OutlierClipper, TFQIRecalculator


def benjamini_hochberg_fdr(p_vals: np.ndarray) -> np.ndarray:
    """Ajusta un vector de p-valores por False Discovery Rate (Benjamini-Hochberg)."""
    n = len(p_vals)
    order = np.argsort(p_vals)
    ranked_p = p_vals[order]
    adj = np.empty(n, dtype=float)
    min_val = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(1.0, ranked_p[i] * n / rank)
        min_val = min(min_val, val)
        adj[i] = min_val
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def compare_lab_distributions_in_multiparas(
    df_multi: pd.DataFrame, lab_cols: List[str], out_dir: Path
) -> pd.DataFrame:
    """
    Compara cada variable de laboratorio entre multíparas con ANTC_DMG=1 y ANTC_DMG=0
    mediante Mann-Whitney U, AUC univariado y ajuste FDR.
    """
    g1 = df_multi[df_multi["ANTC_DMG"] == 1]
    g0 = df_multi[df_multi["ANTC_DMG"] == 0]

    rows = []
    for col in lab_cols:
        x1 = g1[col].dropna().to_numpy(dtype=float)
        x0 = g0[col].dropna().to_numpy(dtype=float)
        if len(x1) < 5 or len(x0) < 5:
            continue

        u_stat, p_val = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        # AUC univariado directo desde el estadístico U: P(X1 > X0)
        auc_univ = float(u_stat / (len(x1) * len(x0)))
        # Distancia al azar (0.50)
        auc_discrimination = max(auc_univ, 1.0 - auc_univ)

        rows.append(
            {
                "feature": col,
                "n_antc1": len(x1),
                "median_antc1": float(np.median(x1)),
                "iqr_antc1": float(np.percentile(x1, 75) - np.percentile(x1, 25)),
                "n_antc0": len(x0),
                "median_antc0": float(np.median(x0)),
                "iqr_antc0": float(np.percentile(x0, 75) - np.percentile(x0, 25)),
                "univariate_auc": auc_univ,
                "auc_distance_from_050": auc_discrimination,
                "p_val_raw": float(p_val),
            }
        )

    df_dist = pd.DataFrame(rows)
    df_dist["p_val_fdr"] = benjamini_hochberg_fdr(df_dist["p_val_raw"].to_numpy())
    df_dist = df_dist.sort_values(by="p_val_raw", ascending=True).reset_index(drop=True)
    df_dist.to_csv(out_dir / "prioridad5_multiparas_lab_distributions.csv", index=False)
    return df_dist


def evaluate_predicting_antecedent_from_lab(
    df_multi: pd.DataFrame,
    config: Dict,
    out_dir: Path,
    n_splits: int = 5,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Entrena LightGBM y Regresión Logística (con OutlierClipper) para predecir
    ANTC_DMG (0 vs 1) en mujeres multíparas usando únicamente Set 2a y Set 2b.
    """
    pat_col = config["schema"]["patient_id"]
    y = df_multi["ANTC_DMG"].to_numpy(dtype=int)
    groups = df_multi[pat_col].to_numpy()
    prev_antc = float(y.mean())

    sets_to_test = {
        "set_2a_lab_sin_glu": config["columns"]["feature_sets"]["set_2a_lab_sin_glu"],
        "set_2b_lab_completo": config["columns"]["feature_sets"]["set_2b_lab_completo"],
    }

    records = []
    for set_name, cols in sets_to_test.items():
        X = df_multi[cols].reset_index(drop=True)

        for model_type in ["lgbm", "logreg"]:
            rocs, praucs = [], []
            for rep in range(n_repeats):
                seed = 5000 + rep * 100
                sgkf = StratifiedGroupKFold(
                    n_splits=n_splits, shuffle=True, random_state=seed
                )
                for fold_idx, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups)):
                    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
                    X_te, y_te = X.iloc[te_idx], y[te_idx]

                    if model_type == "logreg":
                        pipe = Pipeline(
                            [
                                ("tfqi", TFQIRecalculator()),
                                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                ("clip", OutlierClipper(0.5, 99.5)),
                                ("scale", StandardScaler()),
                                (
                                    "clf",
                                    LogisticRegression(
                                        penalty="elasticnet",
                                        solver="saga",
                                        C=0.1,
                                        l1_ratio=0.5,
                                        max_iter=5000,
                                        random_state=seed + fold_idx,
                                    ),
                                ),
                            ]
                        )
                    else:
                        pipe = Pipeline(
                            [
                                ("tfqi", TFQIRecalculator()),
                                (
                                    "clf",
                                    lgb.LGBMClassifier(
                                        n_estimators=150,
                                        learning_rate=0.03,
                                        max_depth=4,
                                        num_leaves=11,
                                        min_child_samples=40,
                                        subsample=0.8,
                                        colsample_bytree=0.7,
                                        reg_alpha=1.0,
                                        reg_lambda=1.0,
                                        random_state=seed + fold_idx,
                                        n_jobs=4,
                                        verbose=-1,
                                    ),
                                ),
                            ]
                        )

                    pipe.fit(X_tr, y_tr)
                    probs = pipe.predict_proba(X_te)[:, 1]
                    rocs.append(float(roc_auc_score(y_te, probs)))
                    praucs.append(float(average_precision_score(y_te, probs)))

            records.append(
                {
                    "target": "ANTC_DMG (en Multíparas)",
                    "n_multiparas": len(df_multi),
                    "n_antc_pos": int(y.sum()),
                    "prevalence_antc": prev_antc,
                    "feature_set": set_name,
                    "model_type": model_type,
                    "roc_auc_mean": float(np.mean(rocs)),
                    "roc_auc_sd": float(np.std(rocs, ddof=1)),
                    "pr_auc_mean": float(np.mean(praucs)),
                    "pr_auc_sd": float(np.std(praucs, ddof=1)),
                    "lift_over_random": float(np.mean(praucs) / prev_antc),
                }
            )

    df_pred_antc = pd.DataFrame(records)
    df_pred_antc.to_csv(
        out_dir / "prioridad5_model_predicting_antecedent.csv", index=False
    )
    return df_pred_antc


def main() -> None:
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    proc_dir = Path(config["paths"]["processed_dir"])
    out_dir = Path(config["paths"]["outputs_dir"]) / "analysis" / "prioridad_5_discusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_parquet(proc_dir / "cohort_A_general.parquet")

    # Filtrar multíparas (PRIMIGESTA_FLAG == 0)
    df_multi = df_a[(df_a["PRIMIGESTA_FLAG"] == 0) | (df_a["ANTC_DMG"] == 1)].copy().reset_index(drop=True)
    lab_cols = [
        c
        for c in config["columns"]["feature_sets"]["set_2b_lab_completo"]
        if c != "DIAS_1T"
    ]

    df_dist = compare_lab_distributions_in_multiparas(df_multi, lab_cols, out_dir)
    df_models = evaluate_predicting_antecedent_from_lab(df_multi, config, out_dir)

    md = [
        "# Reporte Prioridad 5 (Discusión): ¿Capta el Laboratorio de Rutina la Biología del Antecedente de DMG?",
        "",
        f"- **Población analizada (Solo Multíparas, `PRIMIGESTA_FLAG == 0`):** `N = {len(df_multi)}` embarazos.",
        f"  - Multíparas con `ANTC_DMG = 1`: `{int((df_multi['ANTC_DMG']==1).sum())}` ({df_multi['ANTC_DMG'].mean()*100:.2f}%)",
        f"  - Multíparas con `ANTC_DMG = 0`: `{int((df_multi['ANTC_DMG']==0).sum())}` ({(1-df_multi['ANTC_DMG'].mean())*100:.2f}%)",
        "",
        "## 1. Experimento Directo A: ¿Puede un Modelo Predecir `ANTC_DMG = 1` a partir del Laboratorio del 1T?",
        "| Set de Variables | Modelo | Prevalencia `ANTC_DMG` | ROC-AUC (Media ± DE) | PR-AUC (Media ± DE) | Lift sobre Azar |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_models.iterrows():
        md.append(
            f"| `{r['feature_set']}` | {r['model_type'].upper()} | {r['prevalence_antc']*100:.2f}% | "
            f"**{r['roc_auc_mean']:.4f}** (±{r['roc_auc_sd']:.4f}) | "
            f"**{r['pr_auc_mean']:.4f}** (±{r['pr_auc_sd']:.4f}) | {r['lift_over_random']:.2f}x |"
        )

    md.extend(
        [
            "",
            "## 2. Experimento Directo B: Comparación Univariada de Biomarcadores en Multíparas (Top 12 por p-valor)",
            "| Biomarcador | Mediana (IQR) `ANTC=1` | Mediana (IQR) `ANTC=0` | AUC Univariado | p-valor Crudo | p-valor FDR (BH) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in df_dist.head(12).iterrows():
        md.append(
            f"| `{r['feature']}` | {r['median_antc1']:.3f} ({r['iqr_antc1']:.3f}) | "
            f"{r['median_antc0']:.3f} ({r['iqr_antc0']:.3f}) | {r['univariate_auc']:.3f} | "
            f"{r['p_val_raw']:.4f} | **{r['p_val_fdr']:.4f}** |"
        )

    report_path = out_dir / "reporte_prioridad5_discusion.md"
    full_md = "\n".join(md)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(full_md)
    print("=" * 80)
    print(f"✔ Resultados de Prioridad 5 guardados en: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()