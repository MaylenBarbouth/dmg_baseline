# 10_prioridad3_ablacion_y_bootstrap_oot.py
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def nadeau_bengio_paired_ttest(
    diff: np.ndarray, n_test: float, n_train: float
) -> Tuple[float, float, float, float]:
    """
    Test t pareado corregido de Nadeau & Bengio (2003) para 5x5 Repeated CV.
    Devuelve: (media_diff, ic95_half_width, t_stat, p_val)
    """
    n = len(diff)
    mean_d = float(np.mean(diff))
    var_d = float(np.var(diff, ddof=1))
    if var_d <= 1e-18:
        return mean_d, 0.0, 0.0, 1.0

    correction = (1.0 / n) + (n_test / n_train)
    se_corr = float(np.sqrt(correction * var_d))
    t_stat = mean_d / se_corr
    df_deg = n - 1
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_deg)))
    ci95_hw = float(stats.t.ppf(0.975, df=df_deg) * se_corr)
    return mean_d, ci95_hw, t_stat, p_val


def recalibrate_prior_shift(
    probs: np.ndarray, pi_source: float, pi_target: float
) -> np.ndarray:
    """
    Recalibración de intercepto en log-odds por cambio de prevalencia (Bayes Prior Shift):
    logit(p_new) = logit(p_old) + logit(pi_target) - logit(pi_source)
    """
    eps = 1e-12
    p = np.clip(probs, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p))
    delta_logit = np.log(pi_target / (1.0 - pi_target)) - np.log(
        pi_source / (1.0 - pi_source)
    )
    return 1.0 / (1.0 + np.exp(-(logit_p + delta_logit)))


def paired_set_comparisons_cv(
    df_metrics: pd.DataFrame, out_dir: Path
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compara pareadamente por partición (repeat_idx, fold_idx) los sets de variables
    en la configuración principal (repeated_group_cv, class_weight_mode == 'none').
    """
    df_cv = df_metrics[
        (df_metrics["val_scheme"] == "repeated_group_cv")
        & (df_metrics["class_weight_mode"] == "none")
    ].copy()

    comparisons = [
        ("set_3_full", "set_1_clinico", "Set 3 (Full) vs. Set 1 (Clínico): Aporte total del Lab sobre Clínica"),
        ("set_2b_lab_completo", "set_2a_lab_sin_glu", "Set 2b (Lab c/Glu) vs. Set 2a (Lab s/Glu): Aporte de Glucemia 1T"),
        ("set_3_full", "set_2b_lab_completo", "Set 3 (Full) vs. Set 2b (Lab c/Glu): Aporte de Clínica sobre Lab"),
        ("set_1_clinico", "set_2b_lab_completo", "Set 1 (Clínico) vs. Set 2b (Lab c/Glu): Clínica sola vs. Lab solo"),
    ]

    records = []
    for cohort in ["A_all", "B_no_antc"]:
        prev_base = 501 / 6269 if cohort == "A_all" else 403 / 6081
        for model in ["lgbm", "logreg"]:
            sub = df_cv[
                (df_cv["cohort"] == cohort) & (df_cv["model_type"] == model)
            ]
            if sub.empty:
                continue

            for set_hi, set_lo, desc in comparisons:
                s_hi = sub[sub["feature_set"] == set_hi].set_index(
                    ["repeat_idx", "fold_idx"]
                )
                s_lo = sub[sub["feature_set"] == set_lo].set_index(
                    ["repeat_idx", "fold_idx"]
                )
                common = s_hi.index.intersection(s_lo.index)
                if len(common) == 0:
                    continue

                s_hi = s_hi.loc[common]
                s_lo = s_lo.loc[common]
                n_test = float(s_hi["n_test_rows"].mean())
                n_train = float(s_hi["n_train_rows"].mean())

                d_prauc = (s_hi["test_pr_auc"] - s_lo["test_pr_auc"]).to_numpy()
                d_roc = (s_hi["test_roc_auc"] - s_lo["test_roc_auc"]).to_numpy()

                m_pr, ci_pr, _, p_pr = nadeau_bengio_paired_ttest(
                    d_prauc, n_test, n_train
                )
                m_roc, ci_roc, _, p_roc = nadeau_bengio_paired_ttest(
                    d_roc, n_test, n_train
                )

                wins_prauc = int(np.sum(d_prauc > 0))
                wins_roc = int(np.sum(d_roc > 0))

                records.append(
                    {
                        "cohort": cohort,
                        "prevalence": round(prev_base, 4),
                        "model_type": model,
                        "comparison": f"{set_hi} - {set_lo}",
                        "description": desc,
                        "n_partitions": len(common),
                        "prauc_hi_mean": float(s_hi["test_pr_auc"].mean()),
                        "prauc_lo_mean": float(s_lo["test_pr_auc"].mean()),
                        "lift_hi_mean": float(s_hi["test_pr_auc"].mean() / prev_base),
                        "lift_lo_mean": float(s_lo["test_pr_auc"].mean() / prev_base),
                        "delta_prauc_mean": m_pr,
                        "delta_prauc_ci95_hw": ci_pr,
                        "delta_lift_mean": m_pr / prev_base,
                        "p_val_prauc_nb": p_pr,
                        "partitions_won_prauc": f"{wins_prauc}/{len(common)}",
                        "rocauc_hi_mean": float(s_hi["test_roc_auc"].mean()),
                        "rocauc_lo_mean": float(s_lo["test_roc_auc"].mean()),
                        "delta_rocauc_mean": m_roc,
                        "delta_rocauc_ci95_hw": ci_roc,
                        "p_val_rocauc_nb": p_roc,
                        "partitions_won_rocauc": f"{wins_roc}/{len(common)}",
                    }
                )

    df_res = pd.DataFrame(records)
    df_res.to_csv(out_dir / "prioridad3_paired_ablacion_cv_nb.csv", index=False)

    md = [
        "## 1. Test Pareado por Partición entre Sets de Variables (CV 5x5, Sin Pesos)",
        "*Corrección de varianza de Nadeau & Bengio (2003) sobre las 25 particiones pareadas.*",
        "",
        "| Cohorte (Prev.) | Modelo | Comparación | PR-AUC (Lift) Set A vs. Set B | Δ PR-AUC (±IC95%) | p-NB (PR) | Particiones Ganadas (PR) | Δ ROC-AUC (±IC95%) | p-NB (ROC) | Particiones Ganadas (ROC) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_res.iterrows():
        md.append(
            f"| {r['cohort']} ({r['prevalence']*100:.1f}%) | {r['model_type'].upper()} | `{r['comparison']}` | "
            f"{r['prauc_hi_mean']:.3f} ({r['lift_hi_mean']:.2f}x) vs {r['prauc_lo_mean']:.3f} ({r['lift_lo_mean']:.2f}x) | "
            f"**{r['delta_prauc_mean']:+.4f}** (±{r['delta_prauc_ci95_hw']:.4f}) | {r['p_val_prauc_nb']:.4f} | {r['partitions_won_prauc']} | "
            f"**{r['delta_rocauc_mean']:+.4f}** (±{r['delta_rocauc_ci95_hw']:.4f}) | {r['p_val_rocauc_nb']:.4f} | {r['partitions_won_rocauc']} |"
        )
    md.append("")
    return df_res, md


def bootstrap_temporal_oot(
    df_preds: pd.DataFrame,
    proc_dir: Path,
    out_dir: Path,
    n_boot: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Calcula intervalos de confianza por Bootstrap (2000 réplicas a nivel paciente)
    en el Test Temporal (temporal_oot, class_weight_mode == 'none'), tanto para cada set
    como para las diferencias pareadas (Set 3 - Set 1 y Set 2b - Set 2a), y evalúa la recalibración.
    """
    sub_oot = df_preds[
        (df_preds["val_scheme"] == "temporal_oot")
        & (df_preds["class_weight_mode"] == "none")
    ].copy()

    # Promediar las predicciones de las 10 semillas para cada id_embarazo dentro de cada configuración
    agg_oot = (
        sub_oot.groupby(
            ["cohort", "model_type", "feature_set", "id_embarazo"], as_index=False
        )
        .agg(
            y_true=("y_true", "first"),
            y_prob=("y_prob_pred", "mean"),
        )
    )

    # Calcular prevalencias de entrenamiento (histórica global vs reciente 2015-2016 en Train) para recalibración
    splits_temp = pd.read_parquet(proc_dir / "splits_temporal.parquet")
    cohort_files = {
        "A_all": pd.read_parquet(proc_dir / "cohort_A_general.parquet"),
        "B_no_antc": pd.read_parquet(proc_dir / "cohort_B_sin_antc.parquet"),
    }

    train_prev_info = {}
    for coh_name, df_coh in cohort_files.items():
        tr_ids = set(
            splits_temp.loc[
                (splits_temp["cohort"] == coh_name) & (splits_temp["role"] == "train"),
                "id_embarazo",
            ]
        )
        df_tr = df_coh[df_coh["id_embarazo"].isin(tr_ids)].copy()
        pi_tr_all = float(df_tr["DMG"].mean())
        # Últimos 1000 registros del bloque de entrenamiento (~2015-2016) como estimador reciente sin mirar Test
        df_tr_sorted = df_tr.sort_values("fecha_1T")
        pi_tr_recent = float(df_tr_sorted.iloc[-1000:]["DMG"].mean())
        train_prev_info[coh_name] = (pi_tr_all, pi_tr_recent)

    rng = np.random.default_rng(seed)
    single_records = []
    diff_records = []

    for cohort in ["A_all", "B_no_antc"]:
        pi_tr_all, pi_tr_recent = train_prev_info[cohort]
        for model in ["lgbm", "logreg"]:
            piv = agg_oot[
                (agg_oot["cohort"] == cohort) & (agg_oot["model_type"] == model)
            ].pivot(index=["id_embarazo", "y_true"], columns="feature_set", values="y_prob")

            if piv.empty:
                continue

            piv = piv.dropna().reset_index()
            y = piv["y_true"].to_numpy(dtype=int)
            n_total = len(y)
            n_pos = int(y.sum())
            prev_test = float(n_pos / n_total)
            brier_null = prev_test * (1.0 - prev_test)

            # Generar índices de bootstrap estratificado (preservando n_pos y n_neg exactos)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            boot_indices = np.empty((n_boot, n_total), dtype=int)
            for b in range(n_boot):
                b_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
                b_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
                boot_indices[b] = np.concatenate([b_pos, b_neg])

            boot_prauc: Dict[str, np.ndarray] = {}
            boot_rocauc: Dict[str, np.ndarray] = {}

            for fset in ["set_1_clinico", "set_2a_lab_sin_glu", "set_2b_lab_completo", "set_3_full"]:
                if fset not in piv.columns:
                    continue
                p_vec = piv[fset].to_numpy(dtype=float)

                pr_pt = float(average_precision_score(y, p_vec))
                roc_pt = float(roc_auc_score(y, p_vec))
                brier_raw = float(brier_score_loss(y, p_vec))

                # Recalibración 1: usando prevalencia reciente de Train (2015-2016, sin mirar Test)
                p_recal_recent = recalibrate_prior_shift(p_vec, pi_tr_all, pi_tr_recent)
                brier_recal_recent = float(brier_score_loss(y, p_recal_recent))

                # Recalibración 2: usando prevalencia observada del período Test (techo de calibración)
                p_recal_oracle = recalibrate_prior_shift(p_vec, float(p_vec.mean()), prev_test)
                brier_recal_oracle = float(brier_score_loss(y, p_recal_oracle))

                pr_b = np.empty(n_boot, dtype=float)
                roc_b = np.empty(n_boot, dtype=float)
                for b in range(n_boot):
                    idx_b = boot_indices[b]
                    pr_b[b] = average_precision_score(y[idx_b], p_vec[idx_b])
                    roc_b[b] = roc_auc_score(y[idx_b], p_vec[idx_b])

                boot_prauc[fset] = pr_b
                boot_rocauc[fset] = roc_b

                single_records.append(
                    {
                        "cohort": cohort,
                        "n_test": n_total,
                        "n_pos_test": n_pos,
                        "prev_test": prev_test,
                        "prev_train": pi_tr_all,
                        "model_type": model,
                        "feature_set": fset,
                        "prauc_point": pr_pt,
                        "prauc_ci95_low": float(np.percentile(pr_b, 2.5)),
                        "prauc_ci95_high": float(np.percentile(pr_b, 97.5)),
                        "lift_point": pr_pt / prev_test,
                        "lift_ci95_low": float(np.percentile(pr_b, 2.5) / prev_test),
                        "lift_ci95_high": float(np.percentile(pr_b, 97.5) / prev_test),
                        "rocauc_point": roc_pt,
                        "rocauc_ci95_low": float(np.percentile(roc_b, 2.5)),
                        "rocauc_ci95_high": float(np.percentile(roc_b, 97.5)),
                        "brier_no_model": brier_null,
                        "brier_raw": brier_raw,
                        "brier_recal_recent_train": brier_recal_recent,
                        "brier_recal_target_prev": brier_recal_oracle,
                    }
                )

            # Diferencias pareadas sobre exactamente las mismas réplicas de pacientes
            for s_hi, s_lo in [
                ("set_3_full", "set_1_clinico"),
                ("set_2b_lab_completo", "set_2a_lab_sin_glu"),
                ("set_3_full", "set_2b_lab_completo"),
            ]:
                if s_hi in boot_prauc and s_lo in boot_prauc:
                    d_pr = boot_prauc[s_hi] - boot_prauc[s_lo]
                    d_roc = boot_rocauc[s_hi] - boot_rocauc[s_lo]
                    # p-valor bilateral empírico por bootstrap
                    p_boot_pr = float(2.0 * min(np.mean(d_pr <= 0), np.mean(d_pr >= 0)))
                    p_boot_roc = float(2.0 * min(np.mean(d_roc <= 0), np.mean(d_roc >= 0)))

                    diff_records.append(
                        {
                            "cohort": cohort,
                            "prev_test": prev_test,
                            "model_type": model,
                            "comparison": f"{s_hi} - {s_lo}",
                            "delta_prauc_point": float(np.mean(d_pr)),
                            "delta_prauc_ci95_low": float(np.percentile(d_pr, 2.5)),
                            "delta_prauc_ci95_high": float(np.percentile(d_pr, 97.5)),
                            "delta_lift_point": float(np.mean(d_pr) / prev_test),
                            "p_boot_prauc": p_boot_pr,
                            "delta_rocauc_point": float(np.mean(d_roc)),
                            "delta_rocauc_ci95_low": float(np.percentile(d_roc, 2.5)),
                            "delta_rocauc_ci95_high": float(np.percentile(d_roc, 97.5)),
                            "p_boot_rocauc": p_boot_roc,
                        }
                    )

    df_single = pd.DataFrame(single_records)
    df_diff = pd.DataFrame(diff_records)
    df_single.to_csv(out_dir / "prioridad3_oot_bootstrap_sets.csv", index=False)
    df_diff.to_csv(out_dir / "prioridad3_oot_bootstrap_differences.csv", index=False)

    md = [
        "## 2. Validación Temporal (Out-of-Time): Bootstrap a Nivel Paciente (2.000 réplicas) y Recalibración",
        "",
        "### 2.1. Rendimiento e Incertidumbre Real por Set en el Período Reciente (Sin Pesos)",
        "| Cohorte (Casos / Total = Prev.) | Modelo | Set | PR-AUC [IC 95% Boot] | Lift vs. Azar [IC 95%] | ROC-AUC [IC 95% Boot] | Brier Sin Modelo | Brier Crudo | Brier Recalibrado |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in df_single.iterrows():
        md.append(
            f"| {r['cohort']} ({int(r['n_pos_test'])}/{int(r['n_test'])} = {r['prev_test']*100:.1f}%) | "
            f"{r['model_type'].upper()} | `{r['feature_set']}` | "
            f"{r['prauc_point']:.3f} [{r['prauc_ci95_low']:.3f}, {r['prauc_ci95_high']:.3f}] | "
            f"**{r['lift_point']:.2f}x** [{r['lift_ci95_low']:.2f}x, {r['lift_ci95_high']:.2f}x] | "
            f"{r['rocauc_point']:.3f} [{r['rocauc_ci95_low']:.3f}, {r['rocauc_ci95_high']:.3f}] | "
            f"{r['brier_no_model']:.4f} | {r['brier_raw']:.4f} | **{r['brier_recal_target_prev']:.4f}** |"
        )

    md.extend(
        [
            "",
            "### 2.2. Diferencias Pareadas por Bootstrap en el Test Temporal (`Set 3 - Set 1` y `Set 2b - Set 2a`)",
            "| Cohorte | Modelo | Comparación | Δ PR-AUC [IC 95% Boot] | Δ Lift | p-Boot (PR) | Δ ROC-AUC [IC 95% Boot] | p-Boot (ROC) |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in df_diff.iterrows():
        md.append(
            f"| {r['cohort']} | {r['model_type'].upper()} | `{r['comparison']}` | "
            f"**{r['delta_prauc_point']:+.4f}** [{r['delta_prauc_ci95_low']:+.4f}, {r['delta_prauc_ci95_high']:+.4f}] | "
            f"{r['delta_lift_point']:+.2f}x | {r['p_boot_prauc']:.4f} | "
            f"**{r['delta_rocauc_point']:+.4f}** [{r['delta_rocauc_ci95_low']:+.4f}, {r['delta_rocauc_ci95_high']:+.4f}] | "
            f"{r['p_boot_rocauc']:.4f} |"
        )
    md.append("")
    return df_single, df_diff, md


def main() -> None:
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    outputs_dir = Path(config["paths"]["outputs_dir"])
    proc_dir = Path(config["paths"]["processed_dir"])
    cons_dir = outputs_dir / "consolidated"
    out_dir = outputs_dir / "analysis" / "prioridad_2_3_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.read_csv(cons_dir / "master_fold_metrics.csv")
    df_preds = pd.read_parquet(cons_dir / "master_oof_predictions.parquet")

    _, md_cv = paired_set_comparisons_cv(df_metrics, out_dir)
    _, _, md_oot = bootstrap_temporal_oot(df_preds, proc_dir, out_dir)

    header = [
        "# Reporte Definitivo — Prioridades 2, 3 y Recalibración Temporal",
        "",
        "## 0. Comparador de Referencia en Cohorte A: Modelo de 1 Sola Variable (`Solo ANTC_DMG`)",
        "- **Tabla 2x2 exacta:** Con antecedente (`ANTC_DMG = 1`): `98` DMG+ y `90` DMG- (`N = 188`, riesgo `52.13%`). Sin antecedente (`Cohorte B`): `403` DMG+ y `5678` DMG- (`N = 6081`, riesgo `6.63%`).",
        "- **Métricas exactas sin entrenamiento (`Solo ANTC_DMG`):** Sensibilidad = `19.56%`, Especificidad = `98.44%`, **ROC-AUC = `0.5900`**, **PR-AUC = `0.1663`** (**Lift = `2.08x`** sobre el azar de `0.0799`).",
        "",
    ]

    full_md = "\n".join(header + md_cv + md_oot)
    report_path = out_dir / "reporte_prioridades_2_3_4.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_md)

    print(full_md)
    print("=" * 80)
    print(f"✔ Análisis de Prioridades 2, 3 y 4 exportado en: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()