# 04_analyze_stage0_and_stage1.py
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats


def nadeau_bengio_paired_ttest(
    diff: np.ndarray, n_test: float, n_train: float
) -> Tuple[float, float, float, float]:
    """
    Calcula el t-test pareado corregido de Nadeau & Bengio (2003) para
    Repeated K-Fold Cross-Validation.
    Devuelve: (media_diff, ic95_half_width, t_stat, p_val)
    """
    n = len(diff)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan

    mean_d = float(np.mean(diff))
    var_d = float(np.var(diff, ddof=1))

    if var_d <= 1e-18:
        return mean_d, 0.0, 0.0, 1.0

    # Factor de corrección por solapamiento entre conjuntos de entrenamiento
    correction_factor = (1.0 / n) + (n_test / n_train)
    se_corrected = float(np.sqrt(correction_factor * var_d))

    t_stat = mean_d / se_corrected
    df = n - 1
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))
    t_crit = float(stats.t.ppf(0.975, df=df))
    ci95_hw = t_crit * se_corrected

    return mean_d, ci95_hw, t_stat, p_val


def parse_detail_column(series: pd.Series) -> Counter:
    """Desglosa las cadenas 'param=val(BOUND); ...' o 'param=OFF(val)' y cuenta frecuencias."""
    counter: Counter = Counter()
    for val in series.dropna():
        val_str = str(val).strip()
        if not val_str or val_str == "NONE":
            continue
        parts = [p.strip() for p in val_str.split(";") if p.strip()]
        for part in parts:
            if "(" in part and ")" in part:
                param_name = part.split("=")[0].strip()
                tag = part.split("(")[-1].replace(")", "").strip()
                if "OFF" in part:
                    counter[f"{param_name} -> OFF"] += 1
                else:
                    counter[f"{param_name} -> {tag}"] += 1
            else:
                counter[part] += 1
    return counter


def audit_stage0(
    df: pd.DataFrame, out_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Ejecuta la Etapa 0: Integridad, Overfitting Gap y Auditoría de Bordes/OFF."""
    md_lines = []
    md_lines.append("# Reporte de Análisis — Etapas 0 y 1 (Baseline DMG)")
    md_lines.append("")
    md_lines.append("## Etapa 0: Control de Calidad Técnico y Auditoría de Grilla")

    # 0.1 Integridad general
    total_runs = len(df)
    cv_runs = int((df["val_scheme"] == "repeated_group_cv").sum())
    oot_runs = int((df["val_scheme"] == "temporal_oot").sum())
    total_hours = df["elapsed_sec"].sum() / 3600.0
    nan_metrics = int(
        df[["test_pr_auc", "test_roc_auc", "test_brier_score"]].isna().sum().sum()
    )

    md_lines.append("### 0.1. Integridad de Ejecución")
    md_lines.append(f"- **Total de corridas consolidadas:** {total_runs} (Esperadas: 1120)")
    md_lines.append(f"  - Validación Cruzada 5x5 (`repeated_group_cv`): {cv_runs} (Esperadas: 800)")
    md_lines.append(f"  - Auditoría Temporal (`temporal_oot`): {oot_runs} (Esperadas: 320)")
    md_lines.append(f"- **Valores nulos en métricas principales:** {nan_metrics}")
    md_lines.append(f"- **Tiempo total de cómputo acumulado:** {total_hours:.2f} horas")
    md_lines.append("")

    # 0.2 Brecha de sobreajuste y frecuencia de bordes por Modelo y Esquema
    df = df.copy()
    df["overfit_gap_prauc"] = df["best_inner_cv_prauc"] - df["test_pr_auc"]

    audit_grp = (
        df.groupby(["val_scheme", "cohort", "model_type", "class_weight_mode"])
        .agg(
            n_runs=("run_id", "count"),
            inner_prauc_mean=("best_inner_cv_prauc", "mean"),
            test_prauc_mean=("test_pr_auc", "mean"),
            overfit_gap_mean=("overfit_gap_prauc", "mean"),
            overfit_gap_sd=("overfit_gap_prauc", "std"),
            pct_runs_with_boundary=("boundary_flag_any", lambda x: float(np.mean(x) * 100.0)),
            mean_n_params_boundary=("n_params_at_boundary", "mean"),
        )
        .reset_index()
    )
    audit_grp.to_csv(out_dir / "stage0_overfit_and_boundary_summary.csv", index=False)

    md_lines.append("### 0.2. Brecha de Sobreajuste (Inner CV vs. Test) y Frecuencia de Bordes")
    md_lines.append(
        "| Esquema | Cohorte | Modelo | Peso | Runs | Inner PR-AUC | Test PR-AUC | Gap (Inner - Test) | % Corridas en Borde |"
    )
    md_lines.append("| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, r in audit_grp.iterrows():
        md_lines.append(
            f"| {r['val_scheme']} | {r['cohort']} | {r['model_type'].upper()} | {r['class_weight_mode']} | "
            f"{int(r['n_runs'])} | {r['inner_prauc_mean']:.4f} | {r['test_prauc_mean']:.4f} | "
            f"{r['overfit_gap_mean']:+.4f} (±{r['overfit_gap_sd']:.4f}) | {r['pct_runs_with_boundary']:.1f}% |"
        )
    md_lines.append("")

    # 0.3 Desglose de qué hiperparámetros tocaron bordes o se apagaron
    param_records = []
    for model in ["logreg", "lgbm"]:
        sub = df[df["model_type"] == model]
        n_m = len(sub)
        if n_m == 0:
            continue
        b_counts = parse_detail_column(sub["boundary_params_detail"])
        o_counts = parse_detail_column(sub["off_params_detail"])

        for k, v in b_counts.most_common():
            param_records.append(
                {
                    "model_type": model,
                    "event_type": "BOUNDARY_HIT",
                    "detail": k,
                    "count": v,
                    "pct_of_model_runs": round(v / n_m * 100.0, 2),
                }
            )
        for k, v in o_counts.most_common():
            param_records.append(
                {
                    "model_type": model,
                    "event_type": "TURNED_OFF",
                    "detail": k,
                    "count": v,
                    "pct_of_model_runs": round(v / n_m * 100.0, 2),
                }
            )

    df_param_audit = pd.DataFrame(param_records)
    df_param_audit.to_csv(out_dir / "stage0_hyperparam_audit.csv", index=False)

    md_lines.append("### 0.3. Top Hiperparámetros en Borde de Grilla y Estados Apagados (OFF)")
    md_lines.append("| Modelo | Tipo de Evento | Hiperparámetro y Dirección | Conteo | % sobre corridas del modelo |")
    md_lines.append("| :--- | :--- | :--- | :---: | :---: |")
    for _, r in df_param_audit.head(20).iterrows():
        md_lines.append(
            f"| {r['model_type'].upper()} | {r['event_type']} | `{r['detail']}` | {int(r['count'])} | {r['pct_of_model_runs']:.1f}% |"
        )
    md_lines.append("")

    return audit_grp, df_param_audit, md_lines


def build_master_summary_table(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Genera la tabla resumen de las 32 configuraciones en CV y OOT (Media, DE, Mediana, IQR)."""
    summary = (
        df.groupby(["val_scheme", "cohort", "feature_set", "model_type", "class_weight_mode"])
        .agg(
            n_runs=("run_id", "count"),
            prauc_mean=("test_pr_auc", "mean"),
            prauc_sd=("test_pr_auc", "std"),
            prauc_median=("test_pr_auc", "median"),
            prauc_iqr=("test_pr_auc", lambda x: float(np.percentile(x, 75) - np.percentile(x, 25))),
            rocauc_mean=("test_roc_auc", "mean"),
            rocauc_sd=("test_roc_auc", "std"),
            brier_mean=("test_brier_score", "mean"),
            brier_sd=("test_brier_score", "std"),
            logloss_mean=("test_log_loss", "mean"),
            thresh_youden_mean=("threshold_youden", "mean"),
            sens_youden_mean=("sens_youden", "mean"),
            spec_youden_mean=("spec_youden", "mean"),
            f1_youden_mean=("f1_youden", "mean"),
            sens_050_mean=("sens_050", "mean"),
            spec_050_mean=("spec_050", "mean"),
            f1_050_mean=("f1_050", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "stage1_summary_all_configs.csv", index=False)
    return summary


def analyze_stage1_class_weights(
    df_cv: pd.DataFrame, out_dir: Path
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Etapa 1.1: Compara 'none' vs 'balanced' pareando exactamente por (repeat_idx, fold_idx)
    dentro de cada (cohort, feature_set, model_type).
    """
    md_lines = []
    md_lines.append("## Etapa 1: Decisiones Algorítmicas (Validación Cruzada 5x5)")
    md_lines.append("### 1.1. Impacto del Balanceo de Costos (`none` vs. `balanced`)")

    records = []
    keys = ["cohort", "feature_set", "model_type"]
    for (cohort, fset, model), grp in df_cv.groupby(keys):
        sub_none = grp[grp["class_weight_mode"] == "none"].set_index(
            ["repeat_idx", "fold_idx"]
        )
        sub_bal = grp[grp["class_weight_mode"] == "balanced"].set_index(
            ["repeat_idx", "fold_idx"]
        )

        common_idx = sub_none.index.intersection(sub_bal.index)
        if len(common_idx) == 0:
            continue

        s_none = sub_none.loc[common_idx]
        s_bal = sub_bal.loc[common_idx]

        n_test = float(s_none["n_test_rows"].mean())
        n_train = float(s_none["n_train_rows"].mean())

        # Diferencia: balanced - none
        diff_prauc = (s_bal["test_pr_auc"] - s_none["test_pr_auc"]).to_numpy()
        diff_roc = (s_bal["test_roc_auc"] - s_none["test_roc_auc"]).to_numpy()
        diff_brier = (s_bal["test_brier_score"] - s_none["test_brier_score"]).to_numpy()

        m_dprauc, ci_dprauc, _, p_prauc = nadeau_bengio_paired_ttest(
            diff_prauc, n_test, n_train
        )
        m_dbrier, _, _, p_brier = nadeau_bengio_paired_ttest(
            diff_brier, n_test, n_train
        )

        records.append(
            {
                "cohort": cohort,
                "feature_set": fset,
                "model_type": model,
                "n_paired_folds": len(common_idx),
                "prauc_none": float(s_none["test_pr_auc"].mean()),
                "prauc_balanced": float(s_bal["test_pr_auc"].mean()),
                "delta_prauc_bal_minus_none": m_dprauc,
                "p_val_prauc_nb": p_prauc,
                "rocauc_none": float(s_none["test_roc_auc"].mean()),
                "rocauc_balanced": float(s_bal["test_roc_auc"].mean()),
                "delta_rocauc": float(np.mean(diff_roc)),
                "brier_none": float(s_none["test_brier_score"].mean()),
                "brier_balanced": float(s_bal["test_brier_score"].mean()),
                "delta_brier_bal_minus_none": m_dbrier,
                "p_val_brier_nb": p_brier,
                "f1_none_at_youden": float(s_none["f1_youden"].mean()),
                "f1_bal_at_050": float(s_bal["f1_050"].mean()),
                "sens_none_at_youden": float(s_none["sens_youden"].mean()),
                "spec_none_at_youden": float(s_none["spec_youden"].mean()),
                "sens_bal_at_050": float(s_bal["sens_050"].mean()),
                "spec_bal_at_050": float(s_bal["spec_050"].mean()),
            }
        )

    df_cw = pd.DataFrame(records)
    df_cw.to_csv(out_dir / "stage1_class_weight_comparison.csv", index=False)

    md_lines.append(
        "| Cohorte | Set de Variables | Modelo | PR-AUC (`none`) | PR-AUC (`bal`) | Δ PR-AUC (p-NB) | Brier (`none`) | Brier (`bal`) | Sens/Spec (`none` @Youden) | Sens/Spec (`bal` @0.50) |"
    )
    md_lines.append(
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for _, r in df_cw.iterrows():
        md_lines.append(
            f"| {r['cohort']} | {r['feature_set']} | {r['model_type'].upper()} | "
            f"{r['prauc_none']:.4f} | {r['prauc_balanced']:.4f} | "
            f"{r['delta_prauc_bal_minus_none']:+.4f} (p={r['p_val_prauc_nb']:.3f}) | "
            f"{r['brier_none']:.4f} | {r['brier_balanced']:.4f} | "
            f"{r['sens_none_at_youden']:.2f} / {r['spec_none_at_youden']:.2f} | "
            f"{r['sens_bal_at_050']:.2f} / {r['spec_bal_at_050']:.2f} |"
        )
    md_lines.append("")
    return df_cw, md_lines


def analyze_stage1_models_head_to_head(
    df_cv: pd.DataFrame, out_dir: Path, rope: float = 0.005
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Etapa 1.2: Compara LightGBM vs Regresión Logística pareando los 25 folds
    con el test corregido de Nadeau-Bengio y evaluando una ROPE de +-0.005 en PR-AUC.
    """
    md_lines = []
    md_lines.append(
        f"### 1.2. Comparación Pareada LightGBM vs. Regresión Logística (Test Nadeau-Bengio, ROPE = ±{rope})"
    )

    records = []
    keys = ["cohort", "feature_set", "class_weight_mode"]
    for (cohort, fset, cw), grp in df_cv.groupby(keys):
        sub_lgbm = grp[grp["model_type"] == "lgbm"].set_index(
            ["repeat_idx", "fold_idx"]
        )
        sub_lreg = grp[grp["model_type"] == "logreg"].set_index(
            ["repeat_idx", "fold_idx"]
        )

        common_idx = sub_lgbm.index.intersection(sub_lreg.index)
        if len(common_idx) == 0:
            continue

        s_lgbm = sub_lgbm.loc[common_idx]
        s_lreg = sub_lreg.loc[common_idx]

        n_test = float(s_lgbm["n_test_rows"].mean())
        n_train = float(s_lgbm["n_train_rows"].mean())

        diff_prauc = (s_lgbm["test_pr_auc"] - s_lreg["test_pr_auc"]).to_numpy()
        diff_roc = (s_lgbm["test_roc_auc"] - s_lreg["test_roc_auc"]).to_numpy()
        diff_brier = (
            s_lgbm["test_brier_score"] - s_lreg["test_brier_score"]
        ).to_numpy()

        m_dprauc, ci_dprauc, t_prauc, p_prauc = nadeau_bengio_paired_ttest(
            diff_prauc, n_test, n_train
        )
        m_droc, ci_droc, _, p_roc = nadeau_bengio_paired_ttest(
            diff_roc, n_test, n_train
        )

        # Conteo de victorias de LightGBM y empates dentro de la ROPE
        wins_lgbm = int(np.sum(diff_prauc > rope))
        wins_lreg = int(np.sum(diff_prauc < -rope))
        ties_rope = int(np.sum(np.abs(diff_prauc) <= rope))

        # Veredicto práctico-estadístico
        if p_prauc < 0.05 and m_dprauc > rope:
            verdict = "Gana LGBM (Significativo)"
        elif p_prauc < 0.05 and m_dprauc < -rope:
            verdict = "Gana LogReg (Significativo)"
        elif abs(m_dprauc) <= rope:
            verdict = "Equivalencia Práctica (Elegir LogReg por parsimonia)"
        else:
            verdict = "Diferencia No Significativa (Ruido muestral)"

        records.append(
            {
                "cohort": cohort,
                "feature_set": fset,
                "class_weight_mode": cw,
                "n_folds": len(common_idx),
                "lgbm_prauc_mean": float(s_lgbm["test_pr_auc"].mean()),
                "lgbm_prauc_sd": float(s_lgbm["test_pr_auc"].std()),
                "logreg_prauc_mean": float(s_lreg["test_pr_auc"].mean()),
                "logreg_prauc_sd": float(s_lreg["test_pr_auc"].std()),
                "delta_prauc_lgbm_minus_logreg": m_dprauc,
                "delta_prauc_ci95_hw": ci_dprauc,
                "t_stat_nb": t_prauc,
                "p_val_prauc_nb": p_prauc,
                "lgbm_rocauc_mean": float(s_lgbm["test_roc_auc"].mean()),
                "logreg_rocauc_mean": float(s_lreg["test_roc_auc"].mean()),
                "delta_rocauc": m_droc,
                "p_val_rocauc_nb": p_roc,
                "delta_brier": float(np.mean(diff_brier)),
                "folds_lgbm_wins": wins_lgbm,
                "folds_ties_rope": ties_rope,
                "folds_logreg_wins": wins_lreg,
                "verdict": verdict,
            }
        )

    df_comp = pd.DataFrame(records)
    df_comp.to_csv(
        out_dir / "stage1_model_comparison_lgbm_vs_logreg.csv", index=False
    )

    md_lines.append(
        "| Cohorte | Set de Variables | Peso | LGBM PR-AUC | LogReg PR-AUC | Δ PR-AUC (±IC95%) | p-NB | Folds (LGBM / Tie / LogReg) | Veredicto Metodológico |"
    )
    md_lines.append(
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    )
    for _, r in df_comp.iterrows():
        md_lines.append(
            f"| {r['cohort']} | {r['feature_set']} | {r['class_weight_mode']} | "
            f"{r['lgbm_prauc_mean']:.4f} (±{r['lgbm_prauc_sd']:.4f}) | "
            f"{r['logreg_prauc_mean']:.4f} (±{r['logreg_prauc_sd']:.4f}) | "
            f"{r['delta_prauc_lgbm_minus_logreg']:+.4f} (±{r['delta_prauc_ci95_hw']:.4f}) | "
            f"{r['p_val_prauc_nb']:.4f} | "
            f"{r['folds_lgbm_wins']} / {r['folds_ties_rope']} / {r['folds_logreg_wins']} | "
            f"**{r['verdict']}** |"
        )
    md_lines.append("")
    return df_comp, md_lines


def main() -> None:
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    outputs_dir = Path(config["paths"]["outputs_dir"])
    cons_path = outputs_dir / "consolidated" / "master_fold_metrics.csv"
    out_dir = outputs_dir / "analysis" / "stage_0_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cons_path.exists():
        raise FileNotFoundError(
            f"No se encontró {cons_path}. Ejecutá primero 03_consolidate_results.py"
        )

    df = pd.read_csv(cons_path)

    # 1. Ejecutar Etapa 0
    _, _, md_s0 = audit_stage0(df, out_dir)

    # 2. Generar tabla resumen de las 32 configuraciones
    build_master_summary_table(df, out_dir)

    # 3. Ejecutar Etapa 1 (sobre las corridas de Validación Cruzada 5x5)
    df_cv = df[df["val_scheme"] == "repeated_group_cv"].copy()
    _, md_s1_cw = analyze_stage1_class_weights(df_cv, out_dir)
    _, md_s1_mod = analyze_stage1_models_head_to_head(df_cv, out_dir)

    # 4. Ensamblar y guardar el reporte en Markdown
    full_report = "\n".join(md_s0 + md_s1_cw + md_s1_mod)
    report_path = out_dir / "reporte_etapas_0_y_1.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(full_report)
    print("=" * 80)
    print(f"✔ Análisis de Etapas 0 y 1 completado. Archivos guardados en: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()