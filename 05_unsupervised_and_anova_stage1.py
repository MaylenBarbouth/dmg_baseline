# 05_unsupervised_and_anova_stage1.py
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Seguro para correr por consola/SSH sin entorno gráfico activo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols


CLUSTER_ORDER = [
    "C1: Élite OOT Temporal (Coh. A: Clínico / Full-LGBM)",
    "C2: Élite CV 5x5 (Coh. A: Clínico / Full-LGBM)",
    "C3: Moderado Bien Calibrado (Dominado por LGBM)",
    "C4: Moderado Descalibrado (LogReg/LGBM 'balanced')",
    "C5: Colapso a Azar (LogReg 'none' en Sets Lab)",
]

PALETTE_CLUSTERS = {
    "C1: Élite OOT Temporal (Coh. A: Clínico / Full-LGBM)": "#1b7837",
    "C2: Élite CV 5x5 (Coh. A: Clínico / Full-LGBM)": "#5aae61",
    "C3: Moderado Bien Calibrado (Dominado por LGBM)": "#2b83ba",
    "C4: Moderado Descalibrado (LogReg/LGBM 'balanced')": "#fdae61",
    "C5: Colapso a Azar (LogReg 'none' en Sets Lab)": "#d7191c",
}


def compute_anova_eta2(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Ajusta un ANOVA Factorial Tipo II con interacciones clave y calcula Eta^2 (%)."""
    formula = (
        f"{target_col} ~ C(cohort) + C(feature_set) + C(model_type) + C(class_weight_mode) + C(val_scheme) "
        f"+ C(cohort):C(feature_set) + C(model_type):C(class_weight_mode) + C(model_type):C(feature_set)"
    )
    model = ols(formula, data=data).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = aov["sum_sq"].sum()
    aov["eta_sq_pct"] = (aov["sum_sq"] / ss_total) * 100.0
    return aov[["sum_sq", "df", "F", "PR(>F)", "eta_sq_pct"]]


def assign_semantic_clusters(df: pd.DataFrame, raw_labels: np.ndarray) -> Dict[int, str]:
    """
    Asigna los nombres semánticos C1..C5 determinísticamente según las propiedades
    de cada centroide, evitando permutaciones arbitrarias de KMeans entre distintas CPUs.
    """
    cluster_stats = df.groupby(raw_labels).agg(
        prauc=("prauc_mean", "mean"),
        rocauc=("rocauc_mean", "mean"),
        brier=("brier_mean", "mean"),
        oot_frac=("val_scheme", lambda s: (s == "temporal_oot").mean()),
    )

    # C5: El cluster donde colapsa el ROC-AUC (~0.495)
    c5_id = int(cluster_stats["rocauc"].idxmin())
    rem_ids = [int(i) for i in cluster_stats.index if int(i) != c5_id]

    # Los 2 de mayor PR-AUC son los clusters Élite (C1 en OOT y C2 en CV)
    top2_ids = (
        cluster_stats.loc[rem_ids]
        .sort_values(by="prauc", ascending=False)
        .index[:2]
        .tolist()
    )
    if cluster_stats.loc[top2_ids[0], "oot_frac"] >= cluster_stats.loc[top2_ids[1], "oot_frac"]:
        c1_id, c2_id = int(top2_ids[0]), int(top2_ids[1])
    else:
        c1_id, c2_id = int(top2_ids[1]), int(top2_ids[0])

    # Los 2 restantes son los moderados: C3 (menor Brier = calibrado) y C4 (mayor Brier = descalibrado)
    mod_ids = [i for i in rem_ids if i not in (c1_id, c2_id)]
    if cluster_stats.loc[mod_ids[0], "brier"] <= cluster_stats.loc[mod_ids[1], "brier"]:
        c3_id, c4_id = int(mod_ids[0]), int(mod_ids[1])
    else:
        c3_id, c4_id = int(mod_ids[1]), int(mod_ids[0])

    return {
        c1_id: CLUSTER_ORDER[0],
        c2_id: CLUSTER_ORDER[1],
        c3_id: CLUSTER_ORDER[2],
        c4_id: CLUSTER_ORDER[3],
        c5_id: CLUSTER_ORDER[4],
    }


def plot_fig1_pca_and_clusters(
    df: pd.DataFrame,
    metric_cols: List[str],
    components: np.ndarray,
    explained_var: np.ndarray,
    out_dir: Path,
) -> Path:
    """Genera y guarda la Figura 1: Biplot PCA + Heatmap de perfiles por Cluster."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17, 7.2), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    for cl_name in CLUSTER_ORDER:
        sub = df[df["Cluster_Label"] == cl_name]
        ax1.scatter(
            sub["PC1_Discriminacion"],
            sub["PC2_Calibracion"],
            label=f"{cl_name} (n={len(sub)})",
            color=PALETTE_CLUSTERS[cl_name],
            s=95,
            alpha=0.88,
            edgecolors="black",
            linewidth=0.7,
        )

    vector_specs = [
        ("prauc_mean", "PR-AUC", (0.35, 0.32)),
        ("rocauc_mean", "ROC-AUC", (0.45, 0.05)),
        ("f1_youden_mean", "F1 (Youden)", (0.45, -0.22)),
        ("brier_mean", "Brier / Log-Loss", (-0.85, -0.25)),
        ("thresh_youden_mean", "Umbral Youden", (0.75, -0.15)),
        ("sens_youden_mean", "Sensibilidad", (0.35, -0.18)),
        ("spec_youden_mean", "Especificidad", (-0.25, 0.25)),
    ]

    scale_vec = 5.5
    for col, label, (dx, dy) in vector_specs:
        idx = metric_cols.index(col)
        vx = components[0, idx] * scale_vec
        vy = components[1, idx] * scale_vec
        ax1.arrow(
            0, 0, vx, vy,
            color="#333333", alpha=0.7,
            head_width=0.14, head_length=0.16, linewidth=1.3,
        )
        ax1.text(
            vx + dx, vy + dy, label,
            fontsize=9, fontweight="bold", color="#111111",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                alpha=0.85,
                edgecolor="#cccccc",
                linewidth=0.5,
            ),
        )

    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_xlim(-5.0, 4.8)
    ax1.set_ylim(-4.6, 5.0)
    ax1.set_title(
        "A. Biplot PCA de las 64 Configuraciones (16 Métricas)\nAgrupamiento No Supervisado (K-Means, k=5)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax1.set_xlabel(
        f"PC1: Poder Discriminativo Global ({explained_var[0]*100:.1f}% varianza) → Mejor",
        fontsize=10.5, fontweight="bold",
    )
    ax1.set_ylabel(
        f"PC2: Calibración Probabilística ({explained_var[1]*100:.1f}% varianza) → Mejor Calibrado",
        fontsize=10.5, fontweight="bold",
    )
    ax1.legend(
        loc="upper left", fontsize=8.3, frameon=True,
        facecolor="white", framealpha=0.92, title="Clusters Identificados:",
    )
    ax1.grid(True, linestyle=":", alpha=0.4)

    display_metrics = [
        "prauc_mean", "rocauc_mean", "f1_youden_mean",
        "sens_youden_mean", "spec_youden_mean",
        "brier_mean", "logloss_mean", "thresh_youden_mean",
    ]
    display_names = [
        "PR-AUC", "ROC-AUC", "F1 (Youden)",
        "Sens (Youden)", "Spec (Youden)",
        "Brier Score ↓", "Log-Loss ↓", "Umbral Youden",
    ]

    cluster_means_raw = df.groupby("Cluster_Label")[display_metrics].mean().loc[CLUSTER_ORDER]
    cluster_means_z = (cluster_means_raw - df[display_metrics].mean()) / df[display_metrics].std()

    short_labels = [
        f"C1: Élite OOT\n(n={int((df['Cluster_Label']==CLUSTER_ORDER[0]).sum())})",
        f"C2: Élite CV 5x5\n(n={int((df['Cluster_Label']==CLUSTER_ORDER[1]).sum())})",
        f"C3: Moderado\nCalibrado (n={int((df['Cluster_Label']==CLUSTER_ORDER[2]).sum())})",
        f"C4: Moderado\nDescalibrado (n={int((df['Cluster_Label']==CLUSTER_ORDER[3]).sum())})",
        f"C5: Colapso\nLogReg-None (n={int((df['Cluster_Label']==CLUSTER_ORDER[4]).sum())})",
    ]
    cluster_means_z.index = short_labels
    cluster_means_raw.index = short_labels
    cluster_means_z.columns = display_names
    cluster_means_raw.columns = display_names

    sns.heatmap(
        cluster_means_z, ax=ax2, cmap="RdYlBu_r", center=0,
        annot=cluster_means_raw.round(3), fmt=".3f",
        cbar_kws={"label": "Z-Score respecto al promedio global"},
        linewidths=1, linecolor="white",
        annot_kws={"size": 9.5, "fontweight": "bold"},
    )
    ax2.set_title(
        "B. Perfil Cuantitativo Medio de cada Cluster\n(Valores reales anotados; color = Z-score relativo)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax2.set_ylabel("")
    ax2.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9.5, fontweight="bold")
    ax2.set_yticklabels(short_labels, rotation=0, fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    fig1_path = out_dir / "fig1_unsupervised_pca_and_clusters.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig1_path


def plot_fig2_anova_and_interactions(df: pd.DataFrame, out_dir: Path) -> Path:
    """Genera y guarda la Figura 2: Descomposición de Varianza ANOVA (Eta^2) + Escalera de Ablación."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(17, 6.8), gridspec_kw={"width_ratios": [1.05, 1.2]}
    )

    target_map = {
        "prauc_mean": "PR-AUC\n(Ranking Positivos)",
        "rocauc_mean": "ROC-AUC\n(Discriminación)",
        "brier_mean": "Brier Score\n(Calibración)",
        "f1_youden_mean": "F1-Youden\n(Corte Operativo)",
    }

    factor_labels = {
        "C(cohort)": "Cohorte (A vs B)",
        "C(feature_set)": "Set de Variables (1, 2a, 2b, 3)",
        "C(cohort):C(feature_set)": "Interacción: Cohorte × Set Variables",
        "C(model_type)": "Algoritmo (LGBM vs LogReg)",
        "C(class_weight_mode)": "Balanceo (none vs balanced)",
        "C(model_type):C(class_weight_mode)": "Interacción: Algoritmo × Balanceo",
        "C(model_type):C(feature_set)": "Interacción: Algoritmo × Set Variables",
        "C(val_scheme)": "Esquema Validación (CV vs OOT)",
        "Residual": "Residuo (No explicado)",
    }

    eta_df = pd.DataFrame()
    pval_df = pd.DataFrame()
    for col, name in target_map.items():
        aov = compute_anova_eta2(df, col)
        eta_df[name] = aov["eta_sq_pct"]
        pval_df[name] = aov["PR(>F)"]

    eta_df.index = [factor_labels[i] for i in eta_df.index]
    pval_df.index = [factor_labels[i] for i in pval_df.index]

    order_rows = [
        "Cohorte (A vs B)",
        "Set de Variables (1, 2a, 2b, 3)",
        "Interacción: Cohorte × Set Variables",
        "Algoritmo (LGBM vs LogReg)",
        "Balanceo (none vs balanced)",
        "Interacción: Algoritmo × Balanceo",
        "Interacción: Algoritmo × Set Variables",
        "Esquema Validación (CV vs OOT)",
        "Residuo (No explicado)",
    ]
    eta_df = eta_df.loc[order_rows]
    pval_df = pval_df.loc[order_rows]

    annot_matrix = pd.DataFrame(index=eta_df.index, columns=eta_df.columns)
    for r in eta_df.index:
        for c in eta_df.columns:
            val = eta_df.loc[r, c]
            p = pval_df.loc[r, c]
            if pd.isna(p):
                stars = ""
            elif p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = " (ns)"
            annot_matrix.loc[r, c] = f"{val:.1f}%{stars}"

    sns.heatmap(
        eta_df, ax=ax1, cmap="YlGnBu", annot=annot_matrix, fmt="",
        cbar_kws={"label": "% de Varianza Total Explicada (η²)"},
        linewidths=1, linecolor="white",
        annot_kws={"size": 9.5, "fontweight": "bold"},
    )
    ax1.set_title(
        "A. ¿Qué Decisión Impacta Más en Cada Métrica?\nANOVA Factorial Tipo II (% Varianza Explicada η² y Significancia)",
        fontsize=11.5, fontweight="bold", pad=10,
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=9.5, fontweight="bold")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=9.5, fontweight="bold")

    df_cv = df[df["val_scheme"] == "repeated_group_cv"].copy()
    df_cv["Group_Line"] = (
        df_cv["cohort"].map({"A_all": "Coh. A", "B_no_antc": "Coh. B"})
        + " | "
        + df_cv["model_type"].str.upper()
        + " ("
        + df_cv["class_weight_mode"]
        + ")"
    )

    fset_order = ["set_1_clinico", "set_2a_lab_sin_glu", "set_2b_lab_completo", "set_3_full"]
    fset_labels = [
        "Set 1:\nClínico Puro",
        "Set 2a:\nLab (Sin Glu)",
        "Set 2b:\nLab Completo",
        "Set 3:\nFull (Clín + Lab)",
    ]

    style_map = {
        "Coh. A | LGBM (none)": ("#1b7837", "-", "o", 2.6),
        "Coh. A | LGBM (balanced)": ("#5aae61", "--", "s", 1.8),
        "Coh. A | LOGREG (balanced)": ("#fdae61", "--", "^", 2.2),
        "Coh. A | LOGREG (none)": ("#d7191c", "-", "v", 2.4),
        "Coh. B | LGBM (none)": ("#2b83ba", "-", "o", 2.4),
        "Coh. B | LGBM (balanced)": ("#6baed6", "--", "s", 1.6),
        "Coh. B | LOGREG (balanced)": ("#9e9ac8", "--", "^", 2.0),
        "Coh. B | LOGREG (none)": ("#fb6a4a", ":", "v", 2.0),
    }

    for grp_name, (color, ls, marker, lw) in style_map.items():
        sub = df_cv[df_cv["Group_Line"] == grp_name].set_index("feature_set").loc[fset_order]
        ax2.plot(
            range(4), sub["prauc_mean"],
            label=grp_name, color=color, linestyle=ls, marker=marker,
            linewidth=lw, markersize=7.5, alpha=0.92,
        )

    ax2.annotate(
        "Colapso de LogReg (none)\nal incluir variables de Lab",
        xy=(2, 0.0866), xytext=(1.25, 0.045),
        arrowprops=dict(arrowstyle="->", color="#d7191c", lw=1.5),
        fontsize=9, fontweight="bold", color="#b30000",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5f0", edgecolor="#d7191c"),
    )

    ax2.annotate(
        "Techo Clínico en Coh. A (~0.247)\nvs Sinergia LGBM en Set 3 (0.264)",
        xy=(3, 0.2642), xytext=(1.35, 0.255),
        arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.5),
        fontsize=9, fontweight="bold", color="#00441b",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f7fcf5", edgecolor="#1b7837"),
    )

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(fset_labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0.03, 0.295)
    ax2.set_ylabel("PR-AUC Medio en Test (CV 5x5)", fontsize=10.5, fontweight="bold")
    ax2.set_title(
        "B. Escalera de Ablación e Interacciones Críticas (CV 5x5)\nEfecto de Cohorte × Set de Variables × (Modelo + Balanceo)",
        fontsize=11.5, fontweight="bold", pad=10,
    )
    ax2.legend(
        loc="center left", bbox_to_anchor=(0.01, 0.62),
        fontsize=8.3, ncol=2, frameon=True, facecolor="white", framealpha=0.9,
    )
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig2_path = out_dir / "fig2_variance_decomposition_and_effects.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig2_path


def run_paired_and_omnibus_tests(df: pd.DataFrame, out_dir: Path) -> None:
    """Ejecuta tests de Wilcoxon pareados (globales y estratificados) y Friedman ómnibus."""
    all_factors = ["val_scheme", "cohort", "feature_set", "model_type", "class_weight_mode"]
    binary_specs = [
        ("cohort", "A_all", "B_no_antc"),
        ("model_type", "lgbm", "logreg"),
        ("class_weight_mode", "balanced", "none"),
        ("val_scheme", "temporal_oot", "repeated_group_cv"),
    ]

    records = []
    for factor, lvl1, lvl2 in binary_specs:
        other_cols = [c for c in all_factors if c != factor]
        s1 = df[df[factor] == lvl1].set_index(other_cols).sort_index()
        s2 = df[df[factor] == lvl2].set_index(other_cols).sort_index()

        for metric in ["prauc_mean", "rocauc_mean", "brier_mean", "f1_youden_mean"]:
            diff = s1[metric] - s2[metric]
            _, p_w = stats.wilcoxon(s1[metric], s2[metric])
            _, p_t = stats.ttest_rel(s1[metric], s2[metric])
            records.append(
                {
                    "Stratum": "Global (32 pares)",
                    "Factor": factor,
                    "Comparison": f"{lvl1} vs {lvl2}",
                    "Metric": metric,
                    "Mean_1": float(s1[metric].mean()),
                    "Mean_2": float(s2[metric].mean()),
                    "Delta_Mean": float(diff.mean()),
                    "Delta_Median": float(diff.median()),
                    "Wilcoxon_p": float(p_w),
                    "Paired_t_p": float(p_t),
                }
            )

    for m in ["lgbm", "logreg"]:
        sub = df[df["model_type"] == m]
        other = ["val_scheme", "cohort", "feature_set"]
        s_bal = sub[sub["class_weight_mode"] == "balanced"].set_index(other).sort_index()
        s_none = sub[sub["class_weight_mode"] == "none"].set_index(other).sort_index()
        for metric in ["prauc_mean", "rocauc_mean", "brier_mean", "f1_youden_mean"]:
            diff = s_bal[metric] - s_none[metric]
            _, p_w = stats.wilcoxon(s_bal[metric], s_none[metric])
            _, p_t = stats.ttest_rel(s_bal[metric], s_none[metric])
            records.append(
                {
                    "Stratum": f"Solo {m.upper()} (16 pares)",
                    "Factor": "class_weight_mode",
                    "Comparison": "balanced vs none",
                    "Metric": metric,
                    "Mean_1": float(s_bal[metric].mean()),
                    "Mean_2": float(s_none[metric].mean()),
                    "Delta_Mean": float(diff.mean()),
                    "Delta_Median": float(diff.median()),
                    "Wilcoxon_p": float(p_w),
                    "Paired_t_p": float(p_t),
                }
            )

    for cw in ["none", "balanced"]:
        sub = df[df["class_weight_mode"] == cw]
        other = ["val_scheme", "cohort", "feature_set"]
        s_lgbm = sub[sub["model_type"] == "lgbm"].set_index(other).sort_index()
        s_lreg = sub[sub["model_type"] == "logreg"].set_index(other).sort_index()
        for metric in ["prauc_mean", "rocauc_mean", "brier_mean", "f1_youden_mean"]:
            diff = s_lgbm[metric] - s_lreg[metric]
            _, p_w = stats.wilcoxon(s_lgbm[metric], s_lreg[metric])
            _, p_t = stats.ttest_rel(s_lgbm[metric], s_lreg[metric])
            records.append(
                {
                    "Stratum": f"Solo weight={cw} (16 pares)",
                    "Factor": "model_type",
                    "Comparison": "lgbm vs logreg",
                    "Metric": metric,
                    "Mean_1": float(s_lgbm[metric].mean()),
                    "Mean_2": float(s_lreg[metric].mean()),
                    "Delta_Mean": float(diff.mean()),
                    "Delta_Median": float(diff.median()),
                    "Wilcoxon_p": float(p_w),
                    "Paired_t_p": float(p_t),
                }
            )

    pd.DataFrame(records).to_csv(out_dir / "stage1_wilcoxon_paired_factors.csv", index=False)

    other_f = ["val_scheme", "cohort", "model_type", "class_weight_mode"]
    piv_f = df.pivot(
        index=other_f,
        columns="feature_set",
        values=["prauc_mean", "rocauc_mean", "brier_mean", "f1_youden_mean"],
    )
    friedman_rows = []
    for metric in ["prauc_mean", "rocauc_mean", "brier_mean", "f1_youden_mean"]:
        sub_p = piv_f[metric]
        stat_fr, p_fr = stats.friedmanchisquare(
            sub_p["set_1_clinico"],
            sub_p["set_2a_lab_sin_glu"],
            sub_p["set_2b_lab_completo"],
            sub_p["set_3_full"],
        )
        friedman_rows.append(
            {
                "Metric": metric,
                "Friedman_Chi2": float(stat_fr),
                "Friedman_p": float(p_fr),
                "Mean_set_1_clinico": float(sub_p["set_1_clinico"].mean()),
                "Mean_set_2a_lab_sin_glu": float(sub_p["set_2a_lab_sin_glu"].mean()),
                "Mean_set_2b_lab_completo": float(sub_p["set_2b_lab_completo"].mean()),
                "Mean_set_3_full": float(sub_p["set_3_full"].mean()),
            }
        )
    pd.DataFrame(friedman_rows).to_csv(
        out_dir / "stage1_friedman_omnibus_featuresets.csv", index=False
    )


def main() -> None:
    default_path = Path("outputs/analysis/stage_0_1/stage1_summary_all_configs.csv")
    csv_path = default_path if default_path.exists() else Path("stage1_summary_all_configs.csv")
    out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    cat_cols = ["val_scheme", "cohort", "feature_set", "model_type", "class_weight_mode"]
    metric_cols = [c for c in df.columns if c not in cat_cols + ["n_runs"]]

    # 1. PCA con orientación determinística de signos
    X_scaled = StandardScaler().fit_transform(df[metric_cols])
    pca = PCA()
    Z = pca.fit_transform(X_scaled)
    components = pca.components_.copy()

    # Forzar PC1 (+) = mayor PR-AUC (mejor discriminación)
    if np.corrcoef(Z[:, 0], df["prauc_mean"])[0, 1] < 0:
        Z[:, 0] *= -1
        components[0, :] *= -1

    # Forzar PC2 (+) = menor Brier Score (mejor calibración)
    if np.corrcoef(Z[:, 1], df["brier_mean"])[0, 1] > 0:
        Z[:, 1] *= -1
        components[1, :] *= -1

    df["PC1_Discriminacion"] = Z[:, 0]
    df["PC2_Calibracion"] = Z[:, 1]
    df["PC3_Volatilidad_Spec"] = Z[:, 2]

    # 2. K-Means (k=5) con mapeo semántico determinístico
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_scaled)
    df["cluster_id"] = km.labels_
    label_map = assign_semantic_clusters(df, km.labels_)
    df["Cluster_Label"] = df["cluster_id"].map(label_map)

    df.to_csv(out_dir / "stage1_configs_with_pca_and_clusters.csv", index=False)

    # 3. Exportar tablas ANOVA Eta^2, Wilcoxon y Friedman
    for target in ["prauc_mean", "rocauc_mean", "brier_mean", "f1_youden_mean"]:
        aov = compute_anova_eta2(df, target)
        aov.to_csv(out_dir / f"stage1_anova_eta2_{target}.csv")

    run_paired_and_omnibus_tests(df, out_dir)

    # 4. Generar y guardar las 2 figuras en alta resolución (300 DPI)
    fig1_path = plot_fig1_pca_and_clusters(
        df, metric_cols, components, pca.explained_variance_ratio_, out_dir
    )
    fig2_path = plot_fig2_anova_and_interactions(df, out_dir)

    print(f"✔ Tablas CSV exportadas en: {out_dir}")
    print(f"✔ Figura 1 generada en    : {fig1_path}")
    print(f"✔ Figura 2 generada en    : {fig2_path}")


if __name__ == "__main__":
    main()
