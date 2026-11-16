# 06_cluster_composition_and_rules.py
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text


CLUSTER_ORDER = [
    "C1: Élite OOT Temporal",
    "C2: Élite CV 5x5",
    "C3: Moderado Calibrado",
    "C4: Moderado Descalibrado",
    "C5: Colapso LogReg-None",
]


def assign_semantic_clusters(df: pd.DataFrame, raw_labels: np.ndarray) -> pd.Series:
    """Asigna los nombres C1..C5 determinísticamente según los centroides de cada grupo."""
    cs = df.groupby(raw_labels).agg(
        prauc=("prauc_mean", "mean"),
        rocauc=("rocauc_mean", "mean"),
        brier=("brier_mean", "mean"),
        oot_frac=("val_scheme", lambda s: (s == "temporal_oot").mean()),
    )
    c5 = int(cs["rocauc"].idxmin())
    rem = [int(i) for i in cs.index if int(i) != c5]
    top2 = cs.loc[rem].sort_values(by="prauc", ascending=False).index[:2].tolist()
    if cs.loc[top2[0], "oot_frac"] >= cs.loc[top2[1], "oot_frac"]:
        c1, c2 = int(top2[0]), int(top2[1])
    else:
        c1, c2 = int(top2[1]), int(top2[0])
    mod = [i for i in rem if i not in (c1, c2)]
    if cs.loc[mod[0], "brier"] <= cs.loc[mod[1], "brier"]:
        c3, c4 = int(mod[0]), int(mod[1])
    else:
        c3, c4 = int(mod[1]), int(mod[0])

    mapping = {
        c1: CLUSTER_ORDER[0],
        c2: CLUSTER_ORDER[1],
        c3: CLUSTER_ORDER[2],
        c4: CLUSTER_ORDER[3],
        c5: CLUSTER_ORDER[4],
    }
    return pd.Series(raw_labels, index=df.index).map(mapping)


def cramers_v(confusion_matrix: pd.DataFrame) -> Tuple[float, float, float]:
    """Calcula el estadístico Chi-cuadrado, su p-valor y la V de Cramér."""
    chi2, p, _, _ = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    v = float(np.sqrt(phi2 / min(k - 1, r - 1)))
    return v, float(chi2), float(p)


def main() -> None:
    default_path = Path("outputs/analysis/stage_0_1/stage1_summary_all_configs.csv")
    csv_path = default_path if default_path.exists() else Path("stage1_summary_all_configs.csv")
    out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    cat_cols = ["val_scheme", "cohort", "feature_set", "model_type", "class_weight_mode"]
    metric_cols = [c for c in df.columns if c not in cat_cols + ["n_runs"]]

    # 1. Comparar K-Means en espacio completo (16D) vs. PCA reducido (3D y 4D)
    X_scaled = StandardScaler().fit_transform(df[metric_cols])
    Z = PCA().fit_transform(X_scaled)

    km_16d = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_scaled)
    km_pca3 = KMeans(n_clusters=5, random_state=42, n_init=20).fit(Z[:, :3])
    km_pca4 = KMeans(n_clusters=5, random_state=42, n_init=20).fit(Z[:, :4])

    df["Cluster"] = assign_semantic_clusters(df, km_16d.labels_)
    cl_pca3 = assign_semantic_clusters(df, km_pca3.labels_)
    cl_pca4 = assign_semantic_clusters(df, km_pca4.labels_)

    print("=" * 80)
    print("VERIFICACIÓN METODOLÓGICA: K-MEANS EN 16D VS. K-MEANS SOBRE PCA REDUCIDO")
    print("=" * 80)
    print(
        f"Coincidencia exacta 16D vs PCA(3 comp): {(df['Cluster'] == cl_pca3).sum()}/64 "
        f"({(df['Cluster'] == cl_pca3).mean()*100:.1f}%) | ARI = {adjusted_rand_score(km_16d.labels_, km_pca3.labels_):.4f}"
    )
    print(
        f"Coincidencia exacta 16D vs PCA(4 comp): {(df['Cluster'] == cl_pca4).sum()}/64 "
        f"({(df['Cluster'] == cl_pca4).mean()*100:.1f}%) | ARI = {adjusted_rand_score(km_16d.labels_, km_pca4.labels_):.4f}"
    )

    # 2. Calcular asociación estadística entre cada columna de diseño y los Clusters
    assoc_rows = []
    for col in cat_cols:
        ct = pd.crosstab(df["Cluster"], df[col]).loc[CLUSTER_ORDER]
        v, chi2, p = cramers_v(ct)
        nmi = float(normalized_mutual_info_score(df["Cluster"], df[col]))
        assoc_rows.append(
            {
                "Factor": col,
                "Chi2": chi2,
                "p_value": p,
                "Cramers_V": v,
                "NMI": nmi,
            }
        )
    df_assoc = pd.DataFrame(assoc_rows).sort_values(by="Cramers_V", ascending=False)
    df_assoc.to_csv(out_dir / "stage1_cluster_factor_association.csv", index=False)

    print("\nRANKING DE FACTORES QUE DETERMINAN LOS CLUSTERS (V de Cramér):")
    print(df_assoc.to_string(index=False))

    # 3. Entrenar Árbol de Decisión explicativo sobre las 5 columnas categóricas
    X_cat = pd.get_dummies(df[cat_cols])
    dt = DecisionTreeClassifier(random_state=42).fit(X_cat, df["Cluster"])
    rules_text = export_text(dt, feature_names=list(X_cat.columns))
    with open(out_dir / "stage1_cluster_decision_rules.txt", "w", encoding="utf-8") as f:
        f.write(rules_text)

    # 4. Generar Figura 3 (Composición % + Tablero de Ajedrez 8x8 de las 64 corridas)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    short_clusters = [
        "C1: Élite OOT (n=6)",
        "C2: Élite CV (n=6)",
        "C3: Mod. Calibrado (n=25)",
        "C4: Mod. Descalibrado (n=15)",
        "C5: Colapso LogReg (n=12)",
    ]

    factor_configs = [
        ("model_type", "1. Algoritmo (model_type)", ["lgbm", "logreg"], ["LightGBM", "LogReg"], ["#2b83ba", "#fdae61"]),
        ("class_weight_mode", "2. Balanceo (class_weight_mode)", ["none", "balanced"], ["none (Natural)", "balanced (Pesos)"], ["#66c2a5", "#fc8d62"]),
        ("cohort", "3. Cohorte (cohort)", ["A_all", "B_no_antc"], ["Cohorte A (General)", "Cohorte B (Sin ANTC)"], ["#4daf4a", "#984ea3"]),
        ("val_scheme", "4. Esquema Validación (val_scheme)", ["repeated_group_cv", "temporal_oot"], ["CV 5x5 (Poblacional)", "OOT (Temporal)"], ["#377eb8", "#e41a1c"]),
        ("feature_set", "5. Set de Variables (feature_set)", ["set_1_clinico", "set_2a_lab_sin_glu", "set_2b_lab_completo", "set_3_full"],
         ["Set 1: Clínico", "Set 2a: Lab s/Glu", "Set 2b: Lab Comp.", "Set 3: Full"], ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]),
    ]

    fig = plt.figure(figsize=(19.5, 10.5))
    gs = gridspec.GridSpec(5, 2, figure=fig, width_ratios=[1.0, 1.35], hspace=0.68, wspace=0.46)

    for idx, (col, title_lbl, levels, level_names, colors) in enumerate(factor_configs):
        ax = fig.add_subplot(gs[idx, 0])
        ct_counts = pd.crosstab(df["Cluster"], df[col]).loc[CLUSTER_ORDER, levels]
        ct_pct = ct_counts.div(ct_counts.sum(axis=1), axis=0) * 100.0
        v_val, _, p_val = cramers_v(ct_counts)
        p_str = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"

        left_cum = np.zeros(len(CLUSTER_ORDER))
        for lvl, lvl_name, color in zip(levels, level_names, colors):
            widths = ct_pct[lvl].values
            counts = ct_counts[lvl].values
            ax.barh(
                range(len(CLUSTER_ORDER)), widths, left=left_cum,
                color=color, edgecolor="white", height=0.68, label=lvl_name, alpha=0.92,
            )
            for i, (w, c) in enumerate(zip(widths, counts)):
                if w >= 12.0:
                    ax.text(
                        left_cum[i] + w / 2.0, i, f"{w:.0f}% ({c})",
                        ha="center", va="center",
                        color="white" if color not in ["#fdae61", "#66c2a5", "#fc8d62"] else "#111111",
                        fontsize=8.2, fontweight="bold",
                    )
            left_cum += widths

        ax.set_yticks(range(len(CLUSTER_ORDER)))
        ax.set_yticklabels(short_clusters, fontsize=8.6, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.axvline(50, color="#333333", linestyle="--", linewidth=0.9, alpha=0.5)
        if idx == 4:
            ax.set_xlabel("% de Configuraciones dentro del Cluster (Línea punteada = 50%)", fontsize=9, fontweight="bold", labelpad=5)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=len(levels), fontsize=8.2, frameon=False)
        else:
            ax.set_xticklabels([])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=len(levels), fontsize=8.2, frameon=False)
        ax.set_title(f"{title_lbl}  |  Cramér's V = {v_val:.2f} ({p_str})", fontsize=9.8, fontweight="bold", loc="left", pad=5)

    # Panel B: Tablero 8x8
    ax_mat = fig.add_subplot(gs[:, 1])
    row_order = [
        ("A_all", "set_1_clinico"), ("A_all", "set_2a_lab_sin_glu"),
        ("A_all", "set_2b_lab_completo"), ("A_all", "set_3_full"),
        ("B_no_antc", "set_1_clinico"), ("B_no_antc", "set_2a_lab_sin_glu"),
        ("B_no_antc", "set_2b_lab_completo"), ("B_no_antc", "set_3_full"),
    ]
    row_labels = [
        "Cohorte A | Set 1: Clínico", "Cohorte A | Set 2a: Lab s/Glu",
        "Cohorte A | Set 2b: Lab Comp.", "Cohorte A | Set 3: Full",
        "Cohorte B | Set 1: Clínico", "Cohorte B | Set 2a: Lab s/Glu",
        "Cohorte B | Set 2b: Lab Comp.", "Cohorte B | Set 3: Full",
    ]
    col_order = [
        ("lgbm", "none", "repeated_group_cv"), ("lgbm", "none", "temporal_oot"),
        ("lgbm", "balanced", "repeated_group_cv"), ("lgbm", "balanced", "temporal_oot"),
        ("logreg", "none", "repeated_group_cv"), ("logreg", "none", "temporal_oot"),
        ("logreg", "balanced", "repeated_group_cv"), ("logreg", "balanced", "temporal_oot"),
    ]
    col_labels = [
        "LGBM\nnone\n(CV 5x5)", "LGBM\nnone\n(OOT)",
        "LGBM\nbalanced\n(CV 5x5)", "LGBM\nbalanced\n(OOT)",
        "LogReg\nnone\n(CV 5x5)", "LogReg\nnone\n(OOT)",
        "LogReg\nbalanced\n(CV 5x5)", "LogReg\nbalanced\n(OOT)",
    ]

    cluster_to_int = {name: i for i, name in enumerate(CLUSTER_ORDER)}
    cluster_short_code = {name: f"C{i+1}" for i, name in enumerate(CLUSTER_ORDER)}

    mat_int = np.zeros((8, 8), dtype=int)
    mat_text = np.empty((8, 8), dtype=object)
    df_indexed = df.set_index(["cohort", "feature_set", "model_type", "class_weight_mode", "val_scheme"])

    for r_i, (coh, fset) in enumerate(row_order):
        for c_i, (mod, cw, val) in enumerate(col_order):
            row_data = df_indexed.loc[(coh, fset, mod, cw, val)]
            cl_name = row_data["Cluster"]
            c_code = cluster_short_code[cl_name]
            mat_int[r_i, c_i] = cluster_to_int[cl_name]
            mat_text[r_i, c_i] = f"{c_code}\nPR:{row_data['prauc_mean']:.2f}\nBr:{row_data['brier_mean']:.2f}"

    cmap_custom = ListedColormap(["#1b7837", "#5aae61", "#2b83ba", "#fdae61", "#d7191c"])
    sns.heatmap(
        mat_int, ax=ax_mat, cmap=cmap_custom, vmin=0, vmax=4,
        annot=mat_text, fmt="", cbar=False,
        linewidths=1.8, linecolor="white",
        annot_kws={"size": 8.6, "fontweight": "bold", "color": "white"},
    )

    for text_obj in ax_mat.texts:
        if text_obj.get_text().startswith("C4"):
            text_obj.set_color("#111111")

    ax_mat.axhline(4, color="#111111", linewidth=3.2)
    ax_mat.axvline(4, color="#111111", linewidth=3.2)
    ax_mat.axvline(2, color="#111111", linewidth=1.5, linestyle="--")
    ax_mat.axvline(6, color="#111111", linewidth=1.5, linestyle="--")

    ax_mat.set_xticklabels(col_labels, rotation=0, fontsize=8.8, fontweight="bold")
    ax_mat.set_yticklabels(row_labels, rotation=0, fontsize=9.2, fontweight="bold")
    ax_mat.set_title(
        "B. Tablero de Decisión Completo (8×8 = 64 Configuraciones)\n"
        "(Cada celda indica: Cluster [C1..C5] | PR-AUC medio | Brier Score medio)",
        fontsize=11.5, fontweight="bold", pad=12,
    )

    legend_patches = [
        mpatches.Patch(color="#1b7837", label="C1: Élite OOT Temporal (PR~0.33, Br~0.14)"),
        mpatches.Patch(color="#5aae61", label="C2: Élite CV 5x5 (PR~0.25, Br~0.12)"),
        mpatches.Patch(color="#2b83ba", label="C3: Moderado Calibrado (PR~0.12, Br~0.11)"),
        mpatches.Patch(color="#fdae61", label="C4: Moderado Descalibrado (PR~0.13, Br~0.25)"),
        mpatches.Patch(color="#d7191c", label="C5: Colapso LogReg-None (PR~0.09, ROC~0.50)"),
    ]
    ax_mat.legend(
        handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.09),
        ncol=2, fontsize=8.6, frameon=True, facecolor="white", title="Código de Colores de los Clusters:",
    )

    fig.suptitle(
        "Radiografía de los 5 Clusters: ¿Qué Decisiones Determinan el Rendimiento y Cuáles son Independientes?",
        fontsize=13.8, fontweight="bold", y=1.02,
    )
    fig3_path = out_dir / "fig3_cluster_composition_and_decision_map.png"
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✔ Figura 3 guardada en: {fig3_path}")
    print(f"✔ Reglas del Árbol de Decisión guardadas en: {out_dir / 'stage1_cluster_decision_rules.txt'}")


if __name__ == "__main__":
    main()