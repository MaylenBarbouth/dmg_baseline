# 12_test_estadistico_logreg_vs_lgbm.py
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def nadeau_bengio_paired(diff: np.ndarray, n_test: float, n_train: float) -> Tuple[float, float, float]:
    """Devuelve (media_diff, ic95_half_width, p_valor_NB) para las 25 particiones de CV 5x5."""
    n = len(diff)
    mean_d = float(np.mean(diff))
    var_d = float(np.var(diff, ddof=1))
    if var_d <= 1e-18:
        return mean_d, 0.0, 1.0
    se_nb = float(np.sqrt(((1.0 / n) + (n_test / n_train)) * var_d))
    t_stat = mean_d / se_nb
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
    ci95_hw = float(stats.t.ppf(0.975, df=n - 1) * se_nb)
    return mean_d, ci95_hw, p_val


def main() -> None:
    with open("config/experiment_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cons_dir = Path(config["paths"]["outputs_dir"]) / "consolidated"
    out_dir = Path(config["paths"]["outputs_dir"]) / "analysis" / "prioridad_2_3_4"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_m = pd.read_csv(cons_dir / "master_fold_metrics.csv")
    df_p = pd.read_parquet(cons_dir / "master_oof_predictions.parquet")

    # 1. CV 5x5 (Sin pesos): Test pareado de Nadeau-Bengio LogReg vs LightGBM por cada Cohorte y Set
    cv_none = df_m[(df_m["val_scheme"] == "repeated_group_cv") & (df_m["class_weight_mode"] == "none")]
    cv_rows = []
    for coh in ["A_all", "B_no_antc"]:
        for fset in ["set_1_clinico", "set_2a_lab_sin_glu", "set_2b_lab_completo", "set_3_full"]:
            sub = cv_none[(cv_none["cohort"] == coh) & (cv_none["feature_set"] == fset)]
            lr = sub[sub["model_type"] == "logreg"].set_index(["repeat_idx", "fold_idx"])
            lg = sub[sub["model_type"] == "lgbm"].set_index(["repeat_idx", "fold_idx"])
            idx = lr.index.intersection(lg.index)
            lr, lg = lr.loc[idx], lg.loc[idx]

            n_te, n_tr = float(lr["n_test_rows"].mean()), float(lr["n_train_rows"].mean())
            d_pr = (lr["test_pr_auc"] - lg["test_pr_auc"]).to_numpy()
            d_roc = (lr["test_roc_auc"] - lg["test_roc_auc"]).to_numpy()
            d_br = (lr["test_brier_score"] - lg["test_brier_score"]).to_numpy()

            m_pr, ci_pr, p_pr = nadeau_bengio_paired(d_pr, n_te, n_tr)
            m_roc, ci_roc, p_roc = nadeau_bengio_paired(d_roc, n_te, n_tr)
            m_br, ci_br, p_br = nadeau_bengio_paired(d_br, n_te, n_tr)

            cv_rows.append({
                "Cohorte": coh,
                "Set": fset,
                "PR_LogReg_vs_LGBM": f"{lr['test_pr_auc'].mean():.3f} vs {lg['test_pr_auc'].mean():.3f}",
                "Delta_PR (±IC95%)": f"{m_pr:+.4f} (±{ci_pr:.4f})",
                "p_NB_PR": p_pr,
                "Folds_Gana_LR_PR": f"{int(np.sum(d_pr > 0))}/25",
                "ROC_LogReg_vs_LGBM": f"{lr['test_roc_auc'].mean():.3f} vs {lg['test_roc_auc'].mean():.3f}",
                "Delta_ROC (±IC95%)": f"{m_roc:+.4f} (±{ci_roc:.4f})",
                "p_NB_ROC": p_roc,
                "Folds_Gana_LR_ROC": f"{int(np.sum(d_roc > 0))}/25",
                "p_NB_Brier": p_br,
            })

    df_cv_res = pd.DataFrame(cv_rows)
    df_cv_res.to_csv(out_dir / "test_pareado_logreg_vs_lgbm_cv.csv", index=False)

    # 2. Test Temporal OOT (Sin pesos): Bootstrap pareado a nivel paciente (2000 réplicas) LogReg vs LightGBM
    oot_none = df_p[(df_p["val_scheme"] == "temporal_oot") & (df_p["class_weight_mode"] == "none")]
    agg_oot = oot_none.groupby(["cohort", "feature_set", "model_type", "id_embarazo"], as_index=False).agg(
        y_true=("y_true", "first"), y_prob=("y_prob_pred", "mean")
    )

    rng = np.random.default_rng(42)
    n_boot = 2000
    oot_rows = []
    for coh in ["A_all", "B_no_antc"]:
        for fset in ["set_1_clinico", "set_2a_lab_sin_glu", "set_2b_lab_completo", "set_3_full"]:
            piv = agg_oot[(agg_oot["cohort"] == coh) & (agg_oot["feature_set"] == fset)].pivot(
                index=["id_embarazo", "y_true"], columns="model_type", values="y_prob"
            ).dropna().reset_index()

            y = piv["y_true"].to_numpy(dtype=int)
            p_lr = piv["logreg"].to_numpy(dtype=float)
            p_lg = piv["lgbm"].to_numpy(dtype=float)

            pos_i, neg_i = np.where(y == 1)[0], np.where(y == 0)[0]
            d_pr_b, d_roc_b = np.empty(n_boot), np.empty(n_boot)
            for b in range(n_boot):
                idx_b = np.concatenate([
                    rng.choice(pos_i, len(pos_i), replace=True),
                    rng.choice(neg_i, len(neg_i), replace=True),
                ])
                d_pr_b[b] = average_precision_score(y[idx_b], p_lr[idx_b]) - average_precision_score(y[idx_b], p_lg[idx_b])
                d_roc_b[b] = roc_auc_score(y[idx_b], p_lr[idx_b]) - roc_auc_score(y[idx_b], p_lg[idx_b])

            p_boot_pr = float(2.0 * min(np.mean(d_pr_b <= 0), np.mean(d_pr_b >= 0)))
            p_boot_roc = float(2.0 * min(np.mean(d_roc_b <= 0), np.mean(d_roc_b >= 0)))

            oot_rows.append({
                "Cohorte": coh,
                "Set": fset,
                "Delta_PR_OOT [IC95%]": f"{np.mean(d_pr_b):+.4f} [{np.percentile(d_pr_b, 2.5):+.4f}, {np.percentile(d_pr_b, 97.5):+.4f}]",
                "p_Boot_PR": p_boot_pr,
                "Delta_ROC_OOT [IC95%]": f"{np.mean(d_roc_b):+.4f} [{np.percentile(d_roc_b, 2.5):+.4f}, {np.percentile(d_roc_b, 97.5):+.4f}]",
                "p_Boot_ROC": p_boot_roc,
            })

    df_oot_res = pd.DataFrame(oot_rows)
    df_oot_res.to_csv(out_dir / "test_bootstrap_logreg_vs_lgbm_oot.csv", index=False)

    print("\n=== 1. DUELO PAREADO LOGREG VS. LIGHTGBM EN CV 5x5 (NADEAU-BENGIO) ===")
    print(df_cv_res.to_string(index=False))
    print("\n=== 2. DUELO PAREADO LOGREG VS. LIGHTGBM EN TEST TEMPORAL OOT (BOOTSTRAP 2000 RÉPLICAS) ===")
    print(df_oot_res.to_string(index=False))


if __name__ == "__main__":
    main()