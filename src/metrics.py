# src/metrics.py
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def find_optimal_thresholds_on_validation(
    y_val_true: np.ndarray, y_val_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Calcula dos umbrales óptimos usando EXCLUSIVAMENTE las predicciones OOF de validación interna:
    1. Umbral que maximiza el Índice de Youden (Sensibilidad + Especificidad - 1).
    2. Umbral de Screening que garantiza al menos 80% de Sensibilidad maximizando Especificidad.
    """
    fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
    # Evitar el primer umbral infinito de roc_curve
    valid_mask = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[valid_mask], tpr[valid_mask], thresholds[valid_mask]

    if len(thresholds) == 0:
        return 0.5, 0.5

    specificity = 1.0 - fpr
    youden_j = tpr + specificity - 1.0
    best_youden_idx = int(np.argmax(youden_j))
    thresh_youden = float(thresholds[best_youden_idx])

    # Umbral para sensibilidad >= 0.80
    sens80_candidates = np.where(tpr >= 0.80)[0]
    if len(sens80_candidates) > 0:
        # Entre los que cumplen sens >= 0.80, elegir el de mayor especificidad
        best_s80_subidx = sens80_candidates[
            int(np.argmax(specificity[sens80_candidates]))
        ]
        thresh_sens80 = float(thresholds[best_s80_subidx])
    else:
        thresh_sens80 = thresh_youden

    return thresh_youden, thresh_sens80


def compute_threshold_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, prefix: str
) -> Dict[str, float]:
    """Calcula matriz de confusión, Sensibilidad, Especificidad, VPP, VPN y F1 para un umbral dado."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        f"threshold_{prefix}": float(threshold),
        f"tp_{prefix}": int(tp),
        f"fp_{prefix}": int(fp),
        f"tn_{prefix}": int(tn),
        f"fn_{prefix}": int(fn),
        f"sens_{prefix}": float(sens),
        f"spec_{prefix}": float(spec),
        f"ppv_{prefix}": float(ppv),
        f"npv_{prefix}": float(npv),
        f"f1_{prefix}": float(f1),
    }


def evaluate_predictions(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    thresh_youden: float,
    thresh_sens80: float,
) -> Dict[str, float]:
    """Evalúa métricas libres de umbral y métricas en 3 puntos de corte (0.50, Youden, Sens80)."""
    y_prob_clipped = np.clip(y_prob, 1e-15, 1.0 - 1e-15)

    metrics = {
        "test_pr_auc": float(average_precision_score(y_test, y_prob)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "test_brier_score": float(brier_score_loss(y_test, y_prob)),
        "test_log_loss": float(log_loss(y_test, y_prob_clipped)),
    }

    metrics.update(compute_threshold_metrics(y_test, y_prob, 0.50, "050"))
    metrics.update(
        compute_threshold_metrics(y_test, y_prob, thresh_youden, "youden")
    )
    metrics.update(
        compute_threshold_metrics(y_test, y_prob, thresh_sens80, "sens80")
    )

    return metrics