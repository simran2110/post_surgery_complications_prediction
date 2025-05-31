import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, classification_report, roc_curve
)


def compute_ci(metric_values, ci: float = 0.95) -> Tuple[float, float]:
    lower = np.percentile(metric_values, (1 - ci) / 2 * 100)
    upper = np.percentile(metric_values, (1 + ci) / 2 * 100)
    return lower, upper


def bootstrap_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray,
                      y_pred_proba: np.ndarray,
                      n_iterations: int = 1000,
                      ci: float = 0.95) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(42)
    n = len(y_true)

    accs, precs, recalls, f1s, roc_aucs, pr_aucs = [], [], [], [], [], []

    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        y_pp = y_pred_proba[idx]

        accs.append(accuracy_score(y_t, y_p))
        precs.append(precision_score(y_t, y_p, zero_division=0))
        recalls.append(recall_score(y_t, y_p))
        f1s.append(f1_score(y_t, y_p))
        roc_aucs.append(roc_auc_score(y_t, y_pp))
        pr = precision_recall_curve(y_t, y_pp)
        pr_aucs.append(auc(pr[1], pr[0]))

    return {
        'accuracy_ci': compute_ci(accs, ci),
        'precision_ci': compute_ci(precs, ci),
        'recall_ci': compute_ci(recalls, ci),
        'f1_ci': compute_ci(f1s, ci),
        'roc_auc_ci': compute_ci(roc_aucs, ci),
        'pr_auc_ci': compute_ci(pr_aucs, ci)
    }


def evaluate_model_with_ci(model: Any,
                            X: pd.DataFrame,
                            y: pd.Series,
                            ci: float = 0.95,
                            n_iterations: int = 1000) -> Dict[str, Any]:
    """
    Evaluates a classification model with metrics and confidence intervals.

    Parameters:
    ------------
    model: fitted classification model with predict and predict_proba methods
    X: Test feature data
    y: True labels
    ci: Confidence interval (e.g., 0.95 for 95%)
    n_iterations: Number of bootstrap samples

    Returns:
    --------
    Dictionary with metrics and their confidence intervals
    """
    y_pred = model.predict(X)
    y_pred_proba_raw = model.predict_proba(X)
    
    if y_pred_proba_raw.ndim == 1:
        y_pred_proba = y_pred_proba_raw
    else:
        y_pred_proba = y_pred_proba_raw[:, 1]

    # Base metrics
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    precision_vals, recall_vals, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

    report = classification_report(y, y_pred, output_dict=True)

    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

    # Confidence Intervals
    ci_results = bootstrap_metrics(np.array(y), np.array(y_pred), np.array(y_pred_proba),
                                   n_iterations=n_iterations, ci=ci)
    metrics.update(ci_results)

    return y_pred, y_pred_proba, metrics