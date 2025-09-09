
"""
stats_tests.py
--------------
Statistical tests utilities: McNemar's test, paired t-test, Wilcoxon signed-rank test.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

from scipy.stats import ttest_rel, wilcoxon


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> float:
    """
    Returns p-value of McNemar's test on paired classification decisions.
    """
    if not HAVE_STATSMODELS:
        # Fallback: approximate via binomial on discordant pairs
        a_wrong = (y_true != y_pred_a)
        b_wrong = (y_true != y_pred_b)
        b01 = np.sum((a_wrong == 0) & (b_wrong == 1))
        b10 = np.sum((a_wrong == 1) & (b_wrong == 0))
        # Binomial test (two-sided) approximation
        n = b01 + b10
        if n == 0:
            return 1.0
        # simple symmetric probability
        from math import comb
        p = sum(comb(n, k) for k in range(0, min(b01, b10) + 1)) / (2 ** (n - 1))
        return float(min(1.0, 2 * p))
    # Exact mcnemar if available
    table = np.zeros((2, 2), dtype=int)
    table[0, 0] = np.sum((y_pred_a == y_true) & (y_pred_b == y_true))
    table[0, 1] = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    table[1, 0] = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    table[1, 1] = np.sum((y_pred_a != y_true) & (y_pred_b != y_true))
    res = mcnemar(table, exact=True)
    return float(res.pvalue)


def paired_t_and_wilcoxon(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
    t_p = float(ttest_rel(scores_a, scores_b).pvalue)
    try:
        w_p = float(wilcoxon(scores_a, scores_b).pvalue)
    except ValueError:
        w_p = 1.0  # fallback when all differences are zero
    return {"t_p": t_p, "w_p": w_p}
