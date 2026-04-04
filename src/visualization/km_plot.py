"""
km_plot.py
----------
Kaplan-Meier survival curves from GCN Cox head risk scores.

Three plots:
  1. Median risk split — high vs low risk groups (KM + log-rank)
  2. Tertile risk split — high / medium / low (3 groups)
  3. LTS label overlay — compare predicted risk groups against true
     LTS/non-LTS labels to assess clinical alignment
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from src.utils import significance_stars


def plot_km_median_split(risk_scores: np.ndarray,
                         times: np.ndarray,
                         events: np.ndarray,
                         save_path: str = "km_median_split.png"):
    """
    Split patients at median risk score into high/low risk groups.
    Plot KM curves for each group with log-rank p-value.

    Parameters
    ----------
    risk_scores : (N,) Cox head risk scores (higher = more risk)
    times       : (N,) survival times in months
    events      : (N,) 1=deceased, 0=censored
    save_path   : output filename
    """
    median_risk = np.median(risk_scores)
    high_mask   = risk_scores >= median_risk
    low_mask    = ~high_mask

    # Log-rank test
    lr = logrank_test(
        times[high_mask],   times[low_mask],
        events[high_mask],  events[low_mask]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    kmf = KaplanMeierFitter()

    kmf.fit(times[high_mask], events[high_mask], label=f"High risk (n={high_mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#d62728")

    kmf.fit(times[low_mask], events[low_mask], label=f"Low risk (n={low_mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#1f77b4")

    ax.set_xlabel("Time (months)", fontsize=13)
    ax.set_ylabel("Survival probability", fontsize=13)
    ax.set_title("Kaplan-Meier Curves — Median Risk Split\n"
                 f"Log-rank p = {lr.p_value:.4f}", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Annotate p-value
    sig = significance_stars(lr.p_value)
    ax.text(0.98, 0.98, f"p = {lr.p_value:.4f} {sig}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat',
                                   alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}  (log-rank p={lr.p_value:.4f} {sig})")
    return lr.p_value


def plot_km_tertile_split(risk_scores: np.ndarray,
                          times: np.ndarray,
                          events: np.ndarray,
                          save_path: str = "km_tertile_split.png"):
    """
    Split patients into tertiles (low / medium / high risk).
    Multivariate log-rank test across three groups.
    """
    t33, t67 = np.percentile(risk_scores, [33, 67])
    low_mask  = risk_scores < t33
    mid_mask  = (risk_scores >= t33) & (risk_scores < t67)
    high_mask = risk_scores >= t67

    # Multivariate log-rank
    groups = np.where(low_mask, 0, np.where(mid_mask, 1, 2))
    mlr    = multivariate_logrank_test(times, groups, events)

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    labels = [f"Low risk (n={low_mask.sum()})",
              f"Med risk (n={mid_mask.sum()})",
              f"High risk (n={high_mask.sum()})"]
    masks  = [low_mask, mid_mask, high_mask]

    for mask, color, label in zip(masks, colors, labels):
        if mask.sum() > 0:
            kmf.fit(times[mask], events[mask], label=label)
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Time (months)", fontsize=13)
    ax.set_ylabel("Survival probability", fontsize=13)
    ax.set_title("Kaplan-Meier Curves — Tertile Risk Split\n"
                 f"Multivariate log-rank p = {mlr.p_value:.4f}", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    sig = significance_stars(mlr.p_value)
    ax.text(0.98, 0.98, f"p = {mlr.p_value:.4f} {sig}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat',
                                   alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}  (multivariate p={mlr.p_value:.4f} {sig})")
    return mlr.p_value


def plot_km_true_labels(times: np.ndarray,
                        events: np.ndarray,
                        lts_labels: np.ndarray,
                        save_path: str = "km_true_labels.png"):
    """
    KM curves using the ground-truth LTS labels.
    This is the clinical reference curve — shows the actual
    survival difference between LTS and non-LTS patients.
    Compare this against the median-split plot to assess how
    well risk score stratification aligns with true labels.

    Parameters
    ----------
    lts_labels : (N,) 1=LTS, 0=non-LTS
    """
    lr = logrank_test(
        times[lts_labels == 1],   times[lts_labels == 0],
        events[lts_labels == 1],  events[lts_labels == 0]
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    kmf.fit(times[lts_labels == 1], events[lts_labels == 1],
            label=f"LTS (n={int((lts_labels==1).sum())})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#2ca02c")

    kmf.fit(times[lts_labels == 0], events[lts_labels == 0],
            label=f"non-LTS (n={int((lts_labels==0).sum())})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color="#d62728")

    ax.set_xlabel("Time (months)", fontsize=13)
    ax.set_ylabel("Survival probability", fontsize=13)
    ax.set_title("Kaplan-Meier Curves — True LTS Labels (Reference)\n"
                 f"Log-rank p = {lr.p_value:.4f}", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    sig = significance_stars(lr.p_value)
    ax.text(0.98, 0.98, f"p = {lr.p_value:.4f} {sig}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat',
                                   alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}  (log-rank p={lr.p_value:.4f} {sig})")
    return lr.p_value


def generate_all_km_plots(gcn_results: dict,
                          output_dir: str = "."):
    """
    Generate all three KM plots from GCN results dict.

    Parameters
    ----------
    gcn_results : dict from train_gcn() — must contain:
        risk_scores, times_test, events_test, y_test (as numpy array)
    output_dir  : directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    risk   = gcn_results["risk_scores"]
    times  = gcn_results["times_test"]
    events = gcn_results["events_test"]

    # y_test may be pd.Series or np.ndarray
    y_test = gcn_results.get("y_test_np")
    if y_test is None:
        # fall back — reconstruct from confusion matrix not ideal
        # caller should pass y_test_np explicitly
        raise KeyError("gcn_results must contain 'y_test_np' "
                       "(numpy array of true LTS labels)")

    print("\n── Generating KM Plots ──────────────────────────────────────")
    print(f"  Test patients: {len(risk)} | "
          f"Events: {int(events.sum())} | "
          f"Censored: {int((events==0).sum())}")
    print(f"  C-index: {gcn_results['cindex']:.4f}")

    p1 = plot_km_median_split(
        risk, times, events,
        save_path=f"{output_dir}/km_median_split.png")

    p2 = plot_km_tertile_split(
        risk, times, events,
        save_path=f"{output_dir}/km_tertile_split.png")

    p3 = plot_km_true_labels(
        times, events, y_test,
        save_path=f"{output_dir}/km_true_labels.png")

    print(f"\n  Summary:")
    print(f"    Median split log-rank p:     {p1:.4f}")
    print(f"    Tertile split log-rank p:    {p2:.4f}")
    print(f"    True label log-rank p:       {p3:.4f}")
    print("────────────────────────────────────────────────────────────")

    return {"p_median": p1, "p_tertile": p2, "p_true_labels": p3}