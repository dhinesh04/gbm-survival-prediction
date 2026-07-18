"""Kaplan-Meier survival curves from GCN Cox head risk scores."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from src.utils import significance_stars


def _render_km_groups(groups, title, p_value, save_path):
    """groups: list of (times, events, label, color). Draws KM curves for each
    group plus a p-value annotation, then saves to save_path."""
    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    for times_g, events_g, label, color in groups:
        if len(times_g) == 0:
            continue
        kmf.fit(times_g, events_g, label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    ax.set_xlabel("Time (months)", fontsize=13)
    ax.set_ylabel("Survival probability", fontsize=13)
    ax.set_title(f"{title}\np = {p_value:.4f}", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    sig = significance_stars(p_value)
    ax.text(0.98, 0.98, f"p = {p_value:.4f} {sig}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}  (p={p_value:.4f} {sig})")


def plot_km_median_split(risk_scores: np.ndarray,
                         times: np.ndarray,
                         events: np.ndarray,
                         save_path: str = "km_median_split.png"):
    """Split patients at median risk score, plot KM curves with log-rank p-value."""
    median_risk = np.median(risk_scores)
    high_mask   = risk_scores >= median_risk
    low_mask    = ~high_mask

    lr = logrank_test(
        times[high_mask],   times[low_mask],
        events[high_mask],  events[low_mask]
    )

    groups = [
        (times[high_mask], events[high_mask], f"High risk (n={high_mask.sum()})", "#d62728"),
        (times[low_mask],  events[low_mask],  f"Low risk (n={low_mask.sum()})",   "#1f77b4"),
    ]
    _render_km_groups(groups, "Kaplan-Meier Curves — Median Risk Split", lr.p_value, save_path)
    return lr.p_value


def plot_km_tertile_split(risk_scores: np.ndarray,
                          times: np.ndarray,
                          events: np.ndarray,
                          save_path: str = "km_tertile_split.png"):
    """Split patients into risk tertiles, multivariate log-rank test across the three groups."""
    t33, t67  = np.percentile(risk_scores, [33, 67])
    low_mask  = risk_scores < t33
    mid_mask  = (risk_scores >= t33) & (risk_scores < t67)
    high_mask = risk_scores >= t67

    group_ids = np.where(low_mask, 0, np.where(mid_mask, 1, 2))
    mlr = multivariate_logrank_test(times, group_ids, events)

    groups = [
        (times[low_mask],  events[low_mask],  f"Low risk (n={low_mask.sum()})",  "#1f77b4"),
        (times[mid_mask],  events[mid_mask],  f"Med risk (n={mid_mask.sum()})",  "#ff7f0e"),
        (times[high_mask], events[high_mask], f"High risk (n={high_mask.sum()})", "#d62728"),
    ]
    _render_km_groups(groups, "Kaplan-Meier Curves — Tertile Risk Split", mlr.p_value, save_path)
    return mlr.p_value


def plot_km_true_labels(times: np.ndarray,
                        events: np.ndarray,
                        lts_labels: np.ndarray,
                        save_path: str = "km_true_labels.png"):
    """KM curves using ground-truth LTS labels -- the clinical reference curve,
    for comparison against the risk-score-split plots."""
    lr = logrank_test(
        times[lts_labels == 1],   times[lts_labels == 0],
        events[lts_labels == 1],  events[lts_labels == 0]
    )

    groups = [
        (times[lts_labels == 1], events[lts_labels == 1], f"LTS (n={int((lts_labels==1).sum())})",     "#2ca02c"),
        (times[lts_labels == 0], events[lts_labels == 0], f"non-LTS (n={int((lts_labels==0).sum())})", "#d62728"),
    ]
    _render_km_groups(groups, "Kaplan-Meier Curves — True LTS Labels (Reference)", lr.p_value, save_path)
    return lr.p_value


def generate_all_km_plots(gcn_results: dict,
                          output_dir: str = "."):
    """Generate the median-split and tertile-split KM plots from a train_gcn() result dict."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    risk   = gcn_results["risk_scores"]
    times  = gcn_results["times_test"]
    events = gcn_results["events_test"]

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

    print(f"\n  Summary:")
    print(f"    Median split log-rank p:     {p1:.4f}")
    print(f"    Tertile split log-rank p:    {p2:.4f}")
    print("────────────────────────────────────────────────────────────")

    return {"p_median": p1, "p_tertile": p2}
