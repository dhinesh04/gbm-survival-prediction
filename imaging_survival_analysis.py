"""
imaging_survival_analysis.py
-----------------------------
Secondary imaging analysis on the 51-patient BraTS/TCGA overlap cohort.

Tier 1  — GT Shape (LOSO Cox):
    Elastic-Net Cox trained on 12 ground-truth tumor morphology features
    (shape of necrotic core, oedema, and enhancing tumour from GT segmentation).
    Evaluated with Leave-One-Subject-Out (LOSO) cross-validation.

Tier 2  — Transfer + Imaging (LOSO Cox):
    Adds the SurvGCN Cox risk score (pre-trained on 343 multi-omics patients)
    as a 13th feature. The GCN score acts as a molecular prior; LOSO tests
    whether imaging shape adds information on top of it.

GCN baseline — Cox risk score alone (no LOSO; score is pre-computed).

Usage (called automatically by run.sh)
---------------------------------------
    python3 imaging_survival_analysis.py <plots_dir>

    <plots_dir> is the current run's plots folder, e.g.:
        src/outputs/runs/20260707_120000/plots/

    All outputs are written to <plots_dir>.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

# ── Static paths (relative to project root) ───────────────────────────────────
OVERLAP_EXCEL = "BraTS_TCGA_Radiomics_Overlap.xlsx"
CLINICAL_TXT  = "cbioportal_data/data_clinical_patient.txt"

# ── GT shape feature columns ─────────────────────────────────────────────────
GT_SHAPE_COLS = [
    "GT_Necrotic_original_shape_Elongation",
    "GT_Necrotic_original_shape_Flatness",
    "GT_Necrotic_original_shape_MajorAxisLength",
    "GT_Necrotic_original_shape_VoxelVolume",
    "GT_Edema_original_shape_Elongation",
    "GT_Edema_original_shape_Flatness",
    "GT_Edema_original_shape_MajorAxisLength",
    "GT_Edema_original_shape_VoxelVolume",
    "GT_Enhancing_original_shape_Elongation",
    "GT_Enhancing_original_shape_Flatness",
    "GT_Enhancing_original_shape_MajorAxisLength",
    "GT_Enhancing_original_shape_VoxelVolume",
]


# ─────────────────────────────────────────────────────────────────────────────
# CONCORDANCE INDEX
# ─────────────────────────────────────────────────────────────────────────────
def concordance_index(risk_scores, times, events):
    n = len(times)
    concordant = comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = events[i], events[j]
            if ei == 0 and ej == 0:
                continue
            if ei == 1 and ej == 1:
                if times[i] == times[j]:
                    continue
                comparable += 1
                concordant += (risk_scores[i] > risk_scores[j]) if times[i] < times[j] \
                               else (risk_scores[i] < risk_scores[j])
            elif ei == 1:
                if times[i] >= times[j]:
                    continue
                comparable += 1
                concordant += (risk_scores[i] > risk_scores[j])
            else:
                if times[j] >= times[i]:
                    continue
                comparable += 1
                concordant += (risk_scores[i] < risk_scores[j])
    return concordant / comparable if comparable > 0 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(gcn_scores_csv: str) -> pd.DataFrame:
    """Load and merge 51-patient overlap, clinical survival, and GCN scores."""

    df = pd.read_excel(OVERLAP_EXCEL, sheet_name="Overlap_Data")
    df["TCGA_ID"] = df["TCGA_ID"].str.strip()

    clinical = pd.read_csv(CLINICAL_TXT, sep="\t", comment="#")
    clinical["PATIENT_ID"] = clinical["PATIENT_ID"].str.strip()
    clinical["OS_MONTHS"]  = pd.to_numeric(clinical["OS_MONTHS"], errors="coerce")
    clinical["OS_STATUS"]  = clinical["OS_STATUS"].map(
        lambda x: 1 if "DECEASED" in str(x).upper() else 0)

    df = df.merge(
        clinical[["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]],
        left_on="TCGA_ID", right_on="PATIENT_ID", how="inner",
    )

    gcn = pd.read_csv(gcn_scores_csv)
    gcn["PATIENT_ID"] = gcn["PATIENT_ID"].str.strip()
    df = df.merge(gcn[["PATIENT_ID", "cox_risk_score"]], on="PATIENT_ID", how="inner")

    print(f"  Merged cohort: {len(df)} patients")
    print(f"  Events (deceased): {df['OS_STATUS'].sum()} | "
          f"Censored: {(df['OS_STATUS']==0).sum()}")
    print(f"  OS range: {df['OS_MONTHS'].min():.1f} – {df['OS_MONTHS'].max():.1f} months")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOSO ELASTIC-NET COX
# ─────────────────────────────────────────────────────────────────────────────
def loso_elastic_net_cox(X, times, events, l1_ratio=0.5, label=""):
    """
    Leave-One-Subject-Out CoxNet regression.
    Returns predicted risk scores for all N patients.
    """
    N = len(times)
    risk_scores = np.full(N, np.nan)
    n_failed = 0

    for i in range(N):
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False

        X_tr = X[train_mask]
        t_tr = times[train_mask]
        e_tr = events[train_mask].astype(bool)
        X_te = X[i:i+1]

        scaler      = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        y_tr = Surv.from_arrays(event=e_tr, time=t_tr)

        try:
            cox = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alpha_min_ratio=0.1,
                n_alphas=30,
                max_iter=10_000,
                fit_baseline_model=False,
            )
            cox.fit(X_tr_scaled, y_tr)
            mid_alpha_idx = len(cox.alphas_) // 2
            risk_scores[i] = cox.predict(X_te_scaled, alpha=cox.alphas_[mid_alpha_idx])[0]
        except Exception:
            n_failed += 1
            risk_scores[i] = 0.0

    if n_failed > 0:
        print(f"  [{label}] Warning: {n_failed}/{N} folds fell back to neutral risk")

    return risk_scores


# ─────────────────────────────────────────────────────────────────────────────
# KM PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_km_curves(risk_dict, times, events, output_path):
    n_plots = len(risk_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5), sharey=True)
    if n_plots == 1:
        axes = [axes]

    palette = {"High risk": "#d62728", "Low risk": "#1f77b4"}

    for ax, (label, risk) in zip(axes, risk_dict.items()):
        median_r = np.median(risk)
        high_m   = risk >= median_r
        low_m    = ~high_m

        lr    = logrank_test(times[high_m], times[low_m], events[high_m], events[low_m])
        p_str = f"p = {lr.p_value:.4f}" if lr.p_value >= 0.001 else "p < 0.001"

        for mask, grp_label in [(high_m, "High risk"), (low_m, "Low risk")]:
            kmf = KaplanMeierFitter()
            kmf.fit(times[mask], event_observed=events[mask], label=grp_label)
            kmf.plot_survival_function(ax=ax, ci_show=True,
                                       color=palette[grp_label], linewidth=2)

        ax.set_title(f"{label}\n({p_str})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (months)", fontsize=10)
        ax.set_ylabel("Survival probability", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("KM Curves — Median Risk Split (51-patient BraTS/TCGA Overlap)",
                 fontsize=12, y=1.01, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  KM plot saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SPEARMAN CORRELATION PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(gcn_risk, X_gt_df, output_path):
    rhos, pvals = [], []
    for col in X_gt_df.columns:
        rho, pv = stats.spearmanr(gcn_risk, X_gt_df[col].values)
        rhos.append(rho)
        pvals.append(pv)

    short_names = [c.replace("GT_", "").replace("_original_shape_", "\n")
                   for c in X_gt_df.columns]

    corr_df = pd.DataFrame(
        {"Feature": short_names, "Spearman ρ": rhos, "p-value": pvals}
    ).sort_values("Spearman ρ", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if r > 0 else "#1f77b4" for r in corr_df["Spearman ρ"]]
    bars = ax.barh(corr_df["Feature"], corr_df["Spearman ρ"], color=colors, alpha=0.85)

    for bar, (_, row) in zip(bars, corr_df.iterrows()):
        star = ("***" if row["p-value"] < 0.001 else
                "**"  if row["p-value"] < 0.01  else
                "*"   if row["p-value"] < 0.05  else "")
        if star:
            x = bar.get_width()
            ax.text(x + (0.01 if x >= 0 else -0.01),
                    bar.get_y() + bar.get_height() / 2,
                    star, va="center",
                    ha="left" if x >= 0 else "right", fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Spearman ρ (GCN risk score vs GT shape feature)", fontsize=10)
    ax.set_title("Correlation: Multi-Omics GCN Risk Score vs. Tumor Shape Features\n"
                 "(* p<0.05, ** p<0.01, *** p<0.001)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Correlation heatmap saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 imaging_survival_analysis.py <plots_dir>")
        sys.exit(1)

    plots_dir      = sys.argv[1]
    gcn_scores_csv = os.path.join(plots_dir, "gcn_risk_scores.csv")

    if not os.path.exists(gcn_scores_csv):
        raise FileNotFoundError(
            f"gcn_risk_scores.csv not found in {plots_dir}\n"
            "Ensure main.py saved it there (check the diff in main.py)."
        )

    print("=" * 65)
    print("  Imaging Survival Analysis — 51-patient BraTS/TCGA Cohort")
    print(f"  Plots dir: {plots_dir}")
    print("  Tier 1: GT shape features (LOSO Elastic-Net Cox)")
    print("  Tier 2: GCN risk score + GT shape features (LOSO)")
    print("=" * 65)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n── Loading data ──────────────────────────────────────────────")
    df = load_data(gcn_scores_csv)

    times    = df["OS_MONTHS"].values.astype(float)
    events   = df["OS_STATUS"].values.astype(int)
    gcn_risk = df["cox_risk_score"].values.astype(float)
    X_gt_df  = df[GT_SHAPE_COLS]
    X_gt     = X_gt_df.values.astype(float)

    print(f"\n  GT shape features: {X_gt.shape[1]} columns")
    print(f"  Missing values:    {np.isnan(X_gt).sum()}")

    # ── 2. GCN baseline ───────────────────────────────────────────────────────
    print("\n── GCN Baseline (pre-trained, no LOSO) ───────────────────────")
    c_gcn  = concordance_index(gcn_risk, times, events)
    lr_gcn = logrank_test(
        times[gcn_risk >= np.median(gcn_risk)],
        times[gcn_risk <  np.median(gcn_risk)],
        events[gcn_risk >= np.median(gcn_risk)],
        events[gcn_risk <  np.median(gcn_risk)],
    )
    print(f"  C-index: {c_gcn:.4f}   |   Log-rank p = {lr_gcn.p_value:.4f}")

    # ── 3. Tier 1: GT shape only ──────────────────────────────────────────────
    print("\n── Tier 1: GT Shape Features (LOSO) ─────────────────────────")
    loso_risk_gt = loso_elastic_net_cox(X_gt, times, events, label="GT-shape")
    c_gt  = concordance_index(loso_risk_gt, times, events)
    lr_gt = logrank_test(
        times[loso_risk_gt >= np.median(loso_risk_gt)],
        times[loso_risk_gt <  np.median(loso_risk_gt)],
        events[loso_risk_gt >= np.median(loso_risk_gt)],
        events[loso_risk_gt <  np.median(loso_risk_gt)],
    )
    print(f"  LOSO C-index: {c_gt:.4f}   |   Log-rank p = {lr_gt.p_value:.4f}")

    # ── 4. Tier 2: GCN + GT shape ─────────────────────────────────────────────
    print("\n── Tier 2: GCN + GT Shape (LOSO) ────────────────────────────")
    X_combined = np.column_stack([gcn_risk.reshape(-1, 1), X_gt])
    loso_risk_combined = loso_elastic_net_cox(
        X_combined, times, events, label="GCN+GT-shape")
    c_comb  = concordance_index(loso_risk_combined, times, events)
    lr_comb = logrank_test(
        times[loso_risk_combined >= np.median(loso_risk_combined)],
        times[loso_risk_combined <  np.median(loso_risk_combined)],
        events[loso_risk_combined >= np.median(loso_risk_combined)],
        events[loso_risk_combined <  np.median(loso_risk_combined)],
    )
    print(f"  LOSO C-index: {c_comb:.4f}   |   Log-rank p = {lr_comb.p_value:.4f}")

    # ── 5. Rank MAE ───────────────────────────────────────────────────────────
    deceased_mask = events == 1
    def rank_mae(risk):
        pred_rank   = np.argsort(np.argsort(-risk))
        actual_rank = np.argsort(np.argsort(times))
        return float(np.abs(pred_rank[deceased_mask] - actual_rank[deceased_mask]).mean())

    mae_gcn  = rank_mae(gcn_risk)
    mae_gt   = rank_mae(loso_risk_gt)
    mae_comb = rank_mae(loso_risk_combined)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────────────────")
    summary = pd.DataFrame({
        "Approach": [
            "GCN risk score (pre-trained, 343 patients)",
            "GT shape features only  (LOSO Cox, N=51)",
            "GCN + GT shape features (LOSO Cox, N=51)",
        ],
        "C-index":          [c_gcn,           c_gt,           c_comb],
        "Log-rank p":       [lr_gcn.p_value,  lr_gt.p_value,  lr_comb.p_value],
        "Rank MAE (events)":[mae_gcn,          mae_gt,         mae_comb],
        "Evaluation":       ["Direct (no LOSO)", "LOSO",       "LOSO"],
    })

    col_w = [46, 9, 12, 20, 18]
    header = (f"{'Approach':<{col_w[0]}} {'C-index':>{col_w[1]}} "
              f"{'Log-rank p':>{col_w[2]}} {'Rank MAE (events)':>{col_w[3]}} "
              f"{'Evaluation':<{col_w[4]}}")
    print("\n  " + header)
    print("  " + "-" * sum(col_w))
    for _, row in summary.iterrows():
        print(f"  {row['Approach']:<{col_w[0]}} {row['C-index']:>{col_w[1]}.4f} "
              f"{row['Log-rank p']:>{col_w[2]}.4f} "
              f"{row['Rank MAE (events)']:>{col_w[3]}.1f} "
              f"{row['Evaluation']:<{col_w[4]}}")

    delta = c_comb - c_gcn
    note  = ("GT shape adds prognostic value beyond multi-omics (Δ > 0.01)"
             if delta > 0.01 else
             "GT shape provides marginal additional value (|Δ| ≤ 0.01)"
             if delta > -0.01 else
             "GT shape does not improve over multi-omics risk score (Δ < 0)")
    print(f"\n  Interpretation: {note}")
    print(f"  C-index delta (Tier2 vs GCN baseline): {delta:+.4f}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    summary.to_csv(os.path.join(plots_dir, "imaging_summary_table.csv"), index=False)
    print(f"\n  Summary      → {plots_dir}/imaging_summary_table.csv")

    pd.DataFrame({
        "PATIENT_ID":         df["TCGA_ID"].values,
        "BraTS_ID":           df["CaseID"].values,
        "OS_MONTHS":          times,
        "OS_STATUS":          events,
        "gcn_risk":           gcn_risk,
        "loso_risk_gt_shape": loso_risk_gt,
        "loso_risk_combined": loso_risk_combined,
    }).to_csv(os.path.join(plots_dir, "imaging_analysis_results.csv"), index=False)
    print(f"  Predictions  → {plots_dir}/imaging_analysis_results.csv")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n── Generating plots ──────────────────────────────────────────")
    plot_km_curves(
        {
            "GCN Baseline\n(Multi-omics)":   gcn_risk,
            "Tier 1\n(GT Shape LOSO)":       loso_risk_gt,
            "Tier 2\n(GCN + GT Shape LOSO)": loso_risk_combined,
        },
        times, events,
        os.path.join(plots_dir, "imaging_km_plot.png"),
    )

    plot_correlation_heatmap(
        gcn_risk, X_gt_df,
        os.path.join(plots_dir, "imaging_correlation_heatmap.png"),
    )

    print("\n  All done.")


if __name__ == "__main__":
    main()