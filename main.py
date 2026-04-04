"""
main.py
-------
Full pipeline:
  1. Stratified train/test split (preserving LTS ratio)
  2. mRMR feature selection per modality (fit on train only)
  3. Build PSN via SNF on real training patients
  4. PSN diagnostics (hubness + label smoothness z-score)
  5. GCN training with binary + Cox dual head
  6. KM plot generation from Cox risk scores

All hyperparameters are defined in config.py.

Usage
-----
Standalone (uses DEFAULT_DATA_DIR from config):
    python main.py

Called from run_experiments.py with a specific threshold:
    from main import main, load_data, split_data
    results = main(data_dir="data_12m", plots_dir="plots_12m")
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import snf

from config import (
    TEST_SIZE, RANDOM_STATE,
    K_MRMR, K_SNF, MU_SNF, N_ITER_SNF,
    N_PERMUTATIONS, ALPHA_SURVIVAL,
    DEFAULT_DATA_DIR, DEFAULT_PLOTS_DIR,
)
from src.data.feature_selection_mrmr import run_mrmr_all_modalities
from src.models.gcn_train import train_gcn
from src.visualization.km_plot import generate_all_km_plots
from src.graph.survival_aware_psn import build_survival_aware_psn, compare_psn_diagnostics
from src.analysis.baseline_comparison import run_baseline_comparison
from src.analysis.ablation_studies import run_ablation
from src.analysis.driver_gene_analysis import run_driver_gene_analysis


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_dir: str = DEFAULT_DATA_DIR,
              use_mutations: bool = False):
    """
    Load pre-processed data files saved by data_processing.py.

    Returns
    -------
    X_cna, X_mrna, X_meth, X_mut, X_clin : pd.DataFrame
    y_all, os_months, os_status           : pd.Series
    """

    X_cna  = pd.read_csv(f"{data_dir}/cna_data.csv",          index_col=0)
    X_mrna = pd.read_csv(f"{data_dir}/mrna_data.csv",         index_col=0)
    X_meth = pd.read_csv(f"{data_dir}/methylation_data.csv",  index_col=0)
    X_mut  = pd.read_csv(f"{data_dir}/mutation_data.csv",     index_col=0)
    X_clin = pd.read_csv(f"{data_dir}/clinical_data.csv",     index_col=0)
    labels  = pd.read_csv(f"{data_dir}/labels.csv",           index_col=0)

    y_all     = labels["LTS"].astype(int)
    os_months = labels["OS_MONTHS"].astype(float)
    os_status = labels["OS_STATUS"].astype(int)

    # Align all modalities to common patients
    common = (
        X_cna.index
        .intersection(X_mrna.index)
        .intersection(X_meth.index)
        .intersection(X_clin.index)
        .intersection(y_all.index)
    )

    if use_mutations and X_mut is not None:
        common = common.intersection(X_mut.index)

    if use_mutations and X_mut is not None:
        X_cna, X_mrna, X_meth, X_mut, X_clin = (
            X_cna.loc[common], X_mrna.loc[common],
            X_meth.loc[common], X_mut.loc[common], X_clin.loc[common]
        )
    else:
        X_cna, X_mrna, X_meth, X_clin = (
            X_cna.loc[common], X_mrna.loc[common],
            X_meth.loc[common], X_clin.loc[common]
        )

    y_all     = y_all.loc[common]
    os_months = os_months.loc[common]
    os_status = os_status.loc[common]

    print(f"\nAfter aligning common patients: {len(common)}")
    print(f"  LTS={int(y_all.sum())}, non-LTS={int((y_all==0).sum())}")
    print(f"  Deceased={int(os_status.sum())}, "
          f"Censored={int((os_status==0).sum())}")
    print(f"  OS_MONTHS range: [{os_months.min():.1f}, "
          f"{os_months.max():.1f}] months")

    return X_cna, X_mrna, X_meth, X_mut, X_clin, y_all, os_months, os_status


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — STRATIFIED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def split_data(X_cna, X_mrna, X_meth, X_mut, X_clin,
               y_all, os_months, os_status):
    """
    Stratified 70/30 split on patient indices (TEST_SIZE set in config.py).
    Same indices applied to all modalities + survival data.
    """
    print("\n── Stratified Train/Test Split ─────────────────────────────")

    patients = y_all.index
    idx_train, idx_test = train_test_split(
        patients,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all
    )

    X_cna_tr,  X_cna_te  = X_cna.loc[idx_train],  X_cna.loc[idx_test]
    X_mrna_tr, X_mrna_te = X_mrna.loc[idx_train], X_mrna.loc[idx_test]
    X_meth_tr, X_meth_te = X_meth.loc[idx_train], X_meth.loc[idx_test]
    X_mut_tr,  X_mut_te  = (
        (X_mut.loc[idx_train], X_mut.loc[idx_test])
        if X_mut is not None else (None, None)
    )
    X_clin_tr, X_clin_te = X_clin.loc[idx_train].copy(), X_clin.loc[idx_test].copy()
    y_train,   y_test    = y_all.loc[idx_train],   y_all.loc[idx_test]

    os_months_train = os_months.loc[idx_train]
    os_months_test  = os_months.loc[idx_test]
    os_status_train = os_status.loc[idx_train]
    os_status_test  = os_status.loc[idx_test]

    print(f"  Train: {len(idx_train)} patients "
          f"(LTS={y_train.sum()}, non-LTS={(y_train==0).sum()} | "
          f"deceased={os_status_train.sum()}, "
          f"censored={(os_status_train==0).sum()})")
    print(f"  Test:  {len(idx_test)} patients  "
          f"(LTS={y_test.sum()},  non-LTS={(y_test==0).sum()} | "
          f"deceased={os_status_test.sum()}, "
          f"censored={(os_status_test==0).sum()})")
    print(f"  Clinical features: {list(X_clin_tr.columns)}")
    print("────────────────────────────────────────────────────────────\n")

    return (X_cna_tr,  X_cna_te,
            X_mrna_tr, X_mrna_te,
            X_meth_tr, X_meth_te,
            X_mut_tr,  X_mut_te,
            X_clin_tr, X_clin_te,
            y_train,   y_test,
            os_months_train, os_months_test,
            os_status_train, os_status_test)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD PSN VIA SNF
# ─────────────────────────────────────────────────────────────────────────────
def build_psn(cna, mrna, meth, mut=None):
    """
    Build PSN from four mRMR-reduced omics matrices using SNF.
    Clinical features excluded from PSN — used as node attributes only.
    Mutation matrix included: SNF computes patient similarity from
    shared binary mutation profiles alongside continuous CNA/mRNA/methylation.
    """
    print("\n── Building PSN via SNF ────────────────────────────────────")
    W_cna  = snf.make_affinity(cna,  K=K_SNF, mu=MU_SNF)
    W_mrna = snf.make_affinity(mrna, K=K_SNF, mu=MU_SNF)
    W_meth = snf.make_affinity(meth, K=K_SNF, mu=MU_SNF)
    affinity_list = [W_cna, W_mrna, W_meth]
    
    if mut is not None:
        W_mut = snf.make_affinity(mut, K=K_SNF, mu=MU_SNF)
        affinity_list.append(W_mut)

    psn = snf.snf(affinity_list, K=K_SNF, t=N_ITER_SNF)
    print(f"  PSN shape: {psn.shape}  ({len(affinity_list)}, {N_ITER_SNF} SNF iterations)")
    print("────────────────────────────────────────────────────────────\n")
    return psn


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PSN DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
def psn_diagnostics(psn, y_labels, top_k=20):
    """
    Compute hubness (max in-degree) and label smoothness (z-score vs null).
    """
    print("\n── PSN Diagnostics ─────────────────────────────────────────")

    n = psn.shape[0]
    psn_nd = psn.copy()
    np.fill_diagonal(psn_nd, 0)

    is_symmetric  = bool(np.allclose(psn_nd, psn_nd.T, atol=1e-6))
    is_nonneg     = bool((psn_nd >= 0).all())
    indegrees     = (psn_nd > 0).sum(axis=0)
    max_indegree  = int(indegrees.max())

    print(f"  Symmetric:      {is_symmetric}")
    print(f"  Non-negative:   {is_nonneg}")
    print(f"  Max in-degree:  {max_indegree}  (hubness; ideal < {n // 5})")

    # Label smoothness — neighbour class agreement
    topk_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        top_idx = np.argsort(psn_nd[i])[::-1][:top_k]
        topk_mask[i, top_idx] = True

    y_arr = np.array(y_labels)
    agreements = []
    for i in range(n):
        neighbours = np.where(topk_mask[i])[0]
        if len(neighbours):
            agreements.append((y_arr[neighbours] == y_arr[i]).mean())
    observed_agreement = float(np.mean(agreements))

    # Permutation null distribution
    rng = np.random.default_rng(RANDOM_STATE)
    perm_agreements = []
    for _ in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y_arr)
        perm_ag = []
        for i in range(n):
            neighbours = np.where(topk_mask[i])[0]
            if len(neighbours):
                perm_ag.append((y_perm[neighbours] == y_perm[i]).mean())
        perm_agreements.append(np.mean(perm_ag))

    perm_mean = float(np.mean(perm_agreements))
    perm_std  = float(np.std(perm_agreements))
    z_score   = (observed_agreement - perm_mean) / (perm_std + 1e-10)

    print(f"  Neighbour agreement (top-K={top_k}): {observed_agreement:.4f}")
    print(f"  Permutation null (mean±std):  {perm_mean:.4f} ± {perm_std:.4f}")
    print(f"  Label smoothness z-score:     {z_score:.3f}  "
          f"[z > 2.0 = survival signal]")

    if   z_score > 2.0: print(f"  ✓ PSN has survival-relevant structure (z={z_score:.2f})")
    elif z_score > 1.0: print(f"  ~ PSN has weak survival signal (z={z_score:.2f})")
    else:               print(f"  ✗ PSN lacks survival signal (z={z_score:.2f})")

    print("────────────────────────────────────────────────────────────\n")

    return {
        "symmetric":          is_symmetric,
        "nonnegative":        is_nonneg,
        "max_indegree":       max_indegree,
        "neighbor_agreement": observed_agreement,
        "perm_mean":          perm_mean,
        "perm_std":           perm_std,
        "z_score":            z_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(data_dir: str = DEFAULT_DATA_DIR,
         plots_dir: str = DEFAULT_PLOTS_DIR,
         use_mutations: bool = False):
    """
    Run the full preprocessing + GCN training pipeline.

    Parameters
    ----------
    data_dir  : folder containing processed CSVs (output of data_processing.py)
    plots_dir : folder where KM and baseline plots are saved

    Returns
    -------
    dict with pipeline artefacts and gcn_results merged.
    """
    print("=" * 62)
    print("  GBM Survival Prediction — Preprocessing Pipeline")
    print(f"  Data dir:  {data_dir}")
    print(f"  Plots dir: {plots_dir}")
    print("=" * 62)


    # ── 0. Load ──────────────────────────────────────────────────
    (X_cna, X_mrna, X_meth, X_mut,
     X_clin, y_all,
     os_months, os_status) = load_data(data_dir, use_mutations=use_mutations)

    print(f"\nDataset summary:")
    print(f"  Patients:  {len(y_all)}")
    print(f"  LTS:       {int(y_all.sum())} ({100*y_all.mean():.1f}%)")
    print(f"  non-LTS:   {int((y_all==0).sum())} ({100*(1-y_all.mean()):.1f}%)")
    print(f"  CNA:       {X_cna.shape}")
    print(f"  mRNA:      {X_mrna.shape}")
    print(f"  Meth:      {X_meth.shape}")
    if use_mutations and X_mut is not None:
        print(f"  Mutation:  {X_mut.shape}")
    else:
        print(f"  Mutation:  SKIPPED (use_mutations={use_mutations})")
    print(f"  Clinical:  {X_clin.shape}  (cols: {list(X_clin.columns)})")

    # ── 1. Stratified split ──────────────────────────────────────
    (X_cna_tr,  X_cna_te,
     X_mrna_tr, X_mrna_te,
     X_meth_tr, X_meth_te,
     X_mut_tr,  X_mut_te,
     X_clin_tr, X_clin_te,
     y_train,   y_test,
     os_months_train, os_months_test,
     os_status_train, os_status_test) = split_data(
        X_cna, X_mrna, X_meth, X_mut if use_mutations else None, X_clin,
        y_all, os_months, os_status
    )

    # ── 2. mRMR (fit on training only) — 4 modalities ────────────
    mrmr_results = run_mrmr_all_modalities(
        X_cna_tr,  X_mrna_tr,  X_meth_tr,  y_train,
        X_cna_te,  X_mrna_te,  X_meth_te,
        K=K_MRMR,
        X_mut_train=X_mut_tr if use_mutations else None,
        X_mut_test=X_mut_te if use_mutations else None,
    )

    cna_tr_r  = mrmr_results["cna_train"]
    mrna_tr_r = mrmr_results["mrna_train"]
    meth_tr_r = mrmr_results["meth_train"]
    mut_tr_r  = mrmr_results.get("mut_train")
    cna_te_r  = mrmr_results["cna_test"]
    mrna_te_r = mrmr_results["mrna_test"]
    meth_te_r = mrmr_results["meth_test"]
    mut_te_r  = mrmr_results.get("mut_test")

    clin_tr_arr = X_clin_tr.values.astype(float)
    clin_te_arr = X_clin_te.values.astype(float)

    # ── 3. Build PSN — 4 omics modalities ────────────────────────
    psn_omics = build_psn(cna_tr_r, mrna_tr_r, meth_tr_r, mut_tr_r if use_mutations else None)

    # ── 3b. Make PSN survival-aware ──────────────────────────────
    psn_real, S_survival = build_survival_aware_psn(
        psn_omics,
        os_months=os_months_train.values,
        os_status=os_status_train.values,
        alpha=ALPHA_SURVIVAL,
        sigma=None
    )

    # ── 4. PSN diagnostics — compare omics-only vs survival-aware ─
    print("  [Omics-only PSN diagnostics]")
    diag_omics = psn_diagnostics(psn_omics, y_train.values, top_k=20)
    print("  [Survival-aware PSN diagnostics]")
    diag       = psn_diagnostics(psn_real,  y_train.values, top_k=20)

    compare_psn_diagnostics(
        psn_omics, psn_real, y_train.values,
        top_k=20, n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE
    )
    print(f"  <- Compare z={diag['z_score']:.3f} (survival-aware) "
          f"vs z={diag_omics['z_score']:.3f} (omics-only)")

    print("\n── Outputs Ready for GCN ───────────────────────────────────")
    print(f"  psn_real:        {psn_real.shape}")
    train_feature_msg = (
        f"  Train features:  CNA{cna_tr_r.shape} + mRNA{mrna_tr_r.shape} + "
        f"Meth{meth_tr_r.shape} + Mut{mut_tr_r.shape} + Clin{clin_tr_arr.shape}"
        if use_mutations and mut_tr_r is not None else
        f"  Train features:  CNA{cna_tr_r.shape} + mRNA{mrna_tr_r.shape} + "
        f"Meth{meth_tr_r.shape} + Clin{clin_tr_arr.shape}"
    )
    print(train_feature_msg)
    print(f"  y_train:         LTS={int(y_train.sum())}, non-LTS={int((y_train==0).sum())}")
    print("=" * 62)

    return {
        "use_mutations":  use_mutations,
        # PSN
        "psn_real":        psn_real,
        "psn_omics":       psn_omics,
        "S_survival":      S_survival,
        # Training
        "cna_tr_r":        cna_tr_r,
        "mrna_tr_r":       mrna_tr_r,
        "meth_tr_r":       meth_tr_r,
        "mut_tr_r": mut_tr_r if use_mutations else None,
        "clin_tr_arr":     clin_tr_arr,
        "y_train":         y_train,
        "os_months_train": os_months_train,
        "os_status_train": os_status_train,
        # Test
        "cna_te":          cna_te_r,
        "mrna_te":         mrna_te_r,
        "meth_te":         meth_te_r,
        "mut_te": mut_te_r if use_mutations else None,
        "clin_te":         clin_te_arr,
        "y_test":          y_test,
        "os_months_test":  os_months_test,
        "os_status_test":  os_status_test,
        # Feature names
        "cna_features":    mrmr_results["cna_features"],
        "mrna_features":   mrmr_results["mrna_features"],
        "meth_features":   mrmr_results["meth_features"],
        "mut_features": mrmr_results.get("mut_features") if use_mutations else None,        
        # Diagnostics
        "psn_diagnostics":       diag,
        "psn_diagnostics_omics": diag_omics,
        # Paths
        "plots_dir":       plots_dir,
        # Raw full DataFrames for driver gene analysis
        "X_cna_full":      X_cna,
        "X_mrna_full":     X_mrna,
        "X_meth_full":     X_meth,
        "X_mut_full":      X_mut,
        "os_months_full":  os_months,
        "os_status_full":  os_status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT (standalone)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import DEFAULT_THRESHOLD

    parser = argparse.ArgumentParser(
        description="Run the GBM PSN-GCN pipeline for one LTS threshold."
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=f"Processed data directory (default: data_{{threshold}}m)"
    )
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"LTS threshold month label used to name default dirs "
             f"(default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--plots_dir", type=str, default=None,
        help="Plots output directory (default: plots_{threshold}m)"
    )
    parser.add_argument(
        "--use_mutations",
        action="store_true",
        help="Include mutation modality (default: False)"
    )
    args = parser.parse_args()

    data_dir  = args.data_dir  or f"data_{args.threshold}m"
    plots_dir = args.plots_dir or f"plots_{args.threshold}m"
    use_mutations_flag = args.use_mutations

    # ── Run preprocessing pipeline ───────────────────────────────
    results = main(data_dir=data_dir, plots_dir=plots_dir, use_mutations=use_mutations_flag)

    # ── Train GCN — binary + Cox dual head ───────────────────────
    gcn_results = train_gcn(results)

    # ── Generate KM plots from Cox risk scores ───────────────────
    gcn_results["y_test_np"] = results["y_test"].values
    generate_all_km_plots(gcn_results, output_dir=plots_dir)

    # ── Baseline comparison ──────────────────────────────────────
    run_baseline_comparison(results, gcn_results, output_dir=plots_dir)

    # ── Ablation study ───────────────────────────────────────────
    run_ablation(results, output_dir=plots_dir, gcn_results=gcn_results)

    # ── Driver gene analysis — which genes drive risk? ────────────
    run_driver_gene_analysis(
        raw_data = {
            "X_cna":     results["X_cna_full"],
            "X_mrna":    results["X_mrna_full"],
            "X_meth":    results["X_meth_full"],
            "X_mut": results["X_mut_full"],
            "os_months": results["os_months_full"],
            "os_status": results["os_status_full"],
        },
        output_dir =plots_dir,
    )