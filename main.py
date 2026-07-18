"""Full pipeline: split -> mRMR -> PSN via SNF -> PSN diagnostics -> GCN training -> KM plots."""

import argparse
import os
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
from src.models.gcn_train import train_gcn, save_patient_predictions
from src.visualization.km_plot import generate_all_km_plots
from src.graph.survival_aware_psn import build_survival_aware_psn, compare_psn_diagnostics
from src.analysis.baseline_comparison import run_baseline_comparison
from src.analysis.ablation_studies import run_ablation
from src.analysis.driver_gene_deeplift import run_deeplift_analysis


def load_data(data_dir: str = DEFAULT_DATA_DIR,
              use_mutations: bool = False):
    X_cna  = pd.read_csv(f"{data_dir}/cna_data.csv",          index_col=0)
    X_mrna = pd.read_csv(f"{data_dir}/mrna_data.csv",         index_col=0)
    X_meth = pd.read_csv(f"{data_dir}/methylation_data.csv",  index_col=0)
    X_mut  = pd.read_csv(f"{data_dir}/mutation_data.csv",     index_col=0)
    X_clin = pd.read_csv(f"{data_dir}/clinical_data.csv",     index_col=0)
    labels = pd.read_csv(f"{data_dir}/labels.csv",            index_col=0)

    os_months = labels["OS_MONTHS"].astype(float)
    os_status = labels["OS_STATUS"].astype(int)

    # main cohort is the 4 core modalities only; mutation coverage never
    # shrinks it since the GCN itself doesn't consume mutation features
    common = (
        X_cna.index
        .intersection(X_mrna.index)
        .intersection(X_meth.index)
        .intersection(X_clin.index)
        .intersection(os_status.index)
    )

    X_cna     = X_cna.loc[common]
    X_mrna    = X_mrna.loc[common]
    X_meth    = X_meth.loc[common]
    X_clin    = X_clin.loc[common]
    os_months = os_months.loc[common]
    os_status = os_status.loc[common]

    # mutation data (driver-gene analysis only) intersected with the fixed cohort
    if use_mutations:
        mut_common = X_mut.index.intersection(common)
        X_mut = X_mut.loc[mut_common]
        print(f"  Mutation data: {len(mut_common)}/{len(common)} patients "
              f"in the main cohort have mutation coverage.")
    else:
        X_mut = None

    print(f"\nAfter aligning common patients: {len(common)}")
    return X_cna, X_mrna, X_meth, X_mut, X_clin, os_months, os_status


def split_data(X_cna, X_mrna, X_meth, X_mut, X_clin,
               os_months, os_status):
    """Stratified 70/30 split on os_status, applied to all modalities."""
    print("\n── Stratified Train/Test Split ─────────────────────────────")

    idx_train, idx_test = train_test_split(
        os_status.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=os_status,
    )

    def _s(df):
        return df.loc[idx_train], df.loc[idx_test]

    X_cna_tr,  X_cna_te  = _s(X_cna)
    X_mrna_tr, X_mrna_te = _s(X_mrna)
    X_meth_tr, X_meth_te = _s(X_meth)
    X_clin_tr, X_clin_te = X_clin.loc[idx_train].copy(), X_clin.loc[idx_test].copy()
    X_mut_tr,  X_mut_te  = _s(X_mut) if X_mut is not None else (None, None)

    os_months_train = os_months.loc[idx_train]
    os_months_test  = os_months.loc[idx_test]
    os_status_train = os_status.loc[idx_train]
    os_status_test  = os_status.loc[idx_test]

    print(f"  Train: {len(idx_train)} patients  "
          f"(deceased={int(os_status_train.sum())}, "
          f"censored={int((os_status_train==0).sum())})")
    print(f"  Test:  {len(idx_test)} patients  "
          f"(deceased={int(os_status_test.sum())}, "
          f"censored={int((os_status_test==0).sum())})")
    print(f"  Clinical features: {list(X_clin_tr.columns)}")
    print("────────────────────────────────────────────────────────────\n")

    return (X_cna_tr,  X_cna_te,
            X_mrna_tr, X_mrna_te,
            X_meth_tr, X_meth_te,
            X_mut_tr,  X_mut_te,
            X_clin_tr, X_clin_te,
            os_months_train, os_months_test,
            os_status_train, os_status_test)


def build_psn(cna, mrna, meth, mut=None):
    print("\n── Building PSN via SNF ────────────────────────────────────")
    affinities = [
        snf.make_affinity(cna,  K=K_SNF, mu=MU_SNF),
        snf.make_affinity(mrna, K=K_SNF, mu=MU_SNF),
        snf.make_affinity(meth, K=K_SNF, mu=MU_SNF),
    ]
    if mut is not None:
        affinities.append(snf.make_affinity(mut, K=K_SNF, mu=MU_SNF))
    psn = snf.snf(affinities, K=K_SNF, t=N_ITER_SNF)
    print(f"  PSN shape: {psn.shape}  "
          f"({len(affinities)} modalities, {N_ITER_SNF} SNF iterations)")
    print("────────────────────────────────────────────────────────────\n")
    return psn


def psn_diagnostics(psn, y_labels, top_k=20):
    """Hubness + label smoothness z-score."""
    print("\n── PSN Diagnostics ─────────────────────────────────────────")

    n = psn.shape[0]
    psn_nd = psn.copy()
    np.fill_diagonal(psn_nd, 0)

    is_symmetric = bool(np.allclose(psn_nd, psn_nd.T, atol=1e-6))
    is_nonneg    = bool((psn_nd >= 0).all())
    max_indegree = int((psn_nd > 0).sum(axis=0).max())

    print(f"  Symmetric:      {is_symmetric}")
    print(f"  Non-negative:   {is_nonneg}")
    print(f"  Max in-degree:  {max_indegree}  (hubness; ideal < {n // 5})")

    topk_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        topk_mask[i, np.argsort(psn_nd[i])[::-1][:top_k]] = True

    y_arr = np.array(y_labels)
    agreements = [
        (y_arr[np.where(topk_mask[i])[0]] == y_arr[i]).mean()
        for i in range(n) if topk_mask[i].any()
    ]
    observed_agreement = float(np.mean(agreements))

    rng = np.random.default_rng(RANDOM_STATE)
    perm_agreements = []
    for _ in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y_arr)
        perm_agreements.append(np.mean([
            (y_perm[np.where(topk_mask[i])[0]] == y_perm[i]).mean()
            for i in range(n) if topk_mask[i].any()
        ]))

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


def main(data_dir:      str  = DEFAULT_DATA_DIR,
         plots_dir:     str  = DEFAULT_PLOTS_DIR,
         use_mutations: bool = False):
    """Run the full preprocessing + GCN training pipeline."""
    print("=" * 62)
    print("  GBM Survival Prediction — Preprocessing Pipeline")
    print(f"  Data dir:  {data_dir}")
    print(f"  Plots dir: {plots_dir}")
    print("=" * 62)

    # ── 0. Load ──────────────────────────────────────────────────
    (X_cna, X_mrna, X_meth, X_mut, X_clin,
     os_months, os_status) = load_data(
        data_dir, use_mutations=use_mutations)

    print(f"\nDataset summary:")
    print(f"  Patients:  {len(os_status)}")
    print(f"  CNA:       {X_cna.shape}")
    print(f"  mRNA:      {X_mrna.shape}")
    print(f"  Meth:      {X_meth.shape}")
    if use_mutations and X_mut is not None:
        print(f"  Mutation:  {X_mut.shape}")
    else:
        print(f"  Mutation:  SKIPPED")
    print(f"  Clinical:  {X_clin.shape}  (cols: {list(X_clin.columns)})")

    # ── 1. Split ─────────────────────────────────────────────────
    (X_cna_tr,  X_cna_te,
     X_mrna_tr, X_mrna_te,
     X_meth_tr, X_meth_te,
     X_mut_tr,  X_mut_te,
     X_clin_tr, X_clin_te,
     os_months_train, os_months_test,
     os_status_train, os_status_test) = split_data(
        X_cna, X_mrna, X_meth,
        None,
        X_clin, os_months, os_status,
    )
    print((os_months_train[os_status_train==1] > 30).sum(), "/", int(os_status_train.sum()), "long-survivor deceased in TRAIN")
    

    # ── 2. Data-Driven Split for mRMR & Diagnostics ──────────────
    median_surv = os_months_train.median()
    print(f"\n  [Data-Driven Binarization] Median training survival: {median_surv:.1f} months")
    print(f"  Used exclusively for PSN label smoothness diagnostics.")
    y_median_train = (os_months_train > median_surv).astype(int)

    # ── 2b. Continuous, deceased-only target for mRMR relevance ──
    deceased_mask = (os_status_train.values == 1)
    y_logtime_train = np.log(os_months_train.clip(lower=0.01))
    print(f"  [mRMR relevance target] log(OS_MONTHS), fit on "
        f"{deceased_mask.sum()}/{len(os_status_train)} deceased training "
        f"patients -- continuous, not median-split; censored patients' "
        f"lower-bound times are excluded from relevance, not from training.")

    # ── 3. mRMR ──────────────────────────────────────────────────
    mrmr_results = run_mrmr_all_modalities(
        X_cna_tr,  X_mrna_tr,  X_meth_tr,  y_logtime_train,
        X_cna_te,  X_mrna_te,  X_meth_te,
        K=K_MRMR,
        X_mut_train=None,
        X_mut_test=None,
        relevance_mask=deceased_mask,
    )

    cna_tr_r    = mrmr_results["cna_train"]
    mrna_tr_r   = mrmr_results["mrna_train"]
    meth_tr_r   = mrmr_results["meth_train"]
    mut_tr_r    = mrmr_results.get("mut_train")
    cna_te_r    = mrmr_results["cna_test"]
    mrna_te_r   = mrmr_results["mrna_test"]
    meth_te_r   = mrmr_results["meth_test"]
    mut_te_r    = mrmr_results.get("mut_test")
    clin_tr_arr = X_clin_tr.values.astype(float)
    clin_te_arr = X_clin_te.values.astype(float)

    # ── 4. Build PSN ─────────────────────────────────────────────
    psn_omics = build_psn(
        cna_tr_r, mrna_tr_r, meth_tr_r,
    )

    psn_real, S_survival = build_survival_aware_psn(
        psn_omics,
        os_months=os_months_train.values,
        os_status=os_status_train.values,
        alpha=ALPHA_SURVIVAL,
        sigma=None,
    )

    # ── 5. PSN diagnostics ────────────────────────────────────────
    print("  [Omics-only PSN diagnostics]")
    diag_omics = psn_diagnostics(psn_omics, y_median_train.values, top_k=20)
    print("  [Survival-aware PSN diagnostics]")
    diag       = psn_diagnostics(psn_real,  y_median_train.values, top_k=20)

    compare_psn_diagnostics(
        psn_omics, psn_real, y_median_train.values,
        top_k=20, n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE,
    )
    print(f"  <- Compare z={diag['z_score']:.3f} (survival-aware) "
          f"vs z={diag_omics['z_score']:.3f} (omics-only)")

    print("\n── Outputs Ready for GCN ───────────────────────────────────")
    print(f"  psn_real:       {psn_real.shape}")
    if use_mutations and mut_tr_r is not None:
        print(f"  Train features: CNA{cna_tr_r.shape} + mRNA{mrna_tr_r.shape} + "
              f"Meth{meth_tr_r.shape} + Mut{mut_tr_r.shape} + Clin{clin_tr_arr.shape}")
    else:
        print(f"  Train features: CNA{cna_tr_r.shape} + mRNA{mrna_tr_r.shape} + "
              f"Meth{meth_tr_r.shape} + Clin{clin_tr_arr.shape}")
    print("=" * 62)

    return {
        "use_mutations":   use_mutations,
        # PSN
        "psn_real":        psn_real,
        "psn_omics":       psn_omics,
        "S_survival":      S_survival,
        # Training features
        "cna_tr_r":        cna_tr_r,
        "mrna_tr_r":       mrna_tr_r,
        "meth_tr_r":       meth_tr_r,
        "mut_tr_r":        mut_tr_r if use_mutations else None,
        "clin_tr_arr":     clin_tr_arr,
        "os_months_train": os_months_train,
        "os_status_train": os_status_train,
        # Test features
        "cna_te":          cna_te_r,
        "mrna_te":         mrna_te_r,
        "meth_te":         meth_te_r,
        "mut_te":          mut_te_r if use_mutations else None,
        "clin_te":         clin_te_arr,
        "os_months_test":  os_months_test,
        "os_status_test":  os_status_test,
        # Feature names
        "cna_features":    mrmr_results["cna_features"],
        "mrna_features":   mrmr_results["mrna_features"],
        "meth_features":   mrmr_results["meth_features"],
        "mut_features":    mrmr_results.get("mut_features") if use_mutations else None,
        # Diagnostics
        "psn_diagnostics":       diag,
        "psn_diagnostics_omics": diag_omics,
        # Paths
        "plots_dir":       plots_dir,
        # Full DataFrames for driver gene analysis
        "X_cna_full":      X_cna,
        "X_mrna_full":     X_mrna,
        "X_meth_full":     X_meth,
        "X_mut_full":      X_mut,
        "os_months_full":  os_months,
        "os_status_full":  os_status,
        "clin_feature_names": list(X_clin.columns)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the GBM PSN-GCN pipeline."
    )
    parser.add_argument(
        "--data_dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"Processed data directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--plots_dir", type=str, default=DEFAULT_PLOTS_DIR,
        help=f"Plots output directory (default: {DEFAULT_PLOTS_DIR})"
    )
    parser.add_argument(
        "--use_mutations", action="store_true",
        help="Include mutation modality (default: False)"
    )
    args = parser.parse_args()

    results = main(
        data_dir=args.data_dir,
        plots_dir=args.plots_dir,
        use_mutations=args.use_mutations,
    )

    # ── Train GCN — AFT regression + Cox dual head ───────────────
    gcn_results = train_gcn(results)

    import torch
    _m = gcn_results["model"]
    _m.eval()
    with torch.no_grad():
        _pred_log_t_all, _cox_risk_all, _ = _m(
            gcn_results["X_all_final"],
            gcn_results["adj_final"],
        )
    _all_pids = (list(results["os_months_train"].index) +
                list(results["os_months_test"].index))
    pd.DataFrame({
        "PATIENT_ID":      _all_pids,
        "cox_risk_score":  _cox_risk_all.cpu().numpy(),
        "aft_pred_months": torch.exp(_pred_log_t_all).cpu().numpy(),
    }).to_csv(
        os.path.join(args.plots_dir, "gcn_risk_scores.csv"), index=False)
    print(f"  GCN risk scores saved → {args.plots_dir}/gcn_risk_scores.csv")

    # ── Per-patient AFT/Cox predictions (test set) ───────────────
    save_patient_predictions(results, gcn_results, output_dir=args.plots_dir)

    # ── KM plots from Cox risk scores ────────────────────────────
    generate_all_km_plots(gcn_results, output_dir=args.plots_dir)

    # ── Baseline comparison ──────────────────────────────────────
    run_baseline_comparison(results, gcn_results, output_dir=args.plots_dir)

    # ── Ablation study ───────────────────────────────────────────
    run_ablation(results, output_dir=args.plots_dir, gcn_results=gcn_results)

    # ── Driver gene analysis ─────────────────────────────────────
    run_deeplift_analysis(
        gcn_results=gcn_results,
        pipeline_results=results,
        raw_data={
            "X_cna":     results["X_cna_full"],
            "X_mrna":    results["X_mrna_full"],
            "X_meth":    results["X_meth_full"],
            "X_mut":     results["X_mut_full"],
            "os_months": results["os_months_full"],
            "os_status": results["os_status_full"],
        },
        output_dir=args.plots_dir,
    )