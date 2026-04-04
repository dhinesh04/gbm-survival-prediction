"""
ablation_studies.py
-------------------
Ablation study for the multi-omics GBM survival prediction pipeline.

Seven configurations run on the SAME train/test split, all using
survival-aware SNF unless explicitly labelled otherwise:

  A1.  CNA         + Clinical  | mRMR | SNF-SA | GCN
  A2.  mRNA        + Clinical  | mRMR | SNF-SA | GCN
  A3.  Methylation + Clinical  | mRMR | SNF-SA | GCN
  A4.  CNA + mRNA + Meth       | mRMR | SNF-SA | GCN  (no clinical)
  A5.  Clinical only           | —    | SNF-SA | GCN  (4 features, no mRMR)
  A6.  All 4 modalities        | mRMR | SNF (omics-only, NOT surv-aware) | GCN
  A7.  All 4 modalities        | mRMR | SNF-SA | GCN  ← FULL MODEL

All ROC curves are plotted on a single figure with AUC in the legend.

All hyperparameters are imported from config.py.
Class weights are computed dynamically from the pipeline's y_train
so they are correct for every LTS threshold (12 / 18 / 24 months).

Usage
-----
  from ablation_studies import run_ablation
  run_ablation(pipeline_results)          # pass the dict returned by main()
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import snf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

from src.models.gcn_model import GCN
from src.graph.survival_aware_psn import build_survival_aware_psn
from config import (
    HIDDEN_DIM, DROPOUT, LR, WEIGHT_DECAY,
    EPOCHS, PATIENCE, MIN_EPOCHS,
    N_FOLDS, K_TEST, RANDOM_STATE,
    ALPHA_BIN, ALPHA_COX,
    K_SNF, MU_SNF, N_ITER_SNF,
    ALPHA_SURVIVAL,
)
from src.utils import (
    concordance_index,
    cox_partial_likelihood_loss,
    normalise_adjacency,
    attach_test_nodes,
    find_best_threshold,
    compute_class_weights,
    plot_roc_curves,
)


# ─────────────────────────────────────────────────────────────────────────────
# PSN BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def _build_psn(matrices: list,
               os_months: np.ndarray,
               os_status: np.ndarray,
               survival_aware: bool) -> np.ndarray:
    """
    Build PSN from a list of feature matrices via SNF, then optionally
    apply survival-aware blending.
    """
    affinity_mats = [snf.make_affinity(m, K=K_SNF, mu=MU_SNF) for m in matrices]
    psn_omics = affinity_mats[0] if len(affinity_mats) == 1 \
                else snf.snf(affinity_mats, K=K_SNF, t=N_ITER_SNF)

    if survival_aware:
        psn_out, _ = build_survival_aware_psn(
            psn_omics, os_months, os_status,
            alpha=ALPHA_SURVIVAL, sigma=None
        )
        return psn_out
    return psn_omics.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def _run_gcn(X_train_np, y_train_np,
             X_test_np,  y_test_np,
             psn_train,
             times_train, events_train,
             times_test,  events_test,
             class_weights: torch.Tensor,
             label: str
             ) -> dict:
    """
    5-fold CV + final retrain + test evaluation for one ablation config.

    Parameters
    ----------
    class_weights : torch.Tensor([w_nonlts, w_lts]) — passed in from
                    run_ablation(), computed dynamically from y_train.
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    n_tr  = X_train_np.shape[0]
    n_te  = X_test_np.shape[0]
    n_tot = n_tr + n_te

    full_adj = attach_test_nodes(psn_train, X_train_np, X_test_np, k=K_TEST)
    adj      = normalise_adjacency(full_adj)

    X_all = torch.tensor(np.vstack([X_train_np, X_test_np]),      dtype=torch.float)
    y_all = torch.tensor(np.concatenate([y_train_np, y_test_np]), dtype=torch.long)
    t_all = torch.tensor(np.concatenate([times_train, times_test]),   dtype=torch.float)
    e_all = torch.tensor(np.concatenate([events_train, events_test]), dtype=torch.float)
    idx_test = torch.arange(n_tr, n_tot)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs, fold_threshs, fold_epochs = [], [], []

    for tr_idx, val_idx in skf.split(X_train_np, y_train_np):
        idx_tr  = torch.tensor(tr_idx,  dtype=torch.long)
        idx_val = torch.tensor(val_idx, dtype=torch.long)

        model = GCN(n_in=X_all.shape[1], n_hid=HIDDEN_DIM, n_out=2, dropout=DROPOUT)
        opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_auc, best_thresh, best_ep, bad = 0.0, 0.5, MIN_EPOCHS, 0

        for epoch in range(EPOCHS):
            # Train
            model.train(); opt.zero_grad()
            bl, cr, _ = model(X_all, adj)
            l_bin = F.cross_entropy(bl[idx_tr], y_all[idx_tr], weight=class_weights)
            l_cox = cox_partial_likelihood_loss(cr[idx_tr], t_all[idx_tr], e_all[idx_tr])
            (ALPHA_BIN * l_bin + ALPHA_COX * l_cox).backward()
            opt.step()

            # Validate
            model.eval()
            with torch.no_grad():
                bl_all_e, _, _ = model(X_all, adj)
            vp = torch.softmax(bl_all_e[idx_val], dim=1)[:, 1].cpu().numpy()
            vt = y_all[idx_val].cpu().numpy()
            if len(np.unique(vt)) < 2:
                continue

            vauc = roc_auc_score(vt, vp)
            if vauc > best_auc:
                best_auc    = vauc
                best_thresh = find_best_threshold(vp, vt)
                if epoch + 1 >= MIN_EPOCHS:
                    best_ep = epoch + 1
                bad = 0
            else:
                if epoch + 1 >= MIN_EPOCHS:
                    bad += 1
                    if bad >= PATIENCE:
                        break

        fold_aucs.append(best_auc)
        fold_threshs.append(best_thresh)
        fold_epochs.append(best_ep)


    final_epochs = max(MIN_EPOCHS, int(np.median(fold_epochs)))
    mean_val_auc = float(np.mean(fold_aucs))

    # Final retrain on all training patients
    idx_all_tr = torch.arange(n_tr)
    model = GCN(n_in=X_all.shape[1], n_hid=HIDDEN_DIM, n_out=2, dropout=DROPOUT)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(final_epochs):
        model.train(); opt.zero_grad()
        bl, cr, _ = model(X_all, adj)
        l_bin = F.cross_entropy(bl[idx_all_tr], y_all[idx_all_tr], weight=class_weights)
        l_cox = cox_partial_likelihood_loss(cr[idx_all_tr], t_all[idx_all_tr], e_all[idx_all_tr])
        (ALPHA_BIN * l_bin + ALPHA_COX * l_cox).backward()
        opt.step()

    # Test evaluation
    model.eval()
    with torch.no_grad():
        bl_all_f, cr_all_f, _ = model(X_all, adj)

    test_probs = torch.softmax(bl_all_f[idx_test], dim=1)[:, 1].cpu().numpy()
    test_true  = y_all[idx_test].cpu().numpy()
    test_risk  = cr_all_f[idx_test].cpu().numpy()
    test_auc   = roc_auc_score(test_true, test_probs)
    cindex     = concordance_index(test_risk, times_test, events_test)

    print(f"  [{label:45s}]  AUC={test_auc:.4f}  C-index={cindex:.4f}  "
          f"(CV AUC={mean_val_auc:.4f}  epochs={final_epochs})")

    return {
        "label":       label,
        "auc":         test_auc,
        "cindex":      cindex,
        "probs":       test_probs,
        "y_true":      test_true,
        "cv_auc_mean": mean_val_auc,
        "cv_auc_std":  float(np.std(fold_aucs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _print_summary(results: list):
    print("\n" + "=" * 75)
    print("  ABLATION STUDY — SUMMARY TABLE")
    print("=" * 75)
    print(f"  {'Configuration':<47} {'AUC':>6}  {'C-index':>8}  {'CV AUC':>14}")
    print("  " + "-" * 73)
    for r in results:
        marker = " <-- FULL MODEL" if "Full model" in r["label"] else ""
        print(f"  {r['label']:<47} {r['auc']:>6.4f}  "
              f"{r['cindex']:>8.4f}  "
              f"{r['cv_auc_mean']:>6.4f}±{r['cv_auc_std']:.4f}"
              f"{marker}")
    print("=" * 75)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(pipeline: dict, output_dir: str = "plots", gcn_results: dict = None) -> list:
    """
    Run all 7 ablation configurations and produce the combined ROC plot.

    Class weights are computed dynamically from pipeline['y_train'] so
    they are correct for every LTS threshold experiment (12 / 18 / 24m).

    Parameters
    ----------
    pipeline   : dict returned by main.main()
    output_dir : directory for saving the ROC figure
    """
    # ── Unpack pipeline data ─────────────────────────────────────────────────
    cna_tr  = pipeline["cna_tr_r"]
    mrna_tr = pipeline["mrna_tr_r"]
    meth_tr = pipeline["meth_tr_r"]
    clin_tr = pipeline["clin_tr_arr"]
    cna_te  = pipeline["cna_te"]
    mrna_te = pipeline["mrna_te"]
    meth_te = pipeline["meth_te"]
    clin_te = pipeline["clin_te"]

    y_tr = pipeline["y_train"].values
    y_te = pipeline["y_test"].values
    t_tr = pipeline["os_months_train"].values.astype(float)
    e_tr = pipeline["os_status_train"].values.astype(float)
    t_te = pipeline["os_months_test"].values.astype(float)
    e_te = pipeline["os_status_test"].values.astype(float)

    # ── Dynamic class weights ────────────────────────────────────────────────
    class_weights, n_lts, _ = compute_class_weights(y_tr)

    print("\n" + "=" * 75)
    print("  ABLATION STUDY  — 7 Configurations")
    print("=" * 75)
    print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")
    print(f"  LTS train={n_lts} | LTS test={int(y_te.sum())}")
    print(f"  Class weights (dynamic): non-LTS={class_weights[0]:.3f}, LTS={class_weights[1]:.3f}")
    print(f"  Survival-aware alpha = {ALPHA_SURVIVAL}")
    print("=" * 75 + "\n")

    results = []

    # A1 — CNA only ───────────────────────────────────────────────────────────
    print("── A1: CNA + Clinical ──────────────────────────────────────────")
    psn = _build_psn([cna_tr], t_tr, e_tr, survival_aware=True)
    results.append(_run_gcn(
        cna_tr, y_tr, cna_te, y_te, psn,
        t_tr, e_tr, t_te, e_te, class_weights, label="CNA only"
    ))

    # A2 — mRNA only ──────────────────────────────────────────────────────────
    print("── A2: mRNA + Clinical ─────────────────────────────────────────")
    psn = _build_psn([mrna_tr], t_tr, e_tr, survival_aware=True)
    results.append(_run_gcn(
        mrna_tr, y_tr, mrna_te, y_te, psn,
        t_tr, e_tr, t_te, e_te, class_weights, label="mRNA only"
    ))

    # A3 — Methylation only ───────────────────────────────────────────────────
    print("── A3: Methylation + Clinical ──────────────────────────────────")
    psn = _build_psn([meth_tr], t_tr, e_tr, survival_aware=True)
    results.append(_run_gcn(
        meth_tr, y_tr, meth_te, y_te, psn,
        t_tr, e_tr, t_te, e_te, class_weights, label="Methylation only"
    ))

    # A4 — 3 Omics, no Clinical ───────────────────────────────────────────────
    print("── A4: 3 Omics only (no Clinical) ──────────────────────────────")
    psn  = _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, survival_aware=True)
    X_tr = np.hstack([cna_tr,  mrna_tr,  meth_tr])
    X_te = np.hstack([cna_te,  mrna_te,  meth_te])
    results.append(_run_gcn(
        X_tr, y_tr, X_te, y_te, psn,
        t_tr, e_tr, t_te, e_te, class_weights, label="3 Omics only (no Clinical)"
    ))

    # A5 — Clinical only ──────────────────────────────────────────────────────
    print("── A5: Clinical only ───────────────────────────────────────────")
    psn = _build_psn([clin_tr], t_tr, e_tr, survival_aware=True)
    results.append(_run_gcn(
        clin_tr, y_tr, clin_te, y_te, psn,
        t_tr, e_tr, t_te, e_te, class_weights, label="Clinical only"
    ))

    # A6 — All 4 + SNF omics-only (no survival-aware) ─────────────────────────
    print("── A6: All 4 + SNF (omics-only, no survival-aware) ─────────────")
    psn  = _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, survival_aware=False)
    X_tr = np.hstack([cna_tr,  mrna_tr,  meth_tr,  clin_tr])
    X_te = np.hstack([cna_te,  mrna_te,  meth_te,  clin_te])
    results.append(_run_gcn(
        X_tr, y_tr, X_te, y_te, psn,
        t_tr, e_tr, t_te, e_te, class_weights, label="All 4 + SNF (no surv-aware)"
    ))

    # A7 — Full model: inject results from the main GCN run directly.
    # Re-running _run_gcn here would produce different results due to
    # RNG state divergence after CV folds, even with an identical epoch count.
    # Using gcn_results guarantees Table 1 and Table 3 are always consistent.
    print("── A7: All 4 + SNF-SA (FULL MODEL) ────────────────────────────")
    if gcn_results is not None:
        a7 = {
            "label":       "All 4 + SNF-SA (Full model)",
            "auc":         gcn_results["auc"],
            "cindex":      gcn_results["cindex"],
            "probs":       gcn_results["probs"],
            "y_true":      gcn_results["y_true"],
            "cv_auc_mean": gcn_results["cv_val_auc_mean"],
            "cv_auc_std":  gcn_results["cv_val_auc_std"],
        }
        print(f"  [{'All 4 + SNF-SA (Full model)':45s}]  "
              f"AUC={a7['auc']:.4f}  C-index={a7['cindex']:.4f}  "
              f"(CV AUC={a7['cv_auc_mean']:.4f}  "
              f"epochs={gcn_results['final_epochs_used']})")
    else:
        X_tr = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
        X_te = np.hstack([cna_te, mrna_te, meth_te, clin_te])
        a7 = _run_gcn(
            X_tr, y_tr, X_te, y_te, pipeline["psn_real"],
            t_tr, e_tr, t_te, e_te, class_weights,
            label="All 4 + SNF-SA (Full model)"
        )
    results.append(a7)

    # ── Summary + Plot ───────────────────────────────────────────────────────
    _print_summary(results)
    plot_roc_curves(
        results,
        output_path=os.path.join(output_dir, "ablation_roc_curves.png"),
        title="GBM LTS Prediction — Ablation ROC Curves",
    )

    return results