"""
ablation_studies.py
-------------------
Eight configurations on the SAME train/test split.

  A1.  CNA         + Clinical  | mRMR | SNF-SA | GCN
  A2.  mRNA        + Clinical  | mRMR | SNF-SA | GCN
  A3.  Methylation + Clinical  | mRMR | SNF-SA | GCN
  A4.  CNA + mRNA + Meth       | mRMR | SNF-SA | GCN  (no clinical)
  A5.  Clinical only           | —    | SNF-SA | GCN
  A6.  All 4 modalities        | mRMR | SNF (omics-only) | GCN
  A7.  All 4 modalities        | mRMR | SNF-SA | GCN               ← no encoders
  A8.  All 4 modalities        | mRMR | SNF-SA | GCN + mod encoders ← FULL MODEL

A7 vs A8 isolates the contribution of per-modality encoders.
All CV uses per-fold PSN rebuild (no label leakage).
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
from sklearn.metrics import roc_auc_score

from src.models.gcn_model import GCN
from src.graph.survival_aware_psn import build_survival_aware_psn
from config import (
    HIDDEN_DIM, DROPOUT, LR, WEIGHT_DECAY,
    EPOCHS, PATIENCE, MIN_EPOCHS,
    N_FOLDS, K_TEST, RANDOM_STATE,
    ALPHA_BIN, ALPHA_COX,
    K_SNF, MU_SNF, N_ITER_SNF,
    ALPHA_SURVIVAL,
    ENC_DIM,
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
def _build_psn(matrices, os_months, os_status, survival_aware):
    affinities = [snf.make_affinity(m, K=K_SNF, mu=MU_SNF) for m in matrices]
    psn_omics  = (affinities[0] if len(affinities) == 1
                  else snf.snf(affinities, K=K_SNF, t=N_ITER_SNF))
    if survival_aware:
        psn_out, _ = build_survival_aware_psn(
            psn_omics, os_months, os_status, alpha=ALPHA_SURVIVAL, sigma=None)
        return psn_out
    return psn_omics.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRAINING (per-fold PSN + optional modality encoders)
# ─────────────────────────────────────────────────────────────────────────────
def _run_gcn(X_train_np, y_train_np,
             X_test_np,  y_test_np,
             psn_train,
             matrices_train,
             survival_aware,
             times_train, events_train,
             times_test,  events_test,
             class_weights,
             label,
             modality_dims=None):
    """
    modality_dims : list[int] or None
        None  → plain concatenation (A1–A7)
        list  → per-modality encoders before GCN (A8)
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    n_tr  = X_train_np.shape[0]
    n_te  = X_test_np.shape[0]

    # Final adjacency for retraining
    full_adj_final = attach_test_nodes(psn_train, X_train_np, X_test_np, k=K_TEST)
    adj_final      = torch.tensor(normalise_adjacency(full_adj_final), dtype=torch.float)

    X_all_f = torch.tensor(np.vstack([X_train_np, X_test_np]),      dtype=torch.float)
    y_all_f = torch.tensor(np.concatenate([y_train_np, y_test_np]), dtype=torch.long)
    t_all_f = torch.tensor(np.concatenate([times_train, times_test]),    dtype=torch.float)
    e_all_f = torch.tensor(np.concatenate([events_train, events_test]),  dtype=torch.float)
    idx_test = torch.arange(n_tr, n_tr + n_te)

    # 5-fold CV with per-fold PSN rebuild
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs, fold_threshs, fold_epochs = [], [], []

    for tr_idx, val_idx in skf.split(X_train_np, y_train_np):
        fold_matrices = [m[tr_idx] for m in matrices_train]
        fold_psn = _build_psn(
            fold_matrices,
            times_train[tr_idx], events_train[tr_idx],
            survival_aware=survival_aware,
        )
        fold_X_tr  = X_train_np[tr_idx]
        fold_X_val = X_train_np[val_idx]
        n_fold_tr  = len(tr_idx)
        n_fold_val = len(val_idx)

        fold_adj = torch.tensor(
            normalise_adjacency(
                attach_test_nodes(fold_psn, fold_X_tr, fold_X_val, k=K_TEST)
            ), dtype=torch.float)

        X_fold = torch.tensor(np.vstack([fold_X_tr, fold_X_val]), dtype=torch.float)
        y_fold = torch.tensor(
            np.concatenate([y_train_np[tr_idx], y_train_np[val_idx]]),
            dtype=torch.long)
        t_fold = torch.tensor(
            np.concatenate([times_train[tr_idx], times_train[val_idx]]),
            dtype=torch.float)
        e_fold = torch.tensor(
            np.concatenate([events_train[tr_idx], events_train[val_idx]]),
            dtype=torch.float)

        idx_tr_f  = torch.arange(n_fold_tr)
        idx_val_f = torch.arange(n_fold_tr, n_fold_tr + n_fold_val)

        model = GCN(
            n_in=X_fold.shape[1], n_hid=HIDDEN_DIM, n_out=2, dropout=DROPOUT,
            modality_dims=modality_dims, enc_dim=ENC_DIM,
        )
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_auc, best_thresh, best_ep, bad = 0.0, 0.5, MIN_EPOCHS, 0

        for epoch in range(EPOCHS):
            model.train(); opt.zero_grad()
            bl, cr, _ = model(X_fold, fold_adj)
            l_bin = F.cross_entropy(bl[idx_tr_f], y_fold[idx_tr_f],
                                    weight=class_weights)
            l_cox = cox_partial_likelihood_loss(
                cr[idx_tr_f], t_fold[idx_tr_f], e_fold[idx_tr_f])
            (ALPHA_BIN * l_bin + ALPHA_COX * l_cox).backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                bl_all, _, _ = model(X_fold, fold_adj)
            vp = torch.softmax(bl_all[idx_val_f], dim=1)[:, 1].cpu().numpy()
            vt = y_fold[idx_val_f].cpu().numpy()
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

    # Final retrain
    idx_all_tr = torch.arange(n_tr)
    model = GCN(
        n_in=X_all_f.shape[1], n_hid=HIDDEN_DIM, n_out=2, dropout=DROPOUT,
        modality_dims=modality_dims, enc_dim=ENC_DIM,
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(final_epochs):
        model.train(); opt.zero_grad()
        bl, cr, _ = model(X_all_f, adj_final)
        l_bin = F.cross_entropy(bl[idx_all_tr], y_all_f[idx_all_tr],
                                weight=class_weights)
        l_cox = cox_partial_likelihood_loss(
            cr[idx_all_tr], t_all_f[idx_all_tr], e_all_f[idx_all_tr])
        (ALPHA_BIN * l_bin + ALPHA_COX * l_cox).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        bl_f, cr_f, _ = model(X_all_f, adj_final)

    test_probs = torch.softmax(bl_f[idx_test], dim=1)[:, 1].cpu().numpy()
    test_true  = y_all_f[idx_test].cpu().numpy()
    test_risk  = cr_f[idx_test].cpu().numpy()
    test_auc   = roc_auc_score(test_true, test_probs)
    cindex     = concordance_index(test_risk, times_test, events_test)

    print(f"  [{label:50s}]  AUC={test_auc:.4f}  C-index={cindex:.4f}  "
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
def _print_summary(results):
    print("\n" + "=" * 82)
    print("  ABLATION STUDY — SUMMARY TABLE")
    print("=" * 82)
    print(f"  {'Configuration':<52} {'AUC':>6}  {'C-index':>8}  {'CV AUC':>14}")
    print("  " + "-" * 80)
    for r in results:
        marker = " ← FULL MODEL" if "Full model" in r["label"] else ""
        print(f"  {r['label']:<52} {r['auc']:>6.4f}  "
              f"{r['cindex']:>8.4f}  "
              f"{r['cv_auc_mean']:>6.4f}±{r['cv_auc_std']:.4f}"
              f"{marker}")
    print("=" * 82)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(pipeline, output_dir="plots", gcn_results=None):
    """
    A1–A7: same as before (no modality encoders).
    A8: All 4 + SNF-SA + Modality Encoders = FULL MODEL.
         Reuses gcn_results from train_gcn() for consistency.
    """
    cna_tr  = pipeline["cna_tr_r"];   cna_te  = pipeline["cna_te"]
    mrna_tr = pipeline["mrna_tr_r"];  mrna_te = pipeline["mrna_te"]
    meth_tr = pipeline["meth_tr_r"];  meth_te = pipeline["meth_te"]
    clin_tr = pipeline["clin_tr_arr"]; clin_te = pipeline["clin_te"]

    y_tr = pipeline["y_train"].values
    y_te = pipeline["y_test"].values
    t_tr = pipeline["os_months_train"].values.astype(float)
    e_tr = pipeline["os_status_train"].values.astype(float)
    t_te = pipeline["os_months_test"].values.astype(float)
    e_te = pipeline["os_status_test"].values.astype(float)

    class_weights, n_lts, _ = compute_class_weights(y_tr)

    print("\n" + "=" * 82)
    print("  ABLATION STUDY  — 8 Configurations")
    print("=" * 82)
    print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")
    print(f"  LTS train={n_lts} | LTS test={int(y_te.sum())}")
    print(f"  Class weights: non-LTS={class_weights[0]:.3f}, LTS={class_weights[1]:.3f}")
    print(f"  Survival-aware alpha = {ALPHA_SURVIVAL}")
    print(f"  CV: per-fold PSN rebuild (no label leakage)")
    print(f"  A7 vs A8 isolates contribution of modality encoders (ENC_DIM={ENC_DIM})")
    print("=" * 82 + "\n")

    results = []
    kw = dict(times_train=t_tr, events_train=e_tr,
              times_test=t_te,  events_test=e_te,
              class_weights=class_weights)

    # A1 — CNA only ──────────────────────────────────────────────────────────
    print("── A1: CNA + Clinical ──────────────────────────────────────────")
    results.append(_run_gcn(
        cna_tr, y_tr, cna_te, y_te,
        _build_psn([cna_tr], t_tr, e_tr, True),
        matrices_train=[cna_tr], survival_aware=True,
        label="CNA only", **kw,
    ))

    # A2 — mRNA only ─────────────────────────────────────────────────────────
    print("── A2: mRNA + Clinical ─────────────────────────────────────────")
    results.append(_run_gcn(
        mrna_tr, y_tr, mrna_te, y_te,
        _build_psn([mrna_tr], t_tr, e_tr, True),
        matrices_train=[mrna_tr], survival_aware=True,
        label="mRNA only", **kw,
    ))

    # A3 — Methylation only ──────────────────────────────────────────────────
    print("── A3: Methylation + Clinical ──────────────────────────────────")
    results.append(_run_gcn(
        meth_tr, y_tr, meth_te, y_te,
        _build_psn([meth_tr], t_tr, e_tr, True),
        matrices_train=[meth_tr], survival_aware=True,
        label="Methylation only", **kw,
    ))

    # A4 — 3 Omics, no Clinical ──────────────────────────────────────────────
    print("── A4: 3 Omics only (no Clinical) ──────────────────────────────")
    X_tr4 = np.hstack([cna_tr, mrna_tr, meth_tr])
    X_te4 = np.hstack([cna_te, mrna_te, meth_te])
    results.append(_run_gcn(
        X_tr4, y_tr, X_te4, y_te,
        _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, True),
        matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
        label="3 Omics only (no Clinical)", **kw,
    ))

    # A5 — Clinical only ─────────────────────────────────────────────────────
    print("── A5: Clinical only ───────────────────────────────────────────")
    results.append(_run_gcn(
        clin_tr, y_tr, clin_te, y_te,
        _build_psn([clin_tr], t_tr, e_tr, True),
        matrices_train=[clin_tr], survival_aware=True,
        label="Clinical only", **kw,
    ))

    # A6 — All 4 + SNF omics-only (no survival-aware) ────────────────────────
    print("── A6: All 4 + SNF (no survival-aware) ─────────────────────────")
    X_tr6 = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
    X_te6 = np.hstack([cna_te, mrna_te, meth_te, clin_te])
    results.append(_run_gcn(
        X_tr6, y_tr, X_te6, y_te,
        _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, False),
        matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=False,
        label="All 4 + SNF (no surv-aware)", **kw,
    ))

    # A7 — All 4 + SNF-SA, NO modality encoders ──────────────────────────────
    print("── A7: All 4 + SNF-SA (no modality encoders) ───────────────────")
    X_tr7 = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
    X_te7 = np.hstack([cna_te, mrna_te, meth_te, clin_te])
    results.append(_run_gcn(
        X_tr7, y_tr, X_te7, y_te,
        _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, True),
        matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
        label="All 4 + SNF-SA (no encoders)",
        modality_dims=None,        # ← plain concatenation
        **kw,
    ))

    # A8 — All 4 + SNF-SA + Modality Encoders = FULL MODEL ───────────────────
    # Reuses gcn_results from train_gcn() so numbers are exactly consistent
    # with the main results table.
    print("── A8: All 4 + SNF-SA + Modality Encoders (FULL MODEL) ─────────")
    modality_dims = [cna_tr.shape[1], mrna_tr.shape[1],
                     meth_tr.shape[1], clin_tr.shape[1]]
    if gcn_results is not None:
        a8 = {
            "label":       "All 4 + SNF-SA + Encoders (Full model)",
            "auc":         gcn_results["auc"],
            "cindex":      gcn_results["cindex"],
            "probs":       gcn_results["probs"],
            "y_true":      gcn_results["y_true"],
            "cv_auc_mean": gcn_results["cv_val_auc_mean"],
            "cv_auc_std":  gcn_results["cv_val_auc_std"],
        }
        print(f"  [{'All 4 + SNF-SA + Encoders (Full model)':50s}]  "
              f"AUC={a8['auc']:.4f}  C-index={a8['cindex']:.4f}  "
              f"(CV AUC={a8['cv_auc_mean']:.4f}  "
              f"epochs={gcn_results['final_epochs_used']})")
    else:
        X_tr8 = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
        X_te8 = np.hstack([cna_te, mrna_te, meth_te, clin_te])
        a8 = _run_gcn(
            X_tr8, y_tr, X_te8, y_te,
            pipeline["psn_real"],
            matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
            label="All 4 + SNF-SA + Encoders (Full model)",
            modality_dims=modality_dims,
            **kw,
        )
    results.append(a8)

    _print_summary(results)
    plot_roc_curves(
        results,
        output_path=os.path.join(output_dir, "ablation_roc_curves.png"),
        title="GBM LTS Prediction — Ablation ROC Curves",
    )
    return results