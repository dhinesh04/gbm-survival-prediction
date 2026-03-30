"""
gcn_train.py
------------
Binary LTS classification + Cox survival head using GCN.

All hyperparameters are imported from config.py.

Data usage:
  Train      → model weight updates
  Validation → early stopping on combined loss (never touches test)
  Test       → final evaluation, touched exactly once

Cox head:
  Outputs a scalar risk score per patient.
  Trained with Cox partial likelihood loss.
  Risk scores used for KM stratification after training.

Joint loss:
  L = ALPHA_BIN * L_binary + ALPHA_COX * L_cox
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                             balanced_accuracy_score)
from lifelines.statistics import logrank_test
from gcn_model import GCN

from config import (
    HIDDEN_DIM, DROPOUT, LR, WEIGHT_DECAY,
    EPOCHS, PATIENCE, MIN_EPOCHS,
    N_FOLDS, ADJ_THRESHOLD, K_TEST,
    ALPHA_BIN, ALPHA_COX,
    RANDOM_STATE,
)
from utils import (
    concordance_index,
    cox_partial_likelihood_loss,
    normalise_adjacency,
    attach_test_nodes,
    find_best_threshold,
    compute_class_weights,
)


def get_binary_metrics(probs, y_true, threshold):
    preds    = (probs >= threshold).astype(int)
    auc      = roc_auc_score(y_true, probs)
    macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
    bal_acc  = balanced_accuracy_score(y_true, preds)
    acc      = (preds == y_true).mean()
    cm       = confusion_matrix(y_true, preds)
    return {
        "auc":           auc,
        "macro_f1":      macro_f1,
        "balanced_acc":  bal_acc,
        "accuracy":      acc,
        "recall_nonlts": cm[0,0]/cm[0].sum() if cm[0].sum()>0 else 0.0,
        "recall_lts":    cm[1,1]/cm[1].sum() if cm[1].sum()>0 else 0.0,
        "confusion":     cm,
        "threshold":     threshold,
    }


def _print_final(epochs_used, bin_m, cindex, risk_scores, times, events):
    print(f"\n── Final Test Results (epochs used: {epochs_used}) ────────────")
    print(f"\n  Binary head:")
    print(f"    AUC:              {bin_m['auc']:.4f}")
    print(f"    Macro F1:         {bin_m['macro_f1']:.4f}")
    print(f"    Balanced Acc:     {bin_m['balanced_acc']:.4f}")
    print(f"    Accuracy:         {bin_m['accuracy']:.4f}")
    print(f"    Recall (non-LTS): {bin_m['recall_nonlts']:.4f}")
    print(f"    Recall (LTS):     {bin_m['recall_lts']:.4f}")
    print(f"    Decision thresh:  {bin_m['threshold']:.2f}")
    print(f"\n    Confusion matrix (rows=true, cols=pred):")
    print(f"                  Pred non-LTS  Pred LTS")
    print(f"    True non-LTS:     {bin_m['confusion'][0,0]:3d}            {bin_m['confusion'][0,1]:3d}")
    print(f"    True LTS:         {bin_m['confusion'][1,0]:3d}            {bin_m['confusion'][1,1]:3d}")
    print(f"\n  Cox head:")
    print(f"    C-index:          {cindex:.4f}")
    print(f"    Risk score range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    print(f"    Risk score mean:  {risk_scores.mean():.3f} ± {risk_scores.std():.3f}")

    median_risk = np.median(risk_scores)
    high_mask   = risk_scores >= median_risk
    low_mask    = ~high_mask
    if high_mask.sum() > 0 and low_mask.sum() > 0:
        lr = logrank_test(
            times[high_mask],  times[low_mask],
            events[high_mask], events[low_mask]
        )
        print(f"\n  Median-split log-rank p-value: {lr.p_value:.4f}")
        if lr.p_value < 0.05:
            print("  ✓ Significant survival difference between risk groups")
        else:
            print("  ~ No significant survival difference (p >= 0.05)")
    print("────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FOLD TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _train_one_fold(X_all, y_all, adj,
                    times_all, events_all,
                    idx_fold_train, idx_fold_val,
                    class_weights,
                    fold_num):
    model = GCN(n_in=X_all.shape[1], n_hid=HIDDEN_DIM, n_out=2, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    best_val_auc  = 0.0
    best_thresh   = 0.5
    best_epoch    = MIN_EPOCHS
    bad_counter   = 0
    best_state    = None

    for epoch in range(EPOCHS):
        # ── Train ────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        bin_logits, cox_risk, _ = model(X_all, adj)

        l_bin = F.cross_entropy(
            bin_logits[idx_fold_train], y_all[idx_fold_train],
            weight=class_weights
        )
        tr_events = events_all[idx_fold_train]
        tr_times  = times_all[idx_fold_train]
        tr_risk   = cox_risk[idx_fold_train]
        l_cox  = cox_partial_likelihood_loss(tr_risk, tr_times, tr_events)
        loss   = ALPHA_BIN * l_bin + ALPHA_COX * l_cox
        loss.backward()
        optimizer.step()

        # ── Validation ───────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            bin_logits_all, cox_risk_all, _ = model(X_all, adj)

        val_probs = torch.softmax(
            bin_logits_all[idx_fold_val], dim=1)[:, 1].cpu().numpy()
        val_true  = y_all[idx_fold_val].cpu().numpy()

        if len(np.unique(val_true)) < 2:
            continue

        val_auc = roc_auc_score(val_true, val_probs)
        thresh  = find_best_threshold(val_probs, val_true)

        val_l_bin = F.cross_entropy(
            bin_logits_all[idx_fold_val], y_all[idx_fold_val],
            weight=class_weights
        )
        val_l_cox = cox_partial_likelihood_loss(
            cox_risk_all[idx_fold_val],
            times_all[idx_fold_val],
            events_all[idx_fold_val]
        )
        val_loss = (ALPHA_BIN * val_l_bin + ALPHA_COX * val_l_cox).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc  = val_auc
            best_thresh   = thresh
            if epoch + 1 >= MIN_EPOCHS:
                best_epoch  = epoch + 1
            bad_counter   = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            if epoch + 1 >= MIN_EPOCHS:
                bad_counter += 1
                if bad_counter >= PATIENCE:
                    break

    return best_val_auc, best_thresh, best_epoch, best_state


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION + FINAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def run_cross_validation(X_train_np, y_train_np,
                         X_test_np,  y_test_np,
                         psn_train,
                         times_train_np, events_train_np,
                         times_test_np,  events_test_np,
                         class_weights):
    n_train = X_train_np.shape[0]
    n_test  = X_test_np.shape[0]
    n_total = n_train + n_test

    full_adj = attach_test_nodes(psn_train, X_train_np, X_test_np, k=K_TEST)
    adj      = normalise_adjacency(full_adj, threshold=ADJ_THRESHOLD)

    X_all = torch.tensor(np.vstack([X_train_np, X_test_np]), dtype=torch.float)
    y_all = torch.tensor(np.concatenate([y_train_np, y_test_np]), dtype=torch.long)
    times_all  = torch.tensor(
        np.concatenate([times_train_np,  times_test_np]),  dtype=torch.float)
    events_all = torch.tensor(
        np.concatenate([events_train_np, events_test_np]), dtype=torch.float)

    idx_test = torch.arange(n_train, n_total)

    # ── K-Fold CV ─────────────────────────────────────────────────
    lts_per_fold = int(np.sum(y_train_np == 1)) // N_FOLDS
    print(f"\n── {N_FOLDS}-Fold CV (train={n_train}, test={n_test} held out) ──")
    print(f"  ~{lts_per_fold} LTS per val fold | "
          f"MIN_EPOCHS={MIN_EPOCHS} | PATIENCE={PATIENCE}")
    print(f"  Threshold selection: macro F1 (range 0.20-0.75)")
    print(f"  Joint loss: {ALPHA_BIN}xBCE + {ALPHA_COX}xCoxLoss\n")
    print(f"  Fold | Val AUC | Thresh | Best Epoch  | LTS in val")
    print("  " + "─" * 50)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_val_aucs   = []
    fold_thresholds = []
    fold_epochs     = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_np, y_train_np)):
        idx_fold_train = torch.tensor(tr_idx,  dtype=torch.long)
        idx_fold_val   = torch.tensor(val_idx, dtype=torch.long)

        val_auc, thresh, best_ep, _ = _train_one_fold(
            X_all, y_all, adj,
            times_all, events_all,
            idx_fold_train, idx_fold_val,
            class_weights,
            fold_num=fold+1
        )

        val_lts = int(y_train_np[val_idx].sum())
        fold_val_aucs.append(val_auc)
        fold_thresholds.append(thresh)
        fold_epochs.append(best_ep)

        print(f"  {fold+1:4d} | {val_auc:.4f}  | {thresh:.2f}   "
              f"| {best_ep:>10}  | {val_lts}")

    mean_val_auc = float(np.mean(fold_val_aucs))
    std_val_auc  = float(np.std(fold_val_aucs))
    final_thresh = float(np.median(fold_thresholds))
    final_epochs = max(MIN_EPOCHS, int(np.median(fold_epochs)))

    print(f"\n  CV Val AUC:      {mean_val_auc:.4f} ± {std_val_auc:.4f}")
    print(f"  Final epochs:    {final_epochs}")
    print(f"  Final threshold: {final_thresh:.2f}")

    # ── Final retraining on all training patients ─────────────────
    print(f"\n── Final retraining on all {n_train} patients "
          f"({final_epochs} epochs) ──")

    idx_all_train = torch.arange(n_train)

    model = GCN(n_in=X_all.shape[1], n_hid=HIDDEN_DIM, n_out=2, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(final_epochs):
        model.train()
        optimizer.zero_grad()
        bin_logits, cox_risk, _ = model(X_all, adj)

        l_bin = F.cross_entropy(
            bin_logits[idx_all_train], y_all[idx_all_train],
            weight=class_weights
        )
        tr_events = events_all[idx_all_train]
        tr_times  = times_all[idx_all_train]
        tr_risk   = cox_risk[idx_all_train]
        l_cox  = cox_partial_likelihood_loss(tr_risk, tr_times, tr_events)
        loss   = ALPHA_BIN * l_bin + ALPHA_COX * l_cox
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{final_epochs} | "
                  f"loss={loss.item():.4f} "
                  f"(bin={l_bin.item():.4f}, cox={l_cox.item():.4f})")

    # ── Final test evaluation — exactly once ─────────────────────
    print("\n── Test Evaluation (touched exactly once) ───────────────────")
    model.eval()
    with torch.no_grad():
        bin_logits_all, cox_risk_all, _ = model(X_all, adj)

    test_probs  = torch.softmax(
        bin_logits_all[idx_test], dim=1)[:, 1].cpu().numpy()
    test_true   = y_all[idx_test].cpu().numpy()
    bin_metrics = get_binary_metrics(test_probs, test_true, final_thresh)

    test_risk   = cox_risk_all[idx_test].cpu().numpy()
    test_times  = times_test_np
    test_events = events_test_np
    cindex      = concordance_index(test_risk, test_times, test_events)

    _print_final(final_epochs, bin_metrics, cindex,
                 test_risk, test_times, test_events)

    print(f"\n  CV Val AUC (mean±std): {mean_val_auc:.4f} ± {std_val_auc:.4f}")
    print(f"  Final Test AUC:        {bin_metrics['auc']:.4f}")
    gap  = abs(bin_metrics['auc'] - mean_val_auc)
    flag = ("✓ well-calibrated" if gap < 0.05 else
            "~ acceptable"      if gap < 0.10 else "✗ large gap")
    print(f"  Gap (|test-val|):      {gap:.4f}  {flag}")

    return {
        **bin_metrics,
        "probs":             test_probs,
        "y_true":            test_true,
        "cindex":            cindex,
        "risk_scores":       test_risk,
        "times_test":        test_times,
        "events_test":       test_events,
        "cv_val_auc_mean":   mean_val_auc,
        "cv_val_auc_std":    std_val_auc,
        "cv_val_aucs":       fold_val_aucs,
        "cv_thresholds":     fold_thresholds,
        "cv_best_epochs":    fold_epochs,
        "final_threshold":   final_thresh,
        "final_epochs_used": final_epochs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def train_gcn(results: dict) -> dict:
    """
    Train the GCN dual-head model.

    Class weights are computed DYNAMICALLY from the actual training labels
    so they remain correct for every LTS threshold experiment (12 / 18 / 24m).
    Previously these were hardcoded constants which would have been wrong
    for the 12m and 18m cohorts.

    Parameters
    ----------
    results : dict returned by main.main()
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_train_np = np.hstack([
        results["cna_tr_r"],   results["mrna_tr_r"],
        results["meth_tr_r"],  results["clin_tr_arr"]
    ])
    X_test_np = np.hstack([
        results["cna_te"],  results["mrna_te"],
        results["meth_te"], results["clin_te"]
    ])
    y_train_np = results["y_train"].values
    y_test_np  = results["y_test"].values

    times_train  = results["os_months_train"].values.astype(float)
    events_train = results["os_status_train"].values.astype(float)
    times_test   = results["os_months_test"].values.astype(float)
    events_test  = results["os_status_test"].values.astype(float)

    # ── Dynamic class weights (inverse frequency) ─────────────────────────────
    # Delegated to utils.compute_class_weights() — same formula, one place.
    class_weights, n_lts, n_nonlts = compute_class_weights(y_train_np)

    print("=" * 62)
    print("  GCN — Binary + Cox Dual Head | Train/Val/Test Protocol")
    print("=" * 62)
    print(f"  Train: {len(y_train_np)} patients  "
          f"(LTS={n_lts}, non-LTS={n_nonlts})")
    print(f"  Test:  {len(y_test_np)} patients  "
          f"(LTS={int(y_test_np.sum())}, "
          f"non-LTS={int((y_test_np==0).sum())})")
    print(f"  Events in train: {int(events_train.sum())} deceased, "
          f"{int((events_train==0).sum())} censored")
    print(f"  Events in test:  {int(events_test.sum())} deceased, "
          f"{int((events_test==0).sum())} censored")
    print(f"  GCN: {X_train_np.shape[1]} → {HIDDEN_DIM} → {HIDDEN_DIM} "
          f"→ [2 (binary), 1 (cox)]")
    print(f"  Class weights (dynamic): non-LTS={class_weights[0]:.3f}, LTS={class_weights[1]:.3f}")
    print(f"  CV: {N_FOLDS}-fold | MIN_EPOCHS={MIN_EPOCHS} | PATIENCE={PATIENCE}")

    return run_cross_validation(
        X_train_np, y_train_np,
        X_test_np,  y_test_np,
        results["psn_real"],
        times_train, events_train,
        times_test,  events_test,
        class_weights,
    )