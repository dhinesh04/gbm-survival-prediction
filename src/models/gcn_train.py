"""
gcn_train.py
------------
Log-normal AFT regression + Cox survival dual-head GCN.

Changes in this version:
  - AFT head is now a proper log-normal AFT: log(T) ~ Normal(mu(x), sigma^2),
    where mu(x) = fc_reg(H) (per-patient) and sigma = exp(model.log_sigma)
    is a single GLOBAL scalar learned jointly with the network. Loss is the
    true negative log-likelihood (PDF for events, survival function for
    censored) — see src/utils.py: aft_loss(). This replaces the earlier
    fixed-variance squared-error approximation.
  - alpha_AFT / alpha_COX are no longer fixed config values. gcn_train.py
    runs a NESTED grid search: for every (alpha_aft, alpha_cox) candidate
    in ALPHA_AFT_GRID x ALPHA_COX_GRID, a full 5-fold CV is run; the pair
    with the best SELECTION_METRIC (default: lowest CV MAE) is selected,
    then used for the final retrain on all training patients.
  - NO LTS / threshold logic anywhere in this file. 5-fold CV is stratified
    on event status (deceased vs censored), not on any survival-time cutoff.
    threshold_sensitivity() and all AUC-at-threshold code has been removed.
  - Per-fold PSN rebuild retained (CV leakage fix).
  - Modality-aware encoders retained.
"""

import math
import itertools
import numpy as np
import pandas as pd
import torch
import snf as snflib
from sklearn.model_selection import StratifiedKFold
from lifelines.statistics import logrank_test
from src.models.gcn_model import GCN
from src.graph.survival_aware_psn import build_survival_aware_psn

from config import (
    HIDDEN_DIM, DROPOUT, LR, WEIGHT_DECAY,
    EPOCHS, PATIENCE, MIN_EPOCHS,
    N_FOLDS, ADJ_THRESHOLD, K_TEST,
    ALPHA_AFT_GRID, ALPHA_COX_GRID, SELECTION_METRIC,
    RANDOM_STATE,
    K_SNF, MU_SNF, N_ITER_SNF,
    ALPHA_SURVIVAL,
    ENC_DIM,
)
from src.utils import (
    concordance_index,
    cox_partial_likelihood_loss,
    aft_loss,
    normalise_adjacency,
    attach_test_nodes,
)


# ─────────────────────────────────────────────────────────────────────────────
# PSN BUILD HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _build_fold_psn(cna, mrna, meth, os_months, os_status):
    """Build survival-aware PSN on fold-train patients only (CV leakage fix)."""
    affinities = [snflib.make_affinity(m, K=K_SNF, mu=MU_SNF)
                  for m in [cna, mrna, meth]]
    psn_omics  = snflib.snf(affinities, K=K_SNF, t=N_ITER_SNF).astype(np.float32)
    if ALPHA_SURVIVAL > 0:
        psn_sa, _ = build_survival_aware_psn(
            psn_omics, os_months=os_months, os_status=os_status,
            alpha=ALPHA_SURVIVAL, sigma=None,
        )
        return psn_sa
    return psn_omics


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION METRICS  (threshold-free — primary evaluation of the AFT head)
# ─────────────────────────────────────────────────────────────────────────────
def get_regression_metrics(pred_months: np.ndarray,
                           times: np.ndarray,
                           events: np.ndarray) -> dict:
    """
    Evaluate AFT regression head predictions. No threshold required.

    Parameters
    ----------
    pred_months : (N,) predicted survival months  [exp of mu = log t-hat]
    times       : (N,) actual survival / censoring months
    events      : (N,) 1 = deceased, 0 = censored

    Returns
    -------
    dict of metric → value
    """
    uncensored = events == 1

    if uncensored.sum() > 0:
        errors    = np.abs(pred_months[uncensored] - times[uncensored])
        mae       = float(errors.mean())
        rmse      = float(np.sqrt(
            ((pred_months[uncensored] - times[uncensored]) ** 2).mean()))
        median_ae = float(np.median(errors))
        corr      = float(np.corrcoef(
            pred_months[uncensored], times[uncensored])[0, 1])
    else:
        mae = rmse = median_ae = corr = float('nan')

    return {
        "mae":       mae,
        "rmse":      rmse,
        "median_ae": median_ae,
        "pred_corr": corr,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PER-PATIENT PREDICTION TABLE  (threshold-free — no binary label involved)
# ─────────────────────────────────────────────────────────────────────────────
def get_patient_predictions(results: dict, gcn_results: dict) -> pd.DataFrame:
    """
    Build a per-patient comparison of AFT-predicted survival time against
    actual OS_MONTHS, for the held-out test set. No threshold or binary
    label is used anywhere in this function.

    Row order assumption: gcn_results["pred_months"] / ["times_test"] /
    ["events_test"] preserve the same patient order as
    results["os_months_test"] — true throughout this pipeline since mRMR
    and PSN construction never reorder rows, only select/transform columns.

    Returns
    -------
    pd.DataFrame indexed by PATIENT_ID, columns:
        predicted_months     : exp(mu) from the AFT head
        actual_months        : OS_MONTHS (exact if deceased, lower bound if censored)
        event                : 1 = deceased (exact ground truth), 0 = censored (lower bound only)
        signed_error_months  : predicted - actual (sign shows over/under-prediction)
        abs_error_months     : |signed_error|, NaN for censored patients
        censoring_consistent : for censored patients only — True if predicted_months
                                >= actual_months, NaN for deceased patients
        risk_score            : Cox head output, included for cross-reference only
    """
    patient_ids = results["os_months_test"].index

    df = pd.DataFrame({
        "predicted_months": gcn_results["pred_months"],
        "actual_months":    gcn_results["times_test"],
        "event":            gcn_results["events_test"].astype(int),
        "risk_score":       gcn_results["risk_scores"],
    }, index=patient_ids)

    df["signed_error_months"] = df["predicted_months"] - df["actual_months"]

    deceased = df["event"] == 1
    df["abs_error_months"] = np.where(deceased, df["signed_error_months"].abs(), np.nan)
    df["censoring_consistent"] = np.where(
        ~deceased, df["predicted_months"] >= df["actual_months"], np.nan
    )

    return df.sort_values("abs_error_months", ascending=False, na_position="last")


def print_patient_predictions(df: pd.DataFrame, n: int = 20):
    """Print the n deceased patients with the largest absolute error."""
    deceased_df = df[df["event"] == 1]
    censored_df = df[df["event"] == 0]

    print(f"\n── Per-Patient Predictions — Largest errors "
          f"(top {n} of {len(deceased_df)} deceased) ──────────")
    print(f"  {'Patient ID':<14} {'Pred (m)':>9} {'Actual (m)':>11} "
          f"{'Abs Err (m)':>12} {'Risk score':>11}")
    print("  " + "-" * 62)
    for pid, row in deceased_df.head(n).iterrows():
        print(f"  {pid:<14} {row['predicted_months']:>9.1f} "
              f"{row['actual_months']:>11.1f} "
              f"{row['abs_error_months']:>12.1f} "
              f"{row['risk_score']:>11.3f}")

    n_bad_censored = int((~censored_df["censoring_consistent"].astype(bool)).sum())
    print(f"\n  Censored patients: {len(censored_df)} total, "
          f"{n_bad_censored} with predicted time SHORTER than observed "
          f"censoring time (logically inconsistent)")
    print("  " + "-" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _print_final(epochs_used, reg_m, cindex, cindex_aft,
                 risk_scores, times, events, sigma):
    print(f"\n── Final Test Results (epochs used: {epochs_used}) ────────────")
    print(f"\n  AFT Regression head (log-normal, learned σ={sigma:.4f}):")
    print(f"    MAE  (uncensored):  {reg_m['mae']:.2f} months")
    print(f"    RMSE (uncensored):  {reg_m['rmse']:.2f} months")
    print(f"    Median AE:          {reg_m['median_ae']:.2f} months")
    print(f"    Pearson r (uncens): {reg_m['pred_corr']:.4f}")
    print(f"    C-index (AFT):      {cindex_aft:.4f}")
    print(f"\n  Cox head:")
    print(f"    C-index:            {cindex:.4f}")
    print(f"    Risk score range:   [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    print(f"    Risk score mean:    {risk_scores.mean():.3f} ± {risk_scores.std():.3f}")

    median_risk = np.median(risk_scores)
    high_mask   = risk_scores >= median_risk
    low_mask    = ~high_mask
    if high_mask.sum() > 0 and low_mask.sum() > 0:
        lr = logrank_test(
            times[high_mask],  times[low_mask],
            events[high_mask], events[low_mask],
        )
        print(f"\n  Median-split log-rank p-value: {lr.p_value:.4f}")
        if lr.p_value < 0.05:
            print("  ✓ Significant survival difference between risk groups")
        else:
            print("  ~ No significant survival difference (p >= 0.05)")
    print("────────────────────────────────────────────────────────────")


def _print_grid_summary(grid_results: list, selection_metric: str):
    """Print every (alpha_aft, alpha_cox) candidate, sorted best-first."""
    print(f"\n── α Grid Search — Summary (sorted by {selection_metric}) "
          f"────────────────────")
    if selection_metric == "cindex":
        ordered = sorted(grid_results, key=lambda r: r["cv_cindex_mean"], reverse=True)
    else:
        ordered = sorted(grid_results, key=lambda r: r["cv_mae_mean"])

    print(f"  {'α_AFT':>6} {'α_Cox':>6}  {'CV MAE (m)':>14}  "
          f"{'CV C-index':>14}  {'epochs':>7}")
    print("  " + "-" * 58)
    for i, r in enumerate(ordered):
        marker = "  ★ best" if i == 0 else ""
        print(f"  {r['alpha_aft']:>6} {r['alpha_cox']:>6}  "
              f"{r['cv_mae_mean']:>6.2f}±{r['cv_mae_std']:<5.2f} "
              f"  {r['cv_cindex_mean']:>6.4f}±{r['cv_cindex_std']:<5.4f} "
              f" {r['median_epochs']:>7}{marker}")
    print("  " + "-" * 58)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FOLD TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _train_one_fold(X_all, adj, times_all, events_all,
                    idx_fold_train, idx_fold_val,
                    alpha_aft, alpha_cox,
                    modality_dims=None):
    """
    Train one CV fold with AFT (log-normal NLL) + Cox joint loss, for one
    (alpha_aft, alpha_cox) candidate pair.

    Model selection within the fold: minimises combined validation loss
    (alpha_aft * AFT_NLL + alpha_cox * Cox_loss).

    Returns
    -------
    best_val_mae    : float   MAE at the best-val-loss epoch
    best_val_cindex : float   Cox C-index at the best-val-loss epoch
    best_epoch      : int     epoch number of the best val state
    """
    model = GCN(
        n_in=X_all.shape[1], n_hid=HIDDEN_DIM, dropout=DROPOUT,
        modality_dims=modality_dims, enc_dim=ENC_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss   = float('inf')
    best_val_mae    = float('inf')
    best_val_cindex = 0.5
    best_epoch      = MIN_EPOCHS
    bad_counter     = 0

    for epoch in range(EPOCHS):
        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        optimizer.zero_grad()
        pred_log_t, cox_risk, _ = model(X_all, adj)

        l_aft = aft_loss(
            pred_log_t[idx_fold_train],
            times_all[idx_fold_train],
            events_all[idx_fold_train],
            model.log_sigma,
        )
        l_cox = cox_partial_likelihood_loss(
            cox_risk[idx_fold_train],
            times_all[idx_fold_train],
            events_all[idx_fold_train],
        )
        loss = alpha_aft * l_aft + alpha_cox * l_cox
        loss.backward()
        optimizer.step()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            pred_log_t_all, cox_risk_all, _ = model(X_all, adj)

        val_pred_log_t = pred_log_t_all[idx_fold_val]
        val_times      = times_all[idx_fold_val]
        val_events     = events_all[idx_fold_val]
        val_risk       = cox_risk_all[idx_fold_val]

        val_l_aft = aft_loss(val_pred_log_t, val_times, val_events, model.log_sigma)
        val_l_cox = cox_partial_likelihood_loss(val_risk, val_times, val_events)
        val_loss  = (alpha_aft * val_l_aft + alpha_cox * val_l_cox).item()

        val_pred_months = torch.exp(val_pred_log_t).detach().cpu().numpy()
        val_times_np    = val_times.cpu().numpy()
        val_events_np   = val_events.cpu().numpy()
        val_risk_np     = val_risk.detach().cpu().numpy()

        uncensored_val = val_events_np == 1
        if uncensored_val.sum() > 0:
            val_mae = float(np.abs(
                val_pred_months[uncensored_val] - val_times_np[uncensored_val]
            ).mean())
        else:
            val_mae = float('inf')
        val_cindex = concordance_index(val_risk_np, val_times_np, val_events_np)

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_val_mae    = val_mae
            best_val_cindex = val_cindex
            if epoch + 1 >= MIN_EPOCHS:
                best_epoch = epoch + 1
            bad_counter = 0
        else:
            if epoch + 1 >= MIN_EPOCHS:
                bad_counter += 1
                if bad_counter >= PATIENCE:
                    break

    return best_val_mae, best_val_cindex, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION + NESTED ALPHA GRID SEARCH + FINAL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def run_cross_validation(X_train_np, X_test_np,
                         psn_train,
                         cna_tr, mrna_tr, meth_tr,
                         times_train_np, events_train_np,
                         times_test_np,  events_test_np,
                         modality_dims=None):
    """
    Nested grid search over (alpha_aft, alpha_cox) via 5-fold CV per
    candidate, then final retrain + test evaluation using the winning pair.

    CV folds are stratified on event status (deceased vs censored) — no
    LTS label or survival-time threshold is used anywhere in this function.
    """
    n_train = X_train_np.shape[0]
    n_test  = X_test_np.shape[0]
    n_total = n_train + n_test

    strat_labels = events_train_np.astype(int)   # stratify on event status only

    full_adj_final = attach_test_nodes(psn_train, X_train_np, X_test_np, k=K_TEST)
    adj_final      = normalise_adjacency(full_adj_final, threshold=ADJ_THRESHOLD)

    X_all_final      = torch.tensor(np.vstack([X_train_np, X_test_np]), dtype=torch.float)
    times_all_final  = torch.tensor(
        np.concatenate([times_train_np, times_test_np]),  dtype=torch.float)
    events_all_final = torch.tensor(
        np.concatenate([events_train_np, events_test_np]), dtype=torch.float)
    idx_test = torch.arange(n_train, n_total)

    n_candidates = len(ALPHA_AFT_GRID) * len(ALPHA_COX_GRID)
    enc_info = (f"modality encoders {modality_dims}→{ENC_DIM}each"
                if modality_dims else "plain concatenation (no encoders)")
    print(f"\n── Nested α Grid Search: {len(ALPHA_AFT_GRID)}×{len(ALPHA_COX_GRID)} "
          f"= {n_candidates} candidates × {N_FOLDS}-fold CV ──")
    print(f"  α_AFT grid: {ALPHA_AFT_GRID}")
    print(f"  α_Cox grid: {ALPHA_COX_GRID}")
    print(f"  Selection metric: {SELECTION_METRIC}")
    print(f"  CV stratified by event status (deceased vs censored) — "
          f"no LTS label or threshold used anywhere")
    print(f"  Per-fold PSN: rebuilt on fold-train only → val attached via k-NN")
    print(f"  Node features: {enc_info}")
    print(f"  MIN_EPOCHS={MIN_EPOCHS} | PATIENCE={PATIENCE}\n")

    grid_results = []

    for alpha_aft, alpha_cox in itertools.product(ALPHA_AFT_GRID, ALPHA_COX_GRID):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_maes, fold_cindices, fold_epochs = [], [], []

        for tr_idx, val_idx in skf.split(X_train_np, strat_labels):
            fold_psn = _build_fold_psn(
                cna_tr[tr_idx], mrna_tr[tr_idx], meth_tr[tr_idx],
                os_months=times_train_np[tr_idx],
                os_status=events_train_np[tr_idx],
            )

            fold_X_tr  = X_train_np[tr_idx]
            fold_X_val = X_train_np[val_idx]
            n_fold_tr  = len(tr_idx)
            n_fold_val = len(val_idx)

            fold_full_adj = attach_test_nodes(fold_psn, fold_X_tr, fold_X_val, k=K_TEST)
            fold_adj_t    = normalise_adjacency(fold_full_adj, threshold=ADJ_THRESHOLD)

            X_fold_all = torch.tensor(
                np.vstack([fold_X_tr, fold_X_val]), dtype=torch.float)
            times_fold_all = torch.tensor(
                np.concatenate([times_train_np[tr_idx], times_train_np[val_idx]]),
                dtype=torch.float)
            events_fold_all = torch.tensor(
                np.concatenate([events_train_np[tr_idx], events_train_np[val_idx]]),
                dtype=torch.float)

            idx_fold_train = torch.arange(n_fold_tr)
            idx_fold_val_t = torch.arange(n_fold_tr, n_fold_tr + n_fold_val)

            val_mae, val_cindex, best_ep = _train_one_fold(
                X_fold_all, fold_adj_t,
                times_fold_all, events_fold_all,
                idx_fold_train, idx_fold_val_t,
                alpha_aft=alpha_aft, alpha_cox=alpha_cox,
                modality_dims=modality_dims,
            )

            fold_maes.append(val_mae)
            fold_cindices.append(val_cindex)
            fold_epochs.append(best_ep)

        mean_mae, std_mae = float(np.mean(fold_maes)), float(np.std(fold_maes))
        mean_ci,  std_ci  = float(np.mean(fold_cindices)), float(np.std(fold_cindices))
        median_epochs     = max(MIN_EPOCHS, int(np.median(fold_epochs)))

        grid_results.append({
            "alpha_aft": alpha_aft, "alpha_cox": alpha_cox,
            "cv_mae_mean": mean_mae, "cv_mae_std": std_mae,
            "cv_cindex_mean": mean_ci, "cv_cindex_std": std_ci,
            "median_epochs": median_epochs,
        })
        print(f"  α_AFT={alpha_aft:<5} α_Cox={alpha_cox:<5}  "
              f"CV MAE={mean_mae:6.2f}±{std_mae:4.2f}m  "
              f"CV C-idx={mean_ci:.4f}±{std_ci:.4f}  epochs≈{median_epochs}")

    _print_grid_summary(grid_results, SELECTION_METRIC)

    if SELECTION_METRIC == "cindex":
        best = max(grid_results, key=lambda r: r["cv_cindex_mean"])
    else:
        best = min(grid_results, key=lambda r: r["cv_mae_mean"])

    alpha_aft_best = best["alpha_aft"]
    alpha_cox_best = best["alpha_cox"]
    final_epochs   = best["median_epochs"]

    print(f"\n  ★ Selected: α_AFT={alpha_aft_best}  α_Cox={alpha_cox_best}  "
          f"(CV MAE={best['cv_mae_mean']:.2f}±{best['cv_mae_std']:.2f}m, "
          f"CV C-idx={best['cv_cindex_mean']:.4f}±{best['cv_cindex_std']:.4f})")

    # ── Final retraining on all training patients, winning alphas ──────────
    print(f"\n── Final retraining on all {n_train} patients "
          f"({final_epochs} epochs, α_AFT={alpha_aft_best}, α_Cox={alpha_cox_best}) ──")

    model = GCN(
        n_in=X_all_final.shape[1], n_hid=HIDDEN_DIM, dropout=DROPOUT,
        modality_dims=modality_dims, enc_dim=ENC_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    idx_all_train = torch.arange(n_train)

    for epoch in range(final_epochs):
        model.train()
        optimizer.zero_grad()
        pred_log_t, cox_risk, _ = model(X_all_final, adj_final)

        l_aft = aft_loss(
            pred_log_t[idx_all_train],
            times_all_final[idx_all_train],
            events_all_final[idx_all_train],
            model.log_sigma,
        )
        l_cox = cox_partial_likelihood_loss(
            cox_risk[idx_all_train],
            times_all_final[idx_all_train],
            events_all_final[idx_all_train],
        )
        loss = alpha_aft_best * l_aft + alpha_cox_best * l_cox
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{final_epochs} | "
                  f"loss={loss.item():.4f} "
                  f"(aft={l_aft.item():.4f}, cox={l_cox.item():.4f}, "
                  f"σ={model.sigma().item():.4f})")

    # ── Test evaluation (touched exactly once) ─────────────────────────────
    print("\n── Test Evaluation (touched exactly once) ───────────────────")
    model.eval()
    with torch.no_grad():
        pred_log_t_all, cox_risk_all, _ = model(X_all_final, adj_final)

    test_pred_log_t  = pred_log_t_all[idx_test].cpu().numpy()
    test_pred_months = np.exp(test_pred_log_t)
    test_times       = times_test_np
    test_events      = events_test_np

    reg_metrics = get_regression_metrics(test_pred_months, test_times, test_events)

    test_risk   = cox_risk_all[idx_test].cpu().numpy()
    cindex      = concordance_index(test_risk, test_times, test_events)
    cindex_aft  = concordance_index(-test_pred_log_t, test_times, test_events)
    final_sigma = float(model.sigma().item())

    _print_final(final_epochs, reg_metrics, cindex, cindex_aft,
                 test_risk, test_times, test_events, final_sigma)

    print(f"\n  Learned σ (log-normal AFT scale): {final_sigma:.4f}")
    print(f"  CV MAE (winning pair, mean±std): "
          f"{best['cv_mae_mean']:.2f} ± {best['cv_mae_std']:.2f} months")
    print(f"  Final Test MAE:                  {reg_metrics['mae']:.2f} months")
    gap  = abs(reg_metrics['mae'] - best['cv_mae_mean'])
    flag = ("✓ well-calibrated" if gap <  5.0 else
            "~ acceptable"      if gap < 10.0 else "✗ large gap")
    print(f"  Gap (|test-val|):                {gap:.2f} months  {flag}")

    return {
        **reg_metrics,
        "pred_log_t":          test_pred_log_t,
        "pred_months":         test_pred_months,
        "cindex":              cindex,
        "cindex_aft":          cindex_aft,
        "risk_scores":         test_risk,
        "times_test":          test_times,
        "events_test":         test_events,
        "sigma":               final_sigma,
        "alpha_aft":           alpha_aft_best,
        "alpha_cox":           alpha_cox_best,
        "grid_search_results": grid_results,
        "cv_val_mae_mean":     best["cv_mae_mean"],
        "cv_val_mae_std":      best["cv_mae_std"],
        "cv_val_cindex_mean":  best["cv_cindex_mean"],
        "cv_val_cindex_std":   best["cv_cindex_std"],
        "final_epochs_used":   final_epochs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def train_gcn(results: dict) -> dict:
    """
    Train the GCN dual-head model (log-normal AFT regression + Cox), with
    a nested grid search over the joint-loss weights (alpha_aft, alpha_cox).

    No LTS label or threshold is read from `results` — only OS_MONTHS,
    OS_STATUS, and the mRMR-reduced feature matrices are used. CV
    stratification is on event status (deceased vs censored).
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    cna_tr  = results["cna_tr_r"]
    mrna_tr = results["mrna_tr_r"]
    meth_tr = results["meth_tr_r"]
    clin_tr = results["clin_tr_arr"]

    X_train_np = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
    X_test_np  = np.hstack([
        results["cna_te"], results["mrna_te"],
        results["meth_te"], results["clin_te"],
    ])

    modality_dims = [
        cna_tr.shape[1],
        mrna_tr.shape[1],
        meth_tr.shape[1],
        clin_tr.shape[1],
    ]
    gcn_in = ENC_DIM * len(modality_dims)

    times_train  = results["os_months_train"].values.astype(float)
    events_train = results["os_status_train"].values.astype(float)
    times_test   = results["os_months_test"].values.astype(float)
    events_test  = results["os_status_test"].values.astype(float)

    n_candidates = len(ALPHA_AFT_GRID) * len(ALPHA_COX_GRID)

    print("=" * 70)
    print("  GCN — Log-normal AFT Regression + Cox Dual Head")
    print("  Nested α Grid Search | No LTS / threshold logic anywhere")
    print("=" * 70)
    print(f"  Train: {len(times_train)} patients   "
          f"Test: {len(times_test)} patients")
    print(f"  Events in train: {int(events_train.sum())} deceased, "
          f"{int((events_train==0).sum())} censored")
    print(f"  Events in test:  {int(events_test.sum())} deceased, "
          f"{int((events_test==0).sum())} censored")
    print(f"  Architecture: {X_train_np.shape[1]} raw → "
          f"ModalityEncoders({modality_dims}→{ENC_DIM}each) → "
          f"{gcn_in} → {HIDDEN_DIM} → {HIDDEN_DIM} → "
          f"[AFT(mu=log t̂, learned global σ), Cox(risk)]")
    print(f"  CV: {N_FOLDS}-fold (stratified on event status) | "
          f"MIN_EPOCHS={MIN_EPOCHS} | PATIENCE={PATIENCE}")
    print(f"  α grid: {n_candidates} combinations "
          f"(AFT={ALPHA_AFT_GRID}, Cox={ALPHA_COX_GRID})")

    return run_cross_validation(
        X_train_np, X_test_np,
        results["psn_real"],
        cna_tr, mrna_tr, meth_tr,
        times_train, events_train,
        times_test,  events_test,
        modality_dims=modality_dims,
    )