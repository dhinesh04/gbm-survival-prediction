"""Nine ablation configs on the same train/test split, all with the AFT+Cox
dual head. Every config with a graph rebuilds its omics-only SNF PSN per fold
(leakage-safe); configs without a graph use a zero adjacency (self-loops
only -> equivalent to an MLP) throughout CV and final evaluation, not just
at test time.

  1-3. CNA / mRNA / Methylation only, no Clinical | SNF(omics) | encoder only
  4.   3 Omics only, no Clinical                  | SNF(omics) | enc + self-attn
  5.   Clinical only, no omics                    | no SNF     | encoder only
  6.   All 4 modalities                           | no SNF     | enc + self-attn
  7.   All 4 modalities                           | SNF(omics) | no encoder/self-attn
  8.   All 4 modalities                           | SNF(omics) | enc + plain concat
  9.   All 4 modalities                           | SNF(omics) | enc + self-attn  <- full model

Configs 1-3 and 5 have only one token to encode, so self-attention has nothing
to weigh against and is skipped. 8 vs 9 isolates self-attention's contribution,
7 vs 9 isolates the encoder/self-attn front end, 6 vs 9 isolates the graph.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import snf

from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

from src.models.gcn_model import GCN
from config import (
    HIDDEN_DIM, DROPOUT, LR, WEIGHT_DECAY,
    EPOCHS, PATIENCE, MIN_EPOCHS,
    N_FOLDS, K_TEST, RANDOM_STATE,
    ALPHA_AFT, ALPHA_COX,
    K_SNF, MU_SNF, N_ITER_SNF,
    ENC_DIM,
)
from src.utils import (
    concordance_index,
    cox_partial_likelihood_loss,
    aft_loss,
    normalise_adjacency,
    attach_test_nodes,
)


def _build_psn(matrices):
    """Omics-only SNF across `matrices` -- no survival-aware blending."""
    affinities = [snf.make_affinity(m, K=K_SNF, mu=MU_SNF) for m in matrices]
    psn_omics  = (affinities[0] if len(affinities) == 1
                  else snf.snf(affinities, K=K_SNF, t=N_ITER_SNF))
    return psn_omics.astype(np.float32)


def _print_summary(results):
    print("\n" + "=" * 97)
    print("  ABLATION STUDY — SUMMARY TABLE")
    print("=" * 97)
    print(f"  {'Configuration':<52} {'MAE(m)':>7}  {'C-idx(Cox)':>11}  {'IBS':>8}")
    print("  " + "-" * 95)
    for r in results:
        marker = " ← FULL MODEL" if "Full model" in r["label"] else ""
        print(f"  {r['label']:<52} {r['mae']:>7.2f}  "
              f"{r['cindex']:>11.4f}  "
              f"{r['ibs']:>8.4f}  "
              f"{marker}")
    print("=" * 97)


def _plot_ablation_barh(labels, values, color, title, xlabel, xlim, save_path,
                        text_pad=0.01, random_line=False):
    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, values, color=color)
    for bar in bars:
        plt.text(bar.get_width() + text_pad, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.3f}", va='center', fontsize=9)
    if random_line:
        plt.axvline(0.5, linestyle="--", color="red", alpha=0.5, label="Random")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xlim(*xlim)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_ablation_metrics(results, output_dir):
    """Bar charts comparing C-index and IBS across all configurations."""
    labels = [r["label"].split(" (")[0] for r in results]
    c_indices = [r["cindex"] for r in results]
    ibs_scores = [r["ibs"] for r in results]

    _plot_ablation_barh(
        labels, c_indices, "#4c72b0",
        "Ablation Study - Harrell's C-index (Higher = Better)", "C-index",
        (0.3, max(c_indices) + 0.1),
        os.path.join(output_dir, "ablation_cindex.png"),
        random_line=True)

    _plot_ablation_barh(
        labels, ibs_scores, "#55a868",
        "Ablation Study - Integrated Brier Score (Lower = Better)", "IBS",
        (0, max(ibs_scores) + 0.05),
        os.path.join(output_dir, "ablation_ibs.png"),
        text_pad=0.005)


def _build_model(n_in, modality_dims, fusion_type, n_self_attn_modalities):
    return GCN(
        n_in=n_in, n_hid=HIDDEN_DIM, dropout=DROPOUT,
        modality_dims=modality_dims, enc_dim=ENC_DIM,
        fusion_type=fusion_type, n_self_attn_modalities=n_self_attn_modalities,
    )


def _run_gcn(X_train_np, X_test_np,
             matrices_train,
             times_train, events_train, times_test, events_test,
             y_tr_sksurv, y_te_sksurv, time_grid,
             label, modality_dims=None, fusion_type="none",
             n_self_attn_modalities=None,
             X_attach_train=None, X_attach_test=None):
    """matrices_train: raw omics matrices to rebuild the omics-only PSN from
    every fold and at final test (leakage-safe); None skips SNF entirely,
    using a zero adjacency for both CV and final evaluation (a genuine
    "no graph" condition, not just at test time)."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    n_tr = X_train_np.shape[0]
    n_te = X_test_np.shape[0]

    _X_att_tr = X_attach_train if X_attach_train is not None else X_train_np
    _X_att_te = X_attach_test  if X_attach_test  is not None else X_test_np

    psn_train = (np.zeros((n_tr, n_tr), dtype=np.float32) if matrices_train is None
                 else _build_psn(matrices_train))

    full_adj_final = attach_test_nodes(psn_train, _X_att_tr, _X_att_te, k=K_TEST)
    adj_final      = torch.tensor(normalise_adjacency(full_adj_final), dtype=torch.float)

    X_all_f  = torch.tensor(np.vstack([X_train_np, X_test_np]), dtype=torch.float)
    t_all_f  = torch.tensor(np.concatenate([times_train, times_test]), dtype=torch.float)
    e_all_f  = torch.tensor(np.concatenate([events_train, events_test]), dtype=torch.float)
    idx_test = torch.arange(n_tr, n_tr + n_te)

    # 5-fold CV, per-fold PSN rebuild, stratified on censoring
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_maes, fold_epochs = [], []

    for tr_idx, val_idx in skf.split(X_train_np, events_train.astype(int)):
        fold_X_tr      = X_train_np[tr_idx]
        fold_X_val     = X_train_np[val_idx]
        fold_X_att_tr  = _X_att_tr[tr_idx]
        fold_X_att_val = _X_att_tr[val_idx]
        n_fold_tr      = len(tr_idx)
        n_fold_val     = len(val_idx)

        if matrices_train is None:
            fold_psn = np.zeros((n_fold_tr, n_fold_tr), dtype=np.float32)
        else:
            fold_matrices = [m[tr_idx] for m in matrices_train]
            fold_psn = _build_psn(fold_matrices)

        fold_adj = torch.tensor(
            normalise_adjacency(
                attach_test_nodes(fold_psn, fold_X_att_tr, fold_X_att_val, k=K_TEST)
            ), dtype=torch.float)

        X_fold = torch.tensor(np.vstack([fold_X_tr, fold_X_val]), dtype=torch.float)
        t_fold = torch.tensor(
            np.concatenate([times_train[tr_idx], times_train[val_idx]]), dtype=torch.float)
        e_fold = torch.tensor(
            np.concatenate([events_train[tr_idx], events_train[val_idx]]), dtype=torch.float)

        idx_tr_f  = torch.arange(n_fold_tr)
        idx_val_f = torch.arange(n_fold_tr, n_fold_tr + n_fold_val)

        model = _build_model(X_fold.shape[1], modality_dims, fusion_type, n_self_attn_modalities)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float('inf')
        best_val_mae  = float('inf')
        best_ep       = MIN_EPOCHS
        bad           = 0

        for epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            pred_log_t, cox_risk, _ = model(X_fold, fold_adj)
            l_aft = aft_loss(pred_log_t[idx_tr_f], t_fold[idx_tr_f], e_fold[idx_tr_f], model.log_sigma)
            l_cox = cox_partial_likelihood_loss(cox_risk[idx_tr_f], t_fold[idx_tr_f], e_fold[idx_tr_f])
            (ALPHA_AFT * l_aft + ALPHA_COX * l_cox).backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                pred_log_t_all, cox_risk_all, _ = model(X_fold, fold_adj)

            val_pred_log_t = pred_log_t_all[idx_val_f]
            val_t          = t_fold[idx_val_f]
            val_e          = e_fold[idx_val_f]

            val_l_aft = aft_loss(val_pred_log_t, val_t, val_e, model.log_sigma)
            val_l_cox = cox_partial_likelihood_loss(cox_risk_all[idx_val_f], val_t, val_e)
            val_loss  = (ALPHA_AFT * val_l_aft + ALPHA_COX * val_l_cox).item()

            vpm     = torch.exp(val_pred_log_t).cpu().numpy()
            vt_np   = val_t.cpu().numpy()
            ve_np   = val_e.cpu().numpy()
            uncens  = ve_np == 1
            val_mae = (float(np.abs(vpm[uncens] - vt_np[uncens]).mean())
                       if uncens.sum() > 0 else float('inf'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mae  = val_mae
                if epoch + 1 >= MIN_EPOCHS:
                    best_ep = epoch + 1
                bad = 0
            else:
                if epoch + 1 >= MIN_EPOCHS:
                    bad += 1
                    if bad >= PATIENCE:
                        break

        fold_maes.append(best_val_mae)
        fold_epochs.append(best_ep)

    final_epochs = max(MIN_EPOCHS, int(np.median(fold_epochs)))

    # final retrain on all training patients
    idx_all_tr = torch.arange(n_tr)
    model = _build_model(X_all_f.shape[1], modality_dims, fusion_type, n_self_attn_modalities)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(final_epochs):
        model.train()
        opt.zero_grad()
        pred_log_t, cox_risk, _ = model(X_all_f, adj_final)
        l_aft = aft_loss(pred_log_t[idx_all_tr], t_all_f[idx_all_tr], e_all_f[idx_all_tr], model.log_sigma)
        l_cox = cox_partial_likelihood_loss(cox_risk[idx_all_tr], t_all_f[idx_all_tr], e_all_f[idx_all_tr])
        (ALPHA_AFT * l_aft + ALPHA_COX * l_cox).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_log_t_f, cox_risk_f, _ = model(X_all_f, adj_final)

    test_pred_log_t  = pred_log_t_f[idx_test].cpu().numpy()
    test_pred_months = np.exp(test_pred_log_t)
    test_risk        = cox_risk_f[idx_test].cpu().numpy()
    test_sigma       = float(model.sigma().item())

    uncens_te = events_test == 1
    if uncens_te.sum() > 0:
        errors   = np.abs(test_pred_months[uncens_te] - times_test[uncens_te])
        test_mae = float(errors.mean())
    else:
        test_mae = float('nan')

    cindex = concordance_index(test_risk, times_test, events_test)

    try:
        log_t = np.log(time_grid).reshape(1, -1)
        surv_matrix = norm.sf((log_t - test_pred_log_t.reshape(-1, 1)) / test_sigma)
        ibs = integrated_brier_score(y_tr_sksurv, y_te_sksurv, surv_matrix, time_grid)
    except Exception:
        ibs = float('nan')

    print(f"  [{label:50s}]  MAE={test_mae:.2f}m  C-idx={cindex:.4f}  IBS={ibs:.4f}")

    return {"label": label, "mae": test_mae, "cindex": cindex, "ibs": ibs}


def run_ablation(pipeline, output_dir="plots", gcn_results=None):
    cna_tr  = pipeline["cna_tr_r"];    cna_te  = pipeline["cna_te"]
    mrna_tr = pipeline["mrna_tr_r"];   mrna_te = pipeline["mrna_te"]
    meth_tr = pipeline["meth_tr_r"];   meth_te = pipeline["meth_te"]
    clin_tr = pipeline["clin_tr_arr"]; clin_te = pipeline["clin_te"]

    t_tr = pipeline["os_months_train"].values.astype(float)
    e_tr = pipeline["os_status_train"].values.astype(float)
    t_te = pipeline["os_months_test"].values.astype(float)
    e_te = pipeline["os_status_test"].values.astype(float)

    cna_d, mrna_d, meth_d, clin_d = (
        cna_tr.shape[1], mrna_tr.shape[1], meth_tr.shape[1], clin_tr.shape[1])
    mod_dims_4 = [cna_d, mrna_d, meth_d, clin_d]

    y_tr_sksurv = Surv.from_arrays(event=e_tr.astype(bool), time=t_tr)
    y_te_sksurv = Surv.from_arrays(event=e_te.astype(bool), time=t_te)

    t_min = max(t_tr[e_tr == 1].min(), t_te[e_te == 1].min()) + 0.1
    t_max = min(t_tr[e_tr == 1].max(), t_te.max()) - 0.1
    time_grid = np.linspace(t_min, t_max, 100)

    print("\n" + "=" * 82)
    print("  ABLATION STUDY  — 9 Configurations [AFT + Cox Dual Head]")
    print("=" * 82)

    results = []
    kw = dict(times_train=t_tr, events_train=e_tr,
              times_test=t_te,  events_test=e_te,
              y_tr_sksurv=y_tr_sksurv, y_te_sksurv=y_te_sksurv, time_grid=time_grid)

    X_omics_tr, X_omics_te = np.hstack([cna_tr, mrna_tr, meth_tr]), np.hstack([cna_te, mrna_te, meth_te])
    X_all_tr, X_all_te = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr]), np.hstack([cna_te, mrna_te, meth_te, clin_te])

    # 1-3: single omics modality, no Clinical, SNF(omics), encoder only
    results.append(_run_gcn(
        cna_tr, cna_te, matrices_train=[cna_tr],
        label="CNA only, no Clinical (SNF-omics, encoder only)",
        modality_dims=[cna_d], **kw))

    results.append(_run_gcn(
        mrna_tr, mrna_te, matrices_train=[mrna_tr],
        label="mRNA only, no Clinical (SNF-omics, encoder only)",
        modality_dims=[mrna_d], **kw))

    results.append(_run_gcn(
        meth_tr, meth_te, matrices_train=[meth_tr],
        label="Methylation only, no Clinical (SNF-omics, encoder only)",
        modality_dims=[meth_d], **kw))

    # 4: 3 omics only, no Clinical, SNF(omics), enc+self-attn over all 3
    results.append(_run_gcn(
        X_omics_tr, X_omics_te, matrices_train=[cna_tr, mrna_tr, meth_tr],
        label="3 Omics only, no Clinical (SNF-omics, enc+selfattn)",
        modality_dims=[cna_d, mrna_d, meth_d], fusion_type="omics_self_attn",
        n_self_attn_modalities=3, **kw))

    # 5: clinical only, no omics -- single token, no SNF, same reasoning as 1-3
    results.append(_run_gcn(
        clin_tr, clin_te, matrices_train=None,
        label="Clinical only, no omics (no SNF, encoder only)",
        modality_dims=[clin_d], **kw))

    # 6: all 4, no SNF (zero adjacency), enc+self-attn
    results.append(_run_gcn(
        X_all_tr, X_all_te, matrices_train=None,
        label="All 4, no SNF, enc+selfattn",
        modality_dims=mod_dims_4, fusion_type="omics_self_attn",
        n_self_attn_modalities=3, **kw))

    # 7: all 4, SNF(omics), no encoder / no self-attn (raw concat)
    results.append(_run_gcn(
        X_all_tr, X_all_te, matrices_train=[cna_tr, mrna_tr, meth_tr],
        label="All 4, SNF-omics, no encoder/selfattn", **kw))

    # 8: all 4, SNF(omics), enc + plain concat -- isolates self-attention's
    # contribution against config 9, holding the graph and encoders fixed
    results.append(_run_gcn(
        X_all_tr, X_all_te, matrices_train=[cna_tr, mrna_tr, meth_tr],
        label="All 4, SNF-omics, enc+concat (no selfattn)",
        modality_dims=mod_dims_4, fusion_type="none", **kw))

    # 9: all 4, SNF(omics), enc+self-attn -- full model. If gcn_results is
    # supplied, reuses gcn_train.py's already-trained model instead of
    # retraining an identical architecture -- but that model's alpha_aft/
    # alpha_cox come from gcn_train.py's grid search, not the fixed
    # ALPHA_AFT/ALPHA_COX every other row trains with. Pass gcn_results=None
    # to retrain this row with the same fixed alphas for a strict comparison.
    if gcn_results is not None:
        try:
            log_t = np.log(time_grid).reshape(1, -1)
            surv_matrix = norm.sf((log_t - np.array(gcn_results["pred_log_t"]).reshape(-1, 1))
                                   / float(gcn_results["sigma"]))
            ibs = integrated_brier_score(y_tr_sksurv, y_te_sksurv, surv_matrix, time_grid)
        except Exception:
            ibs = float('nan')
        full = {"label": "All 4, SNF-omics, enc+selfattn (Full model)",
                "mae": gcn_results["mae"], "cindex": gcn_results["cindex"], "ibs": ibs}
        print(f"  [{full['label']:50s}]  MAE={full['mae']:.2f}m  "
              f"C-idx={full['cindex']:.4f}  IBS={full['ibs']:.4f}")
        results.append(full)
    else:
        results.append(_run_gcn(
            X_all_tr, X_all_te, matrices_train=[cna_tr, mrna_tr, meth_tr],
            label="All 4, SNF-omics, enc+selfattn (Full model)",
            modality_dims=mod_dims_4, fusion_type="omics_self_attn",
            n_self_attn_modalities=3, **kw))

    _print_summary(results)
    _plot_ablation_metrics(results, output_dir)

    return results