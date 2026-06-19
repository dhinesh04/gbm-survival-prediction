# """
# ablation_studies.py
# -------------------
# Ten configurations on the SAME train/test split.

#   A1.  CNA         + Clinical  | mRMR | SNF-SA | GCN
#   A2.  mRNA        + Clinical  | mRMR | SNF-SA | GCN
#   A3.  Methylation + Clinical  | mRMR | SNF-SA | GCN
#   A4.  CNA + mRNA + Meth       | mRMR | SNF-SA | GCN  (no clinical)
#   A5.  Clinical only           | —    | SNF-SA | GCN
#   A6.  All 4 modalities        | mRMR | SNF (omics-only) | GCN
#   A7.  All 4 modalities        | mRMR | SNF-SA | GCN               ← no encoders
#   A8.  All 4 modalities        | mRMR | SNF-SA | GCN + mod encoders ← FULL MODEL
#   A9.  Graph topology only     | —    | SNF-SA | GCN  (X = I, adj = P_SA)
#   A10. Node features only      | mRMR | —      | GCN  (adj = I → MLP)

# A7 vs A8 isolates the contribution of per-modality encoders.
# A9 vs A8 isolates the contribution of node features.
# A10 vs A8 isolates the contribution of graph neighbourhood aggregation.
# All CV uses per-fold PSN rebuild (no label leakage).
# """

# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import snf

# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score

# from src.models.gcn_model import GCN
# from src.graph.survival_aware_psn import build_survival_aware_psn
# from config import (
#     HIDDEN_DIM, DROPOUT, LR, WEIGHT_DECAY,
#     EPOCHS, PATIENCE, MIN_EPOCHS,
#     N_FOLDS, K_TEST, RANDOM_STATE,
#     ALPHA_BIN, ALPHA_COX,
#     K_SNF, MU_SNF, N_ITER_SNF,
#     ALPHA_SURVIVAL,
#     ENC_DIM,
# )
# from src.utils import (
#     concordance_index,
#     cox_partial_likelihood_loss,
#     normalise_adjacency,
#     attach_test_nodes,
#     find_best_threshold,
#     compute_class_weights,
#     plot_roc_curves,
# )


# # ─────────────────────────────────────────────────────────────────────────────
# # PSN BUILDER
# # ─────────────────────────────────────────────────────────────────────────────
# def _build_psn(matrices, os_months, os_status, survival_aware):
#     affinities = [snf.make_affinity(m, K=K_SNF, mu=MU_SNF) for m in matrices]
#     psn_omics  = (affinities[0] if len(affinities) == 1
#                   else snf.snf(affinities, K=K_SNF, t=N_ITER_SNF))
#     if survival_aware:
#         psn_out, _ = build_survival_aware_psn(
#             psn_omics, os_months, os_status, alpha=ALPHA_SURVIVAL, sigma=None)
#         return psn_out
#     return psn_omics.astype(np.float32)


# # ─────────────────────────────────────────────────────────────────────────────
# # CORE TRAINING (per-fold PSN + optional modality encoders)
# # ─────────────────────────────────────────────────────────────────────────────
# def _run_gcn(X_train_np, y_train_np,
#              X_test_np,  y_test_np,
#              psn_train,
#              matrices_train,
#              survival_aware,
#              times_train, events_train,
#              times_test,  events_test,
#              class_weights,
#              label,
#              modality_dims=None,
#              X_attach_train=None,
#              X_attach_test=None):
#     """
#     modality_dims : list[int] or None
#         None  → plain concatenation (A1–A7, A9–A10)
#         list  → per-modality encoders before GCN (A8)

#     X_attach_train / X_attach_test : np.ndarray or None
#         If provided, used instead of X_train_np/X_test_np for the cosine
#         similarity computation inside attach_test_nodes. This decouples
#         the features used for graph construction (kNN) from those used as
#         GCN node inputs. Required for A9 where X=I would produce all-zero
#         cosine similarities between disjoint one-hot vectors.
#     """
#     torch.manual_seed(RANDOM_STATE)
#     np.random.seed(RANDOM_STATE)

#     n_tr  = X_train_np.shape[0]
#     n_te  = X_test_np.shape[0]

#     # Features used for kNN attachment (may differ from GCN node features)
#     _X_att_tr = X_attach_train if X_attach_train is not None else X_train_np
#     _X_att_te = X_attach_test  if X_attach_test  is not None else X_test_np

#     # Final adjacency for retraining
#     full_adj_final = attach_test_nodes(psn_train, _X_att_tr, _X_att_te, k=K_TEST)
#     adj_final      = torch.tensor(normalise_adjacency(full_adj_final), dtype=torch.float)

#     X_all_f = torch.tensor(np.vstack([X_train_np, X_test_np]),      dtype=torch.float)
#     y_all_f = torch.tensor(np.concatenate([y_train_np, y_test_np]), dtype=torch.long)
#     t_all_f = torch.tensor(np.concatenate([times_train, times_test]),    dtype=torch.float)
#     e_all_f = torch.tensor(np.concatenate([events_train, events_test]),  dtype=torch.float)
#     idx_test = torch.arange(n_tr, n_tr + n_te)

#     # 5-fold CV with per-fold PSN rebuild
#     skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
#     fold_aucs, fold_threshs, fold_epochs = [], [], []

#     for tr_idx, val_idx in skf.split(X_train_np, y_train_np):
#         fold_matrices = [m[tr_idx] for m in matrices_train]
#         fold_psn = _build_psn(
#             fold_matrices,
#             times_train[tr_idx], events_train[tr_idx],
#             survival_aware=survival_aware,
#         )
#         fold_X_tr  = X_train_np[tr_idx]
#         fold_X_val = X_train_np[val_idx]
#         fold_X_att_tr  = _X_att_tr[tr_idx]
#         fold_X_att_val = _X_att_tr[val_idx]
#         n_fold_tr  = len(tr_idx)
#         n_fold_val = len(val_idx)

#         fold_adj = torch.tensor(
#             normalise_adjacency(
#                 attach_test_nodes(fold_psn, fold_X_att_tr, fold_X_att_val, k=K_TEST)
#             ), dtype=torch.float)

#         X_fold = torch.tensor(np.vstack([fold_X_tr, fold_X_val]), dtype=torch.float)
#         y_fold = torch.tensor(
#             np.concatenate([y_train_np[tr_idx], y_train_np[val_idx]]),
#             dtype=torch.long)
#         t_fold = torch.tensor(
#             np.concatenate([times_train[tr_idx], times_train[val_idx]]),
#             dtype=torch.float)
#         e_fold = torch.tensor(
#             np.concatenate([events_train[tr_idx], events_train[val_idx]]),
#             dtype=torch.float)

#         idx_tr_f  = torch.arange(n_fold_tr)
#         idx_val_f = torch.arange(n_fold_tr, n_fold_tr + n_fold_val)

#         model = GCN(
#             n_in=X_fold.shape[1], n_hid=HIDDEN_DIM, n_out=2, dropout=DROPOUT,
#             modality_dims=modality_dims, enc_dim=ENC_DIM,
#         )
#         opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#         best_auc, best_thresh, best_ep, bad = 0.0, 0.5, MIN_EPOCHS, 0

#         for epoch in range(EPOCHS):
#             model.train(); opt.zero_grad()
#             bl, cr, _ = model(X_fold, fold_adj)
#             l_bin = F.cross_entropy(bl[idx_tr_f], y_fold[idx_tr_f],
#                                     weight=class_weights)
#             l_cox = cox_partial_likelihood_loss(
#                 cr[idx_tr_f], t_fold[idx_tr_f], e_fold[idx_tr_f])
#             (ALPHA_BIN * l_bin + ALPHA_COX * l_cox).backward()
#             opt.step()

#             model.eval()
#             with torch.no_grad():
#                 bl_all, _, _ = model(X_fold, fold_adj)
#             vp = torch.softmax(bl_all[idx_val_f], dim=1)[:, 1].cpu().numpy()
#             vt = y_fold[idx_val_f].cpu().numpy()
#             if len(np.unique(vt)) < 2:
#                 continue

#             vauc = roc_auc_score(vt, vp)
#             if vauc > best_auc:
#                 best_auc    = vauc
#                 best_thresh = find_best_threshold(vp, vt)
#                 if epoch + 1 >= MIN_EPOCHS:
#                     best_ep = epoch + 1
#                 bad = 0
#             else:
#                 if epoch + 1 >= MIN_EPOCHS:
#                     bad += 1
#                     if bad >= PATIENCE:
#                         break

#         fold_aucs.append(best_auc)
#         fold_threshs.append(best_thresh)
#         fold_epochs.append(best_ep)

#     final_epochs = max(MIN_EPOCHS, int(np.median(fold_epochs)))
#     mean_val_auc = float(np.mean(fold_aucs))

#     # Final retrain
#     idx_all_tr = torch.arange(n_tr)
#     model = GCN(
#         n_in=X_all_f.shape[1], n_hid=HIDDEN_DIM, n_out=2, dropout=DROPOUT,
#         modality_dims=modality_dims, enc_dim=ENC_DIM,
#     )
#     opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
#     for epoch in range(final_epochs):
#         model.train(); opt.zero_grad()
#         bl, cr, _ = model(X_all_f, adj_final)
#         l_bin = F.cross_entropy(bl[idx_all_tr], y_all_f[idx_all_tr],
#                                 weight=class_weights)
#         l_cox = cox_partial_likelihood_loss(
#             cr[idx_all_tr], t_all_f[idx_all_tr], e_all_f[idx_all_tr])
#         (ALPHA_BIN * l_bin + ALPHA_COX * l_cox).backward()
#         opt.step()

#     model.eval()
#     with torch.no_grad():
#         bl_f, cr_f, _ = model(X_all_f, adj_final)

#     test_probs = torch.softmax(bl_f[idx_test], dim=1)[:, 1].cpu().numpy()
#     test_true  = y_all_f[idx_test].cpu().numpy()
#     test_risk  = cr_f[idx_test].cpu().numpy()
#     test_auc   = roc_auc_score(test_true, test_probs)
#     cindex     = concordance_index(test_risk, times_test, events_test)

#     print(f"  [{label:50s}]  AUC={test_auc:.4f}  C-index={cindex:.4f}  "
#           f"(CV AUC={mean_val_auc:.4f}  epochs={final_epochs})")

#     return {
#         "label":       label,
#         "auc":         test_auc,
#         "cindex":      cindex,
#         "probs":       test_probs,
#         "y_true":      test_true,
#         "cv_auc_mean": mean_val_auc,
#         "cv_auc_std":  float(np.std(fold_aucs)),
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # SUMMARY TABLE
# # ─────────────────────────────────────────────────────────────────────────────
# def _print_summary(results):
#     print("\n" + "=" * 82)
#     print("  ABLATION STUDY — SUMMARY TABLE")
#     print("=" * 82)
#     print(f"  {'Configuration':<52} {'AUC':>6}  {'C-index':>8}  {'CV AUC':>14}")
#     print("  " + "-" * 80)
#     for r in results:
#         marker = " ← FULL MODEL" if "Full model" in r["label"] else ""
#         print(f"  {r['label']:<52} {r['auc']:>6.4f}  "
#               f"{r['cindex']:>8.4f}  "
#               f"{r['cv_auc_mean']:>6.4f}±{r['cv_auc_std']:.4f}"
#               f"{marker}")
#     print("=" * 82)


# # ─────────────────────────────────────────────────────────────────────────────
# # PUBLIC ENTRY POINT
# # ─────────────────────────────────────────────────────────────────────────────
# def run_ablation(pipeline, output_dir="plots", gcn_results=None):
#     """
#     A1–A8: modality and architecture ablations (unchanged).
#     A9: Graph topology only  — X = identity, real P_SA adjacency.
#     A10: Node features only  — full X, zero adjacency (normalises to I → MLP).
#     """
#     cna_tr  = pipeline["cna_tr_r"];   cna_te  = pipeline["cna_te"]
#     mrna_tr = pipeline["mrna_tr_r"];  mrna_te = pipeline["mrna_te"]
#     meth_tr = pipeline["meth_tr_r"];  meth_te = pipeline["meth_te"]
#     clin_tr = pipeline["clin_tr_arr"]; clin_te = pipeline["clin_te"]

#     y_tr = pipeline["y_train"].values
#     y_te = pipeline["y_test"].values
#     t_tr = pipeline["os_months_train"].values.astype(float)
#     e_tr = pipeline["os_status_train"].values.astype(float)
#     t_te = pipeline["os_months_test"].values.astype(float)
#     e_te = pipeline["os_status_test"].values.astype(float)

#     class_weights, n_lts, _ = compute_class_weights(y_tr)

#     print("\n" + "=" * 82)
#     print("  ABLATION STUDY  — 11 Configurations")
#     print("=" * 82)
#     print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")
#     print(f"  LTS train={n_lts} | LTS test={int(y_te.sum())}")
#     print(f"  Class weights: non-LTS={class_weights[0]:.3f}, LTS={class_weights[1]:.3f}")
#     print(f"  Survival-aware alpha = {ALPHA_SURVIVAL}")
#     print(f"  CV: per-fold PSN rebuild (no label leakage)")
#     print(f"  A7 vs A8   isolates modality encoders (ENC_DIM={ENC_DIM})")
#     print(f"  A9a vs A8  isolates node features given P_SA graph")
#     print(f"  A9b vs A9a isolates survival-aware blending in adjacency")
#     print(f"  A10 vs A8  isolates graph aggregation (node features only / MLP)")
#     print("=" * 82 + "\n")

#     results = []
#     kw = dict(times_train=t_tr, events_train=e_tr,
#               times_test=t_te,  events_test=e_te,
#               class_weights=class_weights)

#     # A1 — CNA only ──────────────────────────────────────────────────────────
#     print("── A1: CNA + Clinical ──────────────────────────────────────────")
#     results.append(_run_gcn(
#         cna_tr, y_tr, cna_te, y_te,
#         _build_psn([cna_tr], t_tr, e_tr, True),
#         matrices_train=[cna_tr], survival_aware=True,
#         label="CNA only", **kw,
#     ))

#     # A2 — mRNA only ─────────────────────────────────────────────────────────
#     print("── A2: mRNA + Clinical ─────────────────────────────────────────")
#     results.append(_run_gcn(
#         mrna_tr, y_tr, mrna_te, y_te,
#         _build_psn([mrna_tr], t_tr, e_tr, True),
#         matrices_train=[mrna_tr], survival_aware=True,
#         label="mRNA only", **kw,
#     ))

#     # A3 — Methylation only ──────────────────────────────────────────────────
#     print("── A3: Methylation + Clinical ──────────────────────────────────")
#     results.append(_run_gcn(
#         meth_tr, y_tr, meth_te, y_te,
#         _build_psn([meth_tr], t_tr, e_tr, True),
#         matrices_train=[meth_tr], survival_aware=True,
#         label="Methylation only", **kw,
#     ))

#     # A4 — 3 Omics, no Clinical ──────────────────────────────────────────────
#     print("── A4: 3 Omics only (no Clinical) ──────────────────────────────")
#     X_tr4 = np.hstack([cna_tr, mrna_tr, meth_tr])
#     X_te4 = np.hstack([cna_te, mrna_te, meth_te])
#     results.append(_run_gcn(
#         X_tr4, y_tr, X_te4, y_te,
#         _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, True),
#         matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
#         label="3 Omics only (no Clinical)", **kw,
#     ))

#     # A5 — Clinical only ─────────────────────────────────────────────────────
#     print("── A5: Clinical only ───────────────────────────────────────────")
#     results.append(_run_gcn(
#         clin_tr, y_tr, clin_te, y_te,
#         _build_psn([clin_tr], t_tr, e_tr, True),
#         matrices_train=[clin_tr], survival_aware=True,
#         label="Clinical only", **kw,
#     ))

#     # A6 — All 4 + SNF omics-only (no survival-aware) ────────────────────────
#     print("── A6: All 4 + SNF (no survival-aware) ─────────────────────────")
#     X_tr6 = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
#     X_te6 = np.hstack([cna_te, mrna_te, meth_te, clin_te])
#     results.append(_run_gcn(
#         X_tr6, y_tr, X_te6, y_te,
#         _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, False),
#         matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=False,
#         label="All 4 + SNF (no surv-aware)", **kw,
#     ))

#     # A7 — All 4 + SNF-SA, NO modality encoders ──────────────────────────────
#     print("── A7: All 4 + SNF-SA (no modality encoders) ───────────────────")
#     X_tr7 = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
#     X_te7 = np.hstack([cna_te, mrna_te, meth_te, clin_te])
#     results.append(_run_gcn(
#         X_tr7, y_tr, X_te7, y_te,
#         _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, True),
#         matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
#         label="All 4 + SNF-SA (no encoders)",
#         modality_dims=None,
#         **kw,
#     ))

#     # A8 — All 4 + SNF-SA + Modality Encoders = FULL MODEL ───────────────────
#     print("── A8: All 4 + SNF-SA + Modality Encoders (FULL MODEL) ─────────")
#     modality_dims = [cna_tr.shape[1], mrna_tr.shape[1],
#                      meth_tr.shape[1], clin_tr.shape[1]]
#     if gcn_results is not None:
#         a8 = {
#             "label":       "All 4 + SNF-SA + Encoders (Full model)",
#             "auc":         gcn_results["auc"],
#             "cindex":      gcn_results["cindex"],
#             "probs":       gcn_results["probs"],
#             "y_true":      gcn_results["y_true"],
#             "cv_auc_mean": gcn_results["cv_val_auc_mean"],
#             "cv_auc_std":  gcn_results["cv_val_auc_std"],
#         }
#         print(f"  [{'All 4 + SNF-SA + Encoders (Full model)':50s}]  "
#               f"AUC={a8['auc']:.4f}  C-index={a8['cindex']:.4f}  "
#               f"(CV AUC={a8['cv_auc_mean']:.4f}  "
#               f"epochs={gcn_results['final_epochs_used']})")
#     else:
#         X_tr8 = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
#         X_te8 = np.hstack([cna_te, mrna_te, meth_te, clin_te])
#         a8 = _run_gcn(
#             X_tr8, y_tr, X_te8, y_te,
#             pipeline["psn_real"],
#             matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
#             label="All 4 + SNF-SA + Encoders (Full model)",
#             modality_dims=modality_dims,
#             **kw,
#         )
#     results.append(a8)

#     # Shared setup for A9a and A9b ────────────────────────────────────────────
#     # Identity node features: GCN receives no omics signal, must rely entirely
#     # on neighbourhood aggregation. kNN attachment uses real omics features so
#     # test nodes get meaningful edges (one-hots from disjoint index sets have
#     # zero cosine similarity → broken edges → AUC/C-index collapse without fix).
#     n_tr = len(y_tr)
#     n_te = len(y_te)
#     X_tr_omics = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
#     X_te_omics = np.hstack([cna_te, mrna_te, meth_te, clin_te])
#     X_tr9 = np.eye(n_tr, dtype=np.float32)           # (n_tr, n_tr) — GCN input
#     X_te9 = np.zeros((n_te, n_tr), dtype=np.float32) # (n_te, n_tr) — GCN input

#     # A9a — Graph topology only, P_SA adjacency ──────────────────────────────
#     # Uses the survival-aware PSN (same adjacency as the full model A8).
#     print("── A9a: Graph topology only (X=I, P_SA adj) ────────────────────")
#     results.append(_run_gcn(
#         X_tr9, y_tr, X_te9, y_te,
#         pipeline["psn_real"],
#         matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
#         label="Graph topology only (X=I, P_SA adj)",
#         modality_dims=None,
#         X_attach_train=X_tr_omics,
#         X_attach_test=X_te_omics,
#         **kw,
#     ))

#     # A9b — Graph topology only, P_omics adjacency ───────────────────────────
#     # Uses the pure omics SNF PSN (no survival blending) as the adjacency.
#     # Isolates whether the survival-aware blending in P_SA contributes signal
#     # beyond what the omics graph structure alone provides.
#     print("── A9b: Graph topology only (X=I, P_omics adj) ─────────────────")
#     psn_omics = _build_psn([cna_tr, mrna_tr, meth_tr], t_tr, e_tr, survival_aware=False)
#     results.append(_run_gcn(
#         X_tr9, y_tr, X_te9, y_te,
#         psn_omics,
#         matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=False,
#         label="Graph topology only (X=I, P_omics adj)",
#         modality_dims=None,
#         X_attach_train=X_tr_omics,
#         X_attach_test=X_te_omics,
#         **kw,
#     ))

#     # A10 — Node features only (identity adjacency → MLP) ────────────────────
#     # Full multi-omics X (same as A8); adjacency = zero matrix.
#     # normalise_adjacency adds self-loops: D^{-0.5}(0+I)D^{-0.5} = I.
#     # GCN layers become H = I·X·W = X·W → pure 2-layer MLP, no aggregation.
#     # No PSN rebuild needed in CV folds; zero matrix used throughout.
#     print("── A10: Node features only (zero adj → identity → MLP) ─────────")
#     X_tr10 = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
#     X_te10 = np.hstack([cna_te, mrna_te, meth_te, clin_te])
#     zero_psn = np.zeros((n_tr, n_tr), dtype=np.float32)
#     results.append(_run_gcn(
#         X_tr10, y_tr, X_te10, y_te,
#         zero_psn,
#         matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=False,
#         label="Node features only (zero adj → MLP)",
#         modality_dims=[cna_tr.shape[1], mrna_tr.shape[1],
#                        meth_tr.shape[1], clin_tr.shape[1]],
#         **kw,
#     ))

#     _print_summary(results)
#     plot_roc_curves(
#         results,
#         output_path=os.path.join(output_dir, "ablation_roc_curves.png"),
#         title="GBM LTS Prediction — Ablation ROC Curves",
#     )
#     return results

"""
ablation_studies.py
-------------------
Ten configurations on the SAME train/test split.

  A1.  CNA         + Clinical  | mRMR | SNF-SA | GCN
  A2.  mRNA        + Clinical  | mRMR | SNF-SA | GCN
  A3.  Methylation + Clinical  | mRMR | SNF-SA | GCN
  A4.  CNA + mRNA + Meth       | mRMR | SNF-SA | GCN  (no clinical)
  A5.  Clinical only           | —    | SNF-SA | GCN
  A6.  All 4 modalities        | mRMR | SNF (omics-only) | GCN
  A7.  All 4 modalities        | mRMR | SNF-SA | GCN               ← no encoders
  A8.  All 4 modalities        | mRMR | SNF-SA | GCN + mod encoders ← FULL MODEL
  A9.  Graph topology only     | —    | SNF-SA | GCN  (X = I, adj = P_SA)
  A10. Node features only      | mRMR | —      | GCN  (adj = I → MLP)

A7 vs A8 isolates the contribution of per-modality encoders.
A9 vs A8 isolates the contribution of node features.
A10 vs A8 isolates the contribution of graph neighbourhood aggregation.
All CV uses per-fold PSN rebuild (no label leakage).

AFT changes vs binary-head version:
  - _run_gcn uses AFT regression loss instead of binary cross-entropy.
  - Model selection: min combined val_loss (AFT + Cox) instead of max AUC.
  - Reported per-config metrics: MAE, AUC(thr), C-index (Cox + AFT).
  - threshold read from pipeline dict — correct for all sensitivity runs.
  - plot_roc_curves called with pred_months as score (AUC from threshold).
"""

import os
import numpy as np
import torch
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
    ALPHA_AFT, ALPHA_COX,
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
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _print_summary(results, threshold=12):
    print("\n" + "=" * 97)
    print("  ABLATION STUDY — SUMMARY TABLE")
    print("=" * 97)
    print(f"  {'Configuration':<52} {'MAE(m)':>7}  "
          f"{'AUC(' + str(int(threshold)) + 'm)':>9}  "
          f"{'C-idx(Cox)':>11}  {'CV MAE':>12}")
    print("  " + "-" * 95)
    for r in results:
        marker = " ← FULL MODEL" if "Full model" in r["label"] else ""
        print(f"  {r['label']:<52} {r['mae']:>7.2f}  "
              f"{r['auc_threshold']:>9.4f}  "
              f"{r['cindex']:>11.4f}  "
              f"{r['cv_mae_mean']:>6.2f}±{r['cv_mae_std']:.2f}"
              f"{marker}")
    print("=" * 97)


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _run_gcn(X_train_np, y_train_np,
             X_test_np,  y_test_np,
             psn_train,
             matrices_train,
             survival_aware,
             times_train, events_train,
             times_test,  events_test,
             label,
             modality_dims=None,
             X_attach_train=None,
             X_attach_test=None):
    """
    Train one ablation configuration with AFT + Cox joint loss.

    y_train_np / y_test_np (binary LTS labels derived from OS_MONTHS > threshold)
    are used only for StratifiedKFold splitting and AUC reporting.
    They play no role in loss computation.

    modality_dims : list[int] or None
        None  → plain concatenation (A1–A7, A9–A10)
        list  → per-modality encoders before GCN (A8)

    X_attach_train / X_attach_test : np.ndarray or None
        If provided, used for cosine similarity in attach_test_nodes.
        Required for A9 (X=I).
    """
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    n_tr = X_train_np.shape[0]
    n_te = X_test_np.shape[0]

    _X_att_tr = X_attach_train if X_attach_train is not None else X_train_np
    _X_att_te = X_attach_test  if X_attach_test  is not None else X_test_np

    full_adj_final = attach_test_nodes(psn_train, _X_att_tr, _X_att_te, k=K_TEST)
    adj_final      = torch.tensor(normalise_adjacency(full_adj_final), dtype=torch.float)

    X_all_f  = torch.tensor(np.vstack([X_train_np, X_test_np]), dtype=torch.float)
    t_all_f  = torch.tensor(np.concatenate([times_train, times_test]), dtype=torch.float)
    e_all_f  = torch.tensor(np.concatenate([events_train, events_test]), dtype=torch.float)
    idx_test = torch.arange(n_tr, n_tr + n_te)

    # ── 5-fold CV (per-fold PSN rebuild) ──────────────────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_maes, fold_epochs = [], []

    for tr_idx, val_idx in skf.split(X_train_np, y_train_np):
        fold_matrices = [m[tr_idx] for m in matrices_train]
        fold_psn = _build_psn(
            fold_matrices,
            times_train[tr_idx], events_train[tr_idx],
            survival_aware=survival_aware,
        )
        fold_X_tr      = X_train_np[tr_idx]
        fold_X_val     = X_train_np[val_idx]
        fold_X_att_tr  = _X_att_tr[tr_idx]
        fold_X_att_val = _X_att_tr[val_idx]
        n_fold_tr      = len(tr_idx)
        n_fold_val     = len(val_idx)

        fold_adj = torch.tensor(
            normalise_adjacency(
                attach_test_nodes(fold_psn, fold_X_att_tr, fold_X_att_val, k=K_TEST)
            ), dtype=torch.float)

        X_fold = torch.tensor(np.vstack([fold_X_tr, fold_X_val]), dtype=torch.float)
        t_fold = torch.tensor(
            np.concatenate([times_train[tr_idx], times_train[val_idx]]),
            dtype=torch.float)
        e_fold = torch.tensor(
            np.concatenate([events_train[tr_idx], events_train[val_idx]]),
            dtype=torch.float)

        idx_tr_f  = torch.arange(n_fold_tr)
        idx_val_f = torch.arange(n_fold_tr, n_fold_tr + n_fold_val)

        model = GCN(
            n_in=X_fold.shape[1], n_hid=HIDDEN_DIM, dropout=DROPOUT,
            modality_dims=modality_dims, enc_dim=ENC_DIM,
        )
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_val_loss = float('inf')
        best_val_mae  = float('inf')
        best_ep       = MIN_EPOCHS
        bad           = 0

        for epoch in range(EPOCHS):
            model.train()
            opt.zero_grad()
            pred_log_t, cox_risk, _ = model(X_fold, fold_adj)
            l_aft = aft_loss(pred_log_t[idx_tr_f], t_fold[idx_tr_f], e_fold[idx_tr_f])
            l_cox = cox_partial_likelihood_loss(
                cox_risk[idx_tr_f], t_fold[idx_tr_f], e_fold[idx_tr_f])
            (ALPHA_AFT * l_aft + ALPHA_COX * l_cox).backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                pred_log_t_all, cox_risk_all, _ = model(X_fold, fold_adj)

            val_pred_log_t = pred_log_t_all[idx_val_f]
            val_t          = t_fold[idx_val_f]
            val_e          = e_fold[idx_val_f]

            val_l_aft = aft_loss(val_pred_log_t, val_t, val_e)
            val_l_cox = cox_partial_likelihood_loss(
                cox_risk_all[idx_val_f], val_t, val_e)
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
    mean_val_mae = float(np.mean(fold_maes))

    # ── Final retrain ──────────────────────────────────────────────────────
    idx_all_tr = torch.arange(n_tr)
    model = GCN(
        n_in=X_all_f.shape[1], n_hid=HIDDEN_DIM, dropout=DROPOUT,
        modality_dims=modality_dims, enc_dim=ENC_DIM,
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(final_epochs):
        model.train()
        opt.zero_grad()
        pred_log_t, cox_risk, _ = model(X_all_f, adj_final)
        l_aft = aft_loss(
            pred_log_t[idx_all_tr], t_all_f[idx_all_tr], e_all_f[idx_all_tr])
        l_cox = cox_partial_likelihood_loss(
            cox_risk[idx_all_tr], t_all_f[idx_all_tr], e_all_f[idx_all_tr])
        (ALPHA_AFT * l_aft + ALPHA_COX * l_cox).backward()
        opt.step()

    # ── Test evaluation ────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_log_t_f, cox_risk_f, _ = model(X_all_f, adj_final)

    test_pred_log_t  = pred_log_t_f[idx_test].cpu().numpy()
    test_pred_months = np.exp(test_pred_log_t)
    test_risk        = cox_risk_f[idx_test].cpu().numpy()
    test_true        = y_test_np

    uncens_te = events_test == 1
    if uncens_te.sum() > 0:
        errors    = np.abs(test_pred_months[uncens_te] - times_test[uncens_te])
        test_mae  = float(errors.mean())
        test_rmse = float(np.sqrt(
            ((test_pred_months[uncens_te] - times_test[uncens_te]) ** 2).mean()))
    else:
        test_mae = test_rmse = float('nan')

    try:
        test_auc_thr = float(roc_auc_score(test_true, test_pred_months))
    except Exception:
        test_auc_thr = float('nan')

    cindex     = concordance_index(test_risk, times_test, events_test)
    cindex_aft = concordance_index(-test_pred_log_t, times_test, events_test)

    print(f"  [{label:50s}]  MAE={test_mae:.2f}m  "
          f"AUC(thr)={test_auc_thr:.4f}  C-idx={cindex:.4f}  "
          f"(CV MAE={mean_val_mae:.2f}  epochs={final_epochs})")

    return {
        "label":         label,
        "mae":           test_mae,
        "rmse":          test_rmse,
        "auc_threshold": test_auc_thr,
        "cindex":        cindex,
        "cindex_aft":    cindex_aft,
        "pred_months":   test_pred_months,
        "y_true":        test_true,
        "cv_mae_mean":   mean_val_mae,
        "cv_mae_std":    float(np.std(fold_maes)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(pipeline, output_dir="plots", gcn_results=None):
    """
    A1–A8: modality and architecture ablations.
    A9: Graph topology only  — X = identity, real P_SA adjacency.
    A10: Node features only  — full X, zero adjacency (→ identity → MLP).
    """
    cna_tr  = pipeline["cna_tr_r"];    cna_te  = pipeline["cna_te"]
    mrna_tr = pipeline["mrna_tr_r"];   mrna_te = pipeline["mrna_te"]
    meth_tr = pipeline["meth_tr_r"];   meth_te = pipeline["meth_te"]
    clin_tr = pipeline["clin_tr_arr"]; clin_te = pipeline["clin_te"]

    y_tr = pipeline["y_train"].values
    y_te = pipeline["y_test"].values
    t_tr = pipeline["os_months_train"].values.astype(float)
    e_tr = pipeline["os_status_train"].values.astype(float)
    t_te = pipeline["os_months_test"].values.astype(float)
    e_te = pipeline["os_status_test"].values.astype(float)

    # Read threshold from pipeline — set in main.py, correct for all runs
    threshold = pipeline.get("threshold", 12)
    n_lts     = int(y_tr.sum())

    print("\n" + "=" * 82)
    print(f"  ABLATION STUDY  — 11 Configurations  "
          f"[AFT + Cox | LTS threshold = {int(threshold)}m]")
    print("=" * 82)
    print(f"  Train: {len(y_tr)} | Test: {len(y_te)}")
    print(f"  LTS train={n_lts} | LTS test={int(y_te.sum())}")
    print(f"  Survival-aware alpha = {ALPHA_SURVIVAL}")
    print(f"  CV: per-fold PSN rebuild (no label leakage)")
    print(f"  Model selection: min val_loss (AFT + Cox)")
    print(f"  A7 vs A8   isolates modality encoders (ENC_DIM={ENC_DIM})")
    print(f"  A9a vs A8  isolates node features given P_SA graph")
    print(f"  A9b vs A9a isolates survival-aware blending in adjacency")
    print(f"  A10 vs A8  isolates graph aggregation (node features only / MLP)")
    print("=" * 82 + "\n")

    results = []
    kw = dict(times_train=t_tr, events_train=e_tr,
              times_test=t_te,  events_test=e_te)

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
        modality_dims=None,
        **kw,
    ))

    # A8 — All 4 + SNF-SA + Modality Encoders = FULL MODEL ───────────────────
    print("── A8: All 4 + SNF-SA + Modality Encoders (FULL MODEL) ─────────")
    modality_dims_full = [cna_tr.shape[1], mrna_tr.shape[1],
                          meth_tr.shape[1], clin_tr.shape[1]]
    if gcn_results is not None:
        a8 = {
            "label":         "All 4 + SNF-SA + Encoders (Full model)",
            "mae":           gcn_results["mae"],
            "rmse":          gcn_results.get("rmse", float('nan')),
            "auc_threshold": gcn_results["threshold_sensitivity"][12],
            "cindex":        gcn_results["cindex"],
            "cindex_aft":    gcn_results["cindex_aft"],
            "pred_months":   gcn_results["pred_months"],
            "y_true":        y_te,
            "cv_mae_mean":   gcn_results["cv_val_mae_mean"],
            "cv_mae_std":    gcn_results["cv_val_mae_std"],
        }
        print(f"  [{'All 4 + SNF-SA + Encoders (Full model)':50s}]  "
              f"MAE={a8['mae']:.2f}m  AUC(thr)={a8['auc_threshold']:.4f}  "
              f"C-idx={a8['cindex']:.4f}  "
              f"(CV MAE={a8['cv_mae_mean']:.2f}  "
              f"epochs={gcn_results['final_epochs_used']})")
    else:
        X_tr8 = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
        X_te8 = np.hstack([cna_te, mrna_te, meth_te, clin_te])
        a8 = _run_gcn(
            X_tr8, y_tr, X_te8, y_te,
            pipeline["psn_real"],
            matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
            label="All 4 + SNF-SA + Encoders (Full model)",
            modality_dims=modality_dims_full,
            **kw,
        )
    results.append(a8)

    # A9 setup: identity node features ────────────────────────────────────────
    n_tr_val   = len(y_tr)
    n_te_val   = len(y_te)
    X_tr_omics = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
    X_te_omics = np.hstack([cna_te, mrna_te, meth_te, clin_te])
    X_tr9      = np.eye(n_tr_val, dtype=np.float32)
    X_te9      = np.zeros((n_te_val, n_tr_val), dtype=np.float32)

    # A9a — Graph topology only, P_SA adjacency ──────────────────────────────
    print("── A9a: Graph topology only (X=I, P_SA adj) ────────────────────")
    results.append(_run_gcn(
        X_tr9, y_tr, X_te9, y_te,
        pipeline["psn_real"],
        matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=True,
        label="Graph topology only (X=I, P_SA adj)",
        modality_dims=None,
        X_attach_train=X_tr_omics,
        X_attach_test=X_te_omics,
        **kw,
    ))

    # A9b — Graph topology only, P_omics adjacency ───────────────────────────
    print("── A9b: Graph topology only (X=I, P_omics adj) ─────────────────")
    psn_omics = _build_psn(
        [cna_tr, mrna_tr, meth_tr], t_tr, e_tr, survival_aware=False)
    results.append(_run_gcn(
        X_tr9, y_tr, X_te9, y_te,
        psn_omics,
        matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=False,
        label="Graph topology only (X=I, P_omics adj)",
        modality_dims=None,
        X_attach_train=X_tr_omics,
        X_attach_test=X_te_omics,
        **kw,
    ))

    # A10 — Node features only (identity adjacency → MLP) ────────────────────
    print("── A10: Node features only (zero adj → identity → MLP) ─────────")
    X_tr10   = np.hstack([cna_tr, mrna_tr, meth_tr, clin_tr])
    X_te10   = np.hstack([cna_te, mrna_te, meth_te, clin_te])
    zero_psn = np.zeros((n_tr_val, n_tr_val), dtype=np.float32)
    results.append(_run_gcn(
        X_tr10, y_tr, X_te10, y_te,
        zero_psn,
        matrices_train=[cna_tr, mrna_tr, meth_tr], survival_aware=False,
        label="Node features only (zero adj → MLP)",
        modality_dims=[cna_tr.shape[1], mrna_tr.shape[1],
                       meth_tr.shape[1], clin_tr.shape[1]],
        **kw,
    ))

    _print_summary(results, threshold=threshold)

    roc_data = [
        {"label": r["label"],
         "probs": r["pred_months"],
         "y_true": r["y_true"],
         "auc":    r["auc_threshold"]}
        for r in results
    ]
    plot_roc_curves(
        roc_data,
        output_path=os.path.join(output_dir, "ablation_roc_curves.png"),
        title=f"GBM Survival Prediction — Ablation ROC Curves "
              f"(AFT threshold AUC, {int(threshold)}m)",
    )
    return results