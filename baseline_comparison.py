"""
baseline_comparison.py
----------------------
Compares the GCN dual-head model against classical and deep-learning
baselines on the same held-out test set.

BINARY HEAD BASELINES (LTS classification, evaluated by AUC):
  1.  Logistic Regression        (L2, class-balanced)
  2.  Naive Bayes                (GaussianNB)
  3.  K-Nearest Neighbours       (k=11, distance-weighted)
  4.  Decision Tree              (class-balanced, max_depth=5)
  5.  Random Forest              (500 trees, class-balanced)
  6.  Gradient Boosting          (sklearn HistGradientBoosting)
  7.  Support Vector Machine     (RBF kernel, probability=True)
  8.  MLP / DNN                  (3-layer, sklearn)
  [9. GCN (our model)            — injected from gcn_results]

SURVIVAL HEAD BASELINES (time-to-event, evaluated by C-index):
  1.  CoxPH — Classical          (lifelines, L2 penalised)
  2.  Lasso-Cox                  (lifelines, L1 penalised)
  3.  Random Survival Forest     (scikit-survival, 500 trees)
  4.  Gradient Boosting Survival (scikit-survival)
  5.  DeepSurv                   (pycox / PyTorch)
  6.  DeepHit                    (pycox / PyTorch, single-risk)
  [7. GCN-Cox (our model)        — injected from gcn_results]

All baselines use the SAME feature matrix as the full GCN model
(mRMR-reduced CNA + mRNA + Methylation + Clinical, concatenated)
and the SAME train/test split — results are directly comparable.

Features are StandardScaler-normalised before being passed to any
distance-based or gradient-based model (tree models use raw features).
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.naive_bayes   import GaussianNB
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier,
                                   HistGradientBoostingClassifier)
from sklearn.svm           import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics        import roc_auc_score

from config import RANDOM_STATE
from utils import concordance_index, plot_roc_curves

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _scale(X_train, X_test):
    """StandardScaler fit on train, applied to both."""
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


def _sksurv_y(events, times):
    """Build scikit-survival structured array target."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(events, times)],
        dtype=[('event', bool), ('time', float)]
    )


# ─────────────────────────────────────────────────────────────────────────────
# BINARY BASELINES
# ─────────────────────────────────────────────────────────────────────────────
def run_binary_baselines(X_train, y_train, X_test, y_test):
    """Fit 8 classifiers on training data, evaluate AUC on test set."""
    X_tr_sc, X_te_sc = _scale(X_train, X_test)

    classifiers = [
        ("Logistic Regression",
         LogisticRegression(C=0.1, class_weight='balanced',
                            max_iter=2000, random_state=RANDOM_STATE),
         True),
        ("Naive Bayes",
         GaussianNB(),
         True),
        ("K-Nearest Neighbours",
         KNeighborsClassifier(n_neighbors=11, weights='distance',
                              metric='euclidean'),
         True),
        ("Decision Tree",
         DecisionTreeClassifier(max_depth=5, class_weight='balanced',
                                random_state=RANDOM_STATE),
         False),
        ("Random Forest",
         RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                max_features='sqrt', random_state=RANDOM_STATE,
                                n_jobs=-1),
         False),
        ("Gradient Boosting",
         HistGradientBoostingClassifier(max_iter=300, max_depth=4,
                                        learning_rate=0.05,
                                        random_state=RANDOM_STATE),
         False),
        ("SVM (RBF)",
         SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
             class_weight='balanced', random_state=RANDOM_STATE),
         True),
        ("MLP / DNN",
         MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                       activation='relu', alpha=0.001,
                       max_iter=500, random_state=RANDOM_STATE,
                       early_stopping=True, validation_fraction=0.15),
         True),
    ]

    results = []
    print("\n── Binary Baselines ────────────────────────────────────────────")
    print(f"  {'Model':<28} {'Test AUC':>9}")
    print("  " + "-" * 40)

    for name, clf, scaled in classifiers:
        Xtr = X_tr_sc if scaled else X_train
        Xte = X_te_sc if scaled else X_test
        clf.fit(Xtr, y_train)
        probs = clf.predict_proba(Xte)[:, 1]
        auc   = roc_auc_score(y_test, probs)
        results.append({"name": name, "auc": auc,
                        "probs": probs, "y_true": y_test})
        print(f"  {name:<28} {auc:>9.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SURVIVAL BASELINES
# ─────────────────────────────────────────────────────────────────────────────
def run_survival_baselines(X_train, times_train, events_train,
                           X_test,  times_test,  events_test):
    """Fit 6 survival models, evaluate Harrell C-index on test set."""
    X_tr_sc, X_te_sc = _scale(X_train, X_test)
    results = []

    print("\n── Survival Baselines ──────────────────────────────────────────")
    print(f"  {'Model':<35} {'C-index':>9}")
    print("  " + "-" * 47)

    # ── 1. Classical CoxPH (L2) ──────────────────────────────────────────
    try:
        from lifelines import CoxPHFitter
        import pandas as pd

        cols = [f"f{i}" for i in range(X_tr_sc.shape[1])]
        df_tr = pd.DataFrame(X_tr_sc, columns=cols)
        df_tr['time']  = times_train
        df_tr['event'] = events_train.astype(int)
        df_te = pd.DataFrame(X_te_sc, columns=cols)

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_tr, duration_col='time', event_col='event')
        risk = cph.predict_partial_hazard(df_te).values
        ci   = concordance_index(risk, times_test, events_test)
        results.append({"name": "CoxPH (L2)", "cindex": ci})
        print(f"  {'CoxPH (L2)':<35} {ci:>9.4f}")
    except Exception as ex:
        print(f"  CoxPH failed: {ex}")

    # ── 2. Lasso-Cox ─────────────────────────────────────────────────────
    try:
        from lifelines import CoxPHFitter
        import pandas as pd

        cols = [f"f{i}" for i in range(X_tr_sc.shape[1])]
        df_tr = pd.DataFrame(X_tr_sc, columns=cols)
        df_tr['time']  = times_train
        df_tr['event'] = events_train.astype(int)
        df_te = pd.DataFrame(X_te_sc, columns=cols)

        lasso_cph = CoxPHFitter(penalizer=1.0, l1_ratio=1.0)
        lasso_cph.fit(df_tr, duration_col='time', event_col='event')
        risk = lasso_cph.predict_partial_hazard(df_te).values
        ci   = concordance_index(risk, times_test, events_test)
        results.append({"name": "Lasso-Cox", "cindex": ci})
        print(f"  {'Lasso-Cox':<35} {ci:>9.4f}")
    except Exception as ex:
        print(f"  Lasso-Cox failed: {ex}")

    # ── 3. Random Survival Forest ─────────────────────────────────────────
    try:
        from sksurv.ensemble import RandomSurvivalForest

        y_tr = _sksurv_y(events_train, times_train)
        y_te = _sksurv_y(events_test,  times_test)

        rsf = RandomSurvivalForest(n_estimators=500, max_features='sqrt',
                                   random_state=RANDOM_STATE, n_jobs=-1)
        rsf.fit(X_train, y_tr)
        risk = rsf.predict(X_test)
        ci   = concordance_index(risk, times_test, events_test)
        results.append({"name": "Random Survival Forest", "cindex": ci})
        print(f"  {'Random Survival Forest':<35} {ci:>9.4f}")
    except Exception as ex:
        print(f"  Random Survival Forest failed: {ex}")

    # ── 4. Gradient Boosting Survival ─────────────────────────────────────
    try:
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        y_tr = _sksurv_y(events_train, times_train)
        gbsa = GradientBoostingSurvivalAnalysis(
            n_estimators=200, learning_rate=0.05,
            max_depth=3, random_state=RANDOM_STATE)
        gbsa.fit(X_train, y_tr)
        risk = gbsa.predict(X_test)
        ci   = concordance_index(risk, times_test, events_test)
        results.append({"name": "GB Survival", "cindex": ci})
        print(f"  {'GB Survival':<35} {ci:>9.4f}")
    except Exception as ex:
        print(f"  GB Survival failed: {ex}")

    # ── 5. DeepSurv ───────────────────────────────────────────────────────
    try:
        import torchtuples as tt
        from pycox.models import CoxPH as PyCoxCoxPH

        Xtr_f = X_tr_sc.astype(np.float32)
        Xte_f = X_te_sc.astype(np.float32)
        y_ds_tr = (times_train.astype(np.float32),
                   events_train.astype(np.float32))

        net_ds = tt.practical.MLPVanilla(
            Xtr_f.shape[1], [64, 64], 1,
            batch_norm=True, dropout=0.3)
        model_ds = PyCoxCoxPH(net_ds, tt.optim.Adam(0.001))
        model_ds.fit(Xtr_f, y_ds_tr,
                     batch_size=64, epochs=100,
                     callbacks=[tt.callbacks.EarlyStopping(patience=15)],
                     verbose=False)

        risk = model_ds.predict(Xte_f).flatten()
        ci   = concordance_index(risk, times_test, events_test)
        results.append({"name": "DeepSurv", "cindex": ci})
        print(f"  {'DeepSurv':<35} {ci:>9.4f}")
    except Exception as ex:
        print(f"  DeepSurv failed: {ex}")

    # ── 6. DeepHit ────────────────────────────────────────────────────────
    try:
        import torchtuples as tt
        from pycox.models import DeepHitSingle

        num_durations = 20
        labtrans = DeepHitSingle.label_transform(num_durations)
        y_dh_tr  = labtrans.fit_transform(
            times_train.astype(np.float32),
            events_train.astype(np.float32))

        Xtr_f = X_tr_sc.astype(np.float32)
        Xte_f = X_te_sc.astype(np.float32)

        net_dh = tt.practical.MLPVanilla(
            Xtr_f.shape[1], [64, 64], labtrans.out_features,
            batch_norm=True, dropout=0.3)
        model_dh = DeepHitSingle(net_dh, tt.optim.Adam(0.001),
                                 alpha=0.2, sigma=0.1,
                                 duration_index=labtrans.cuts)
        model_dh.fit(Xtr_f, y_dh_tr,
                     batch_size=64, epochs=100,
                     callbacks=[tt.callbacks.EarlyStopping(patience=15)],
                     verbose=False)

        surv = model_dh.predict_surv_df(Xte_f)
        risk = -surv.index[surv.apply(
            lambda col: (col <= 0.5).idxmax()
        )].values
        ci   = concordance_index(risk, times_test, events_test)
        results.append({"name": "DeepHit", "cindex": ci})
        print(f"  {'DeepHit':<35} {ci:>9.4f}")
    except Exception as ex:
        print(f"  DeepHit failed: {ex}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def _plot_survival_bars(surv_results, gcn_cindex, output_dir, title_suffix=""):
    names = [r["name"] for r in surv_results] + ["GCN-Cox (ours)"]
    cis   = [r["cindex"] for r in surv_results] + [gcn_cindex]
    order   = np.argsort(cis)
    names_s = [names[i] for i in order]
    cis_s   = [cis[i]   for i in order]
    colors  = ["#1a1a1a" if "GCN" in n else "#4c72b0" for n in names_s]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names_s, cis_s, color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, cis_s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', ha='left', fontsize=10)

    ax.axvline(0.5, linestyle='--', color='red', linewidth=1.2,
               alpha=0.7, label='Random (0.50)')
    ax.set_xlabel("Harrell's C-index", fontsize=12)
    ax.set_title(f"GBM Survival Prediction\nC-index Comparison"
                 f"{' — ' + title_suffix if title_suffix else ''}",
                 fontsize=14)
    ax.set_xlim(0.3, min(1.0, max(cis_s) + 0.08))
    ax.legend(fontsize=10);  ax.grid(axis='x', alpha=0.3)

    path = f"{output_dir}/baseline_survival_cindex.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Survival C-index plot saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLES
# ─────────────────────────────────────────────────────────────────────────────
def _print_binary_summary(binary_results, gcn_auc):
    print("\n" + "=" * 55)
    print("  BINARY CLASSIFICATION — SUMMARY (Test AUC)")
    print("=" * 55)
    all_r = binary_results + [{"name": "GCN (ours)", "auc": gcn_auc}]
    for r in sorted(all_r, key=lambda x: -x["auc"]):
        marker = " <--" if "GCN" in r["name"] else ""
        print(f"  {r['name']:<30} {r['auc']:.4f}{marker}")
    print("=" * 55)


def _print_survival_summary(surv_results, gcn_cindex):
    print("\n" + "=" * 55)
    print("  SURVIVAL MODELS — SUMMARY (C-index)")
    print("=" * 55)
    all_r = surv_results + [{"name": "GCN-Cox (ours)", "cindex": gcn_cindex}]
    for r in sorted(all_r, key=lambda x: -x["cindex"]):
        marker = " <--" if "GCN" in r["name"] else ""
        print(f"  {r['name']:<35} {r['cindex']:.4f}{marker}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_baseline_comparison(pipeline: dict, gcn_results: dict,
                             output_dir: str = "plots",
                             title_suffix: str = ""):
    """
    Run all baseline comparisons and generate plots.

    Parameters
    ----------
    pipeline     : dict returned by main.main()
    gcn_results  : dict returned by train_gcn()
    output_dir   : directory to save plots
    title_suffix : optional string added to plot titles (e.g. 'LTS=12m')
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train = np.hstack([pipeline["cna_tr_r"],  pipeline["mrna_tr_r"],
                          pipeline["meth_tr_r"], pipeline["clin_tr_arr"]])
    X_test  = np.hstack([pipeline["cna_te"],     pipeline["mrna_te"],
                          pipeline["meth_te"],    pipeline["clin_te"]])

    y_train = pipeline["y_train"].values
    y_test  = pipeline["y_test"].values
    t_train = pipeline["os_months_train"].values.astype(float)
    e_train = pipeline["os_status_train"].values.astype(float)
    t_test  = pipeline["os_months_test"].values.astype(float)
    e_test  = pipeline["os_status_test"].values.astype(float)

    print("\n" + "=" * 62)
    print("  BASELINE COMPARISON" + (f" — {title_suffix}" if title_suffix else ""))
    print("=" * 62)
    print(f"  Feature matrix: {X_train.shape[1]} features "
          f"(CNA+mRNA+Meth+Clinical, mRMR-reduced)")
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")
    print(f"  LTS train={int(y_train.sum())} | LTS test={int(y_test.sum())}")
    print("=" * 62)

    gcn_auc    = gcn_results["auc"]
    gcn_probs  = gcn_results["probs"]
    gcn_ytrue  = gcn_results["y_true"]
    gcn_cindex = gcn_results["cindex"]

    binary_results = run_binary_baselines(X_train, y_train, X_test, y_test)
    surv_results   = run_survival_baselines(X_train, t_train, e_train,
                                            X_test,  t_test,  e_test)

    _print_binary_summary(binary_results, gcn_auc)
    _print_survival_summary(surv_results, gcn_cindex)

    # ── ROC plot — baselines + GCN as last entry (black) ─────────────────────
    # Rename 'name' → 'label' for plot_roc_curves compatibility, then append GCN
    roc_entries = [{"label": r["name"], "probs": r["probs"],
                    "y_true": r["y_true"], "auc": r["auc"]}
                   for r in binary_results]
    roc_entries.append({"label": "GCN (ours)", "probs": gcn_probs,
                        "y_true": gcn_ytrue,   "auc": gcn_auc})

    baseline_colors = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#000000",
    ]
    title = (f"GBM LTS Binary Classification\nROC Curve Comparison"
             f"{' — ' + title_suffix if title_suffix else ''}")
    plot_roc_curves(roc_entries,
                    output_path=f"{output_dir}/baseline_binary_roc.png",
                    title=title,
                    colors=baseline_colors)

    _plot_survival_bars(surv_results, gcn_cindex, output_dir, title_suffix)

    return {"binary": binary_results, "survival": surv_results}