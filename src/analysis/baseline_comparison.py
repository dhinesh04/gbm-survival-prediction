"""Compares the GCN dual-head model against classical AFT and deep-learning
survival baselines (Weibull AFT, Log-Normal AFT, Random Survival Forest,
DeepSurv, DeepHit) on the same held-out test set, using Harrell's C-index
and Integrated Brier Score (IBS).

Two comparisons are run:
  1. Single train/test split (legacy, kept for continuity with existing
     report figures) -- one point estimate per model, not on its own strong
     enough to claim one model beats another.
  2. 5-fold CV (new) -- every model refit from scratch per fold, using the
     same StratifiedKFold split (stratified on event status, same
     random_state) as gcn_train.py / ablation_studies.py. Reports mean±std
     and is what should drive any "model X beats model Y" claim.

Features are StandardScaler-normalised before any distance- or
gradient-based model; the scaler is refit per fold for the CV comparison.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.interpolate import interp1d

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

from config import RANDOM_STATE, N_FOLDS
from src.utils import concordance_index

warnings.filterwarnings("ignore")


def _scale(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


def _sksurv_y(events, times):
    return Surv.from_arrays(event=events.astype(bool), time=times.astype(float))


def _interpolate_surv(times, surv_matrix, time_grid):
    """Interpolate pycox survival curves (times x patients) onto the shared
    IBS time_grid, clipped to [0, 1] to avoid extrapolation artifacts."""
    f = interp1d(times, surv_matrix, axis=0, fill_value="extrapolate")
    return np.clip(f(time_grid).T, 0.0, 1.0)


# per-model (train, val) fit+score functions, shared by the single-split and
# CV comparisons below so each model's fitting logic lives in exactly one place
def _cv_weibull_aft(X_tr, t_tr, e_tr, X_val, t_val, e_val,
                    time_grid, y_tr_sksurv, y_val_sksurv):
    from lifelines import WeibullAFTFitter
    cols  = [f"f{i}" for i in range(X_tr.shape[1])]
    df_tr = pd.DataFrame(X_tr, columns=cols)
    df_tr['time'], df_tr['event'] = t_tr, e_tr
    df_val = pd.DataFrame(X_val, columns=cols)

    weibull = WeibullAFTFitter(penalizer=0.5)  # higher penalizer to force convergence on collinear omics data
    weibull.fit(df_tr, duration_col='time', event_col='event')

    risk = -weibull.predict_expectation(df_val).values  # longer expected survival = lower risk
    ci   = concordance_index(risk, t_val, e_val)

    ibs = float('nan')
    if time_grid is not None:
        try:
            surv_df = weibull.predict_survival_function(df_val, times=time_grid)
            ibs = integrated_brier_score(y_tr_sksurv, y_val_sksurv, surv_df.values.T, time_grid)
        except Exception:
            ibs = float('nan')
    return ci, ibs


def _cv_lognormal_aft(X_tr, t_tr, e_tr, X_val, t_val, e_val,
                      time_grid, y_tr_sksurv, y_val_sksurv):
    from lifelines import LogNormalAFTFitter
    cols  = [f"f{i}" for i in range(X_tr.shape[1])]
    df_tr = pd.DataFrame(X_tr, columns=cols)
    df_tr['time'], df_tr['event'] = t_tr, e_tr
    df_val = pd.DataFrame(X_val, columns=cols)

    lognorm = LogNormalAFTFitter(penalizer=0.1)
    lognorm.fit(df_tr, duration_col='time', event_col='event')

    risk = -lognorm.predict_expectation(df_val).values
    ci   = concordance_index(risk, t_val, e_val)

    ibs = float('nan')
    if time_grid is not None:
        try:
            surv_df = lognorm.predict_survival_function(df_val, times=time_grid)
            ibs = integrated_brier_score(y_tr_sksurv, y_val_sksurv, surv_df.values.T, time_grid)
        except Exception:
            ibs = float('nan')
    return ci, ibs


def _cv_rsf(X_tr, t_tr, e_tr, X_val, t_val, e_val,
           time_grid, y_tr_sksurv, y_val_sksurv):
    from sksurv.ensemble import RandomSurvivalForest
    rsf = RandomSurvivalForest(n_estimators=500, max_features='sqrt',
                               random_state=RANDOM_STATE, n_jobs=-1)
    rsf.fit(X_tr, y_tr_sksurv)

    risk = rsf.predict(X_val)
    ci   = concordance_index(risk, t_val, e_val)

    ibs = float('nan')
    if time_grid is not None:
        try:
            surv_funcs  = rsf.predict_survival_function(X_val)
            surv_matrix = np.array([fn(time_grid) for fn in surv_funcs])
            ibs = integrated_brier_score(y_tr_sksurv, y_val_sksurv, surv_matrix, time_grid)
        except Exception:
            ibs = float('nan')
    return ci, ibs


def _cv_deepsurv(X_tr, t_tr, e_tr, X_val, t_val, e_val,
                time_grid, y_tr_sksurv, y_val_sksurv):
    import torchtuples as tt
    from pycox.models import CoxPH as PyCoxCoxPH

    Xtr_f, Xval_f = X_tr.astype(np.float32), X_val.astype(np.float32)
    y_ds_tr = (t_tr.astype(np.float32), e_tr.astype(np.float32))

    net_ds   = tt.practical.MLPVanilla(Xtr_f.shape[1], [64, 64], 1, batch_norm=True, dropout=0.3)
    model_ds = PyCoxCoxPH(net_ds, tt.optim.Adam(0.001))
    model_ds.fit(Xtr_f, y_ds_tr, batch_size=64, epochs=100,
                callbacks=[tt.callbacks.EarlyStopping(patience=15)], verbose=False)
    _ = model_ds.compute_baseline_hazards()  # needed before it can output a survival curve S(t)

    risk = model_ds.predict(Xval_f).flatten()
    ci   = concordance_index(risk, t_val, e_val)

    ibs = float('nan')
    if time_grid is not None:
        try:
            surv_df_ds  = model_ds.predict_surv_df(Xval_f)
            surv_matrix = _interpolate_surv(surv_df_ds.index.values, surv_df_ds.values, time_grid)
            ibs = integrated_brier_score(y_tr_sksurv, y_val_sksurv, surv_matrix, time_grid)
        except Exception:
            ibs = float('nan')
    return ci, ibs


def _cv_deephit(X_tr, t_tr, e_tr, X_val, t_val, e_val,
               time_grid, y_tr_sksurv, y_val_sksurv):
    import torchtuples as tt
    from pycox.models import DeepHitSingle

    Xtr_f, Xval_f = X_tr.astype(np.float32), X_val.astype(np.float32)
    labtrans = DeepHitSingle.label_transform(20)
    y_dh_tr  = labtrans.fit_transform(t_tr.astype(np.float32), e_tr.astype(np.float32))

    net_dh   = tt.practical.MLPVanilla(Xtr_f.shape[1], [64, 64], labtrans.out_features,
                                       batch_norm=True, dropout=0.3)
    model_dh = DeepHitSingle(net_dh, tt.optim.Adam(0.001), alpha=0.2, sigma=0.1,
                             duration_index=labtrans.cuts)
    model_dh.fit(Xtr_f, y_dh_tr, batch_size=64, epochs=100,
                callbacks=[tt.callbacks.EarlyStopping(patience=15)], verbose=False)

    surv_df_dh = model_dh.predict_surv_df(Xval_f)

    def _median_surv(col):
        below = (col <= 0.5).values
        return col.index[below.argmax()] if below.any() else col.index[-1]

    risk = -surv_df_dh.apply(_median_surv).values
    ci   = concordance_index(risk, t_val, e_val)

    ibs = float('nan')
    if time_grid is not None:
        try:
            surv_matrix = _interpolate_surv(surv_df_dh.index.values, surv_df_dh.values, time_grid)
            ibs = integrated_brier_score(y_tr_sksurv, y_val_sksurv, surv_matrix, time_grid)
        except Exception:
            ibs = float('nan')
    return ci, ibs


_CV_MODEL_FNS = [
    ("Weibull AFT",    _cv_weibull_aft),
    ("Log-Normal AFT", _cv_lognormal_aft),
    ("RSF",            _cv_rsf),
    ("DeepSurv",       _cv_deepsurv),
    ("DeepHit",        _cv_deephit),
]


def run_survival_baselines(X_train, times_train, events_train,
                           X_test,  times_test,  events_test):
    """Fit each baseline once on the full training set, evaluate once on the
    held-out test set. Single point estimate -- see run_survival_baselines_cv()
    for the comparison that should actually drive any claim."""
    X_tr_sc, X_te_sc = _scale(X_train, X_test)
    y_tr_sksurv = _sksurv_y(events_train, times_train)
    y_te_sksurv = _sksurv_y(events_test, times_test)

    # IPCW time grid must stay strictly inside the observed training times
    t_min = max(times_train[events_train==1].min(), times_test[events_test==1].min()) + 0.1
    t_max = min(times_train[events_train==1].max(), times_test.max()) - 0.1
    time_grid = np.linspace(t_min, t_max, 100)

    results = []
    print("\n── Survival Baselines — single train/test split (legacy) ───────")
    print(f"  {'Model':<30} {'C-index':>9} {'IBS':>9}")
    print("  " + "-" * 50)

    for name, fn in _CV_MODEL_FNS:
        try:
            ci, ibs = fn(X_tr_sc, times_train, events_train,
                        X_te_sc, times_test, events_test,
                        time_grid, y_tr_sksurv, y_te_sksurv)
            results.append({"name": name, "cindex": ci, "ibs": ibs})
            print(f"  {name:<30} {ci:>9.4f} {ibs:>9.4f}")
        except Exception as ex:
            print(f"  {name} failed: {ex}")

    return results, time_grid, y_tr_sksurv, y_te_sksurv


def run_survival_baselines_cv(X_train, times_train, events_train, n_folds=N_FOLDS):
    """5-fold CV evaluation of every classical/deep survival baseline, using the
    same StratifiedKFold split (event status, same random_state) as gcn_train.py
    / ablation_studies.py so the resulting mean±std is directly comparable to
    the GCN's own CV numbers. Each model is refit per fold with its own
    StandardScaler (fit on that fold only, no leakage); the IBS time grid is
    rebuilt per fold, bounded to that fold's own event times.

    Returns a list of dicts: name, cv_cindex_mean, cv_cindex_std, cv_ibs_mean, cv_ibs_std.
    """
    fold_cindex = {name: [] for name, _ in _CV_MODEL_FNS}
    fold_ibs    = {name: [] for name, _ in _CV_MODEL_FNS}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n── Survival Baselines — {n_folds}-fold CV "
          f"(stratified on event status, same split as GCN's own CV) ──")

    for fold_i, (tr_idx, val_idx) in enumerate(
            skf.split(X_train, events_train.astype(int)), start=1):

        X_fold_tr, X_fold_val = X_train[tr_idx], X_train[val_idx]
        t_fold_tr, t_fold_val = times_train[tr_idx], times_train[val_idx]
        e_fold_tr, e_fold_val = events_train[tr_idx], events_train[val_idx]

        X_fold_tr_sc, X_fold_val_sc = _scale(X_fold_tr, X_fold_val)
        y_fold_tr_sksurv  = _sksurv_y(e_fold_tr, t_fold_tr)
        y_fold_val_sksurv = _sksurv_y(e_fold_val, t_fold_val)

        # upper bound uses the fold's overall max time (event + censored), not
        # just its max event time -- a censored longest-follow-up patient would
        # otherwise produce a grid that silently exceeds the true IPCW bound
        fold_time_grid = None
        try:
            ft_min_bound = max(t_fold_tr[e_fold_tr == 1].min(),
                               t_fold_val[e_fold_val == 1].min())
            ft_max_bound = min(t_fold_tr.max(), t_fold_val.max())
            margin = max(0.1, 0.01 * (ft_max_bound - ft_min_bound))
            ft_min = ft_min_bound + margin
            ft_max = ft_max_bound - margin
            if ft_max > ft_min:
                fold_time_grid = np.linspace(ft_min, ft_max, 50)
        except Exception:
            fold_time_grid = None

        grid_note = "" if fold_time_grid is not None else "  [IBS grid degenerate this fold]"
        print(f"  Fold {fold_i}/{n_folds}  (train={len(tr_idx)}, val={len(val_idx)}){grid_note}")

        for name, fn in _CV_MODEL_FNS:
            try:
                ci, ibs = fn(X_fold_tr_sc, t_fold_tr, e_fold_tr,
                            X_fold_val_sc, t_fold_val, e_fold_val,
                            fold_time_grid, y_fold_tr_sksurv, y_fold_val_sksurv)
            except Exception as ex:
                print(f"    {name} failed on fold {fold_i}: {ex}")
                ci, ibs = float('nan'), float('nan')
            fold_cindex[name].append(ci)
            fold_ibs[name].append(ibs)

    cv_results = []
    for name, _ in _CV_MODEL_FNS:
        ci_vals  = [v for v in fold_cindex[name] if not np.isnan(v)]
        ibs_vals = [v for v in fold_ibs[name]    if not np.isnan(v)]
        row = {
            "name": name,
            "cv_cindex_mean": float(np.mean(ci_vals))  if ci_vals  else float('nan'),
            "cv_cindex_std":  float(np.std(ci_vals))   if ci_vals  else float('nan'),
            "cv_ibs_mean":    float(np.mean(ibs_vals)) if ibs_vals else float('nan'),
            "cv_ibs_std":     float(np.std(ibs_vals))  if ibs_vals else float('nan'),
        }
        cv_results.append(row)
        print(f"  {name:<16}  CV C-idx={row['cv_cindex_mean']:.4f}±{row['cv_cindex_std']:.4f}  "
              f"CV IBS={row['cv_ibs_mean']:.4f}±{row['cv_ibs_std']:.4f}")

    return cv_results


def _plot_metrics(results_list, metric_key, xlabel, title, output_path, reverse_sort=False):
    """Minimalist horizontal barplot for the single-split comparison."""
    names = [r["name"] for r in results_list]
    vals  = [r[metric_key] for r in results_list]

    order = np.argsort(vals)
    if reverse_sort:
        order = order[::-1]

    names_s = [names[i] for i in order]
    vals_s  = [vals[i]   for i in order]
    colors  = ["#2b2b2b" if "GCN" in n else "#e0e0e0" for n in names_s]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')

    bars = ax.barh(names_s, vals_s, color=colors, height=0.5)

    for bar, val in zip(bars, vals_s):
        ax.text(bar.get_width() + (0.01 if not reverse_sort else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', ha='left' if not reverse_sort else 'right',
                fontsize=9, color="#2b2b2b")

    ax.set_xlabel(xlabel, fontsize=10, color="#4a4a4a")
    ax.set_title(title, fontsize=12, pad=15)

    if metric_key == "cindex":
        ax.axvline(0.5, linestyle='--', color='#a0a0a0', linewidth=1.0, zorder=0)
        ax.set_xlim(0.3, min(1.0, max(vals_s) + 0.1))
    else:
        ax.set_xlim(0.0, max(vals_s) + 0.05)  # IBS: lower is better

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def _plot_cv_metrics(cv_results, gcn_cv_row, output_dir):
    all_rows = list(cv_results)
    if gcn_cv_row is not None:
        all_rows.append(gcn_cv_row)

    names     = [r["name"] for r in all_rows]
    ci_means  = [r["cv_cindex_mean"] for r in all_rows]
    ci_stds   = [r["cv_cindex_std"]  for r in all_rows]
    ibs_means = [r["cv_ibs_mean"]    for r in all_rows]
    ibs_stds  = [r["cv_ibs_std"]     for r in all_rows]
    colors    = ["#2b2b2b" if "GCN" in n else "#9fb8d8" for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    order_ci = np.argsort(ci_means)
    axes[0].barh([names[i] for i in order_ci], [ci_means[i] for i in order_ci],
                xerr=[ci_stds[i] for i in order_ci],
                color=[colors[i] for i in order_ci], capsize=4)
    axes[0].set_xlabel("CV C-index (mean ± std)")
    axes[0].set_title("Risk Ranking — 5-fold CV")
    axes[0].axvline(0.5, linestyle='--', color='red', alpha=0.4)

    valid_ibs_idx = [i for i, v in enumerate(ibs_means) if not np.isnan(v)]
    order_ibs = sorted(valid_ibs_idx, key=lambda i: ibs_means[i], reverse=True)
    axes[1].barh([names[i] for i in order_ibs], [ibs_means[i] for i in order_ibs],
                xerr=[ibs_stds[i] for i in order_ibs],
                color=[colors[i] for i in order_ibs], capsize=4)
    axes[1].set_xlabel("CV IBS (mean ± std, lower better)")
    axes[1].set_title("Calibration — 5-fold CV")

    plt.tight_layout()
    path = os.path.join(output_dir, "baseline_cv_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  CV comparison plot saved → {path}")


def _print_summary(all_results):
    print("\n" + "=" * 55)
    print("  SURVIVAL MODELS — SINGLE-SPLIT SUMMARY (legacy)")
    print("=" * 55)
    print(f"  {'Model':<30} {'C-index':>9} {'IBS':>9}")
    print("  " + "-" * 51)

    for r in sorted(all_results, key=lambda x: x.get("ibs", float('inf'))
                     if x.get("ibs") is not None else float('inf')):
        marker = " <--" if "GCN" in r["name"] else ""
        c_str = f"{r['cindex']:.4f}" if r.get("cindex") is not None else "N/A"
        ibs_str = f"{r['ibs']:.4f}" if r.get("ibs") is not None else "N/A"
        print(f"  {r['name']:<30} {c_str:>9} {ibs_str:>9}{marker}")
    print("=" * 55)
    print("  ⚠ Single point estimate per model — see the CV summary below "
          "before drawing any conclusion about which model is better.")


def _print_cv_summary(cv_results, gcn_cv_row=None):
    print("\n" + "=" * 70)
    print("  SURVIVAL MODELS — 5-FOLD CV SUMMARY (mean ± std)")
    print("=" * 70)
    all_rows = list(cv_results)
    if gcn_cv_row is not None:
        all_rows.append(gcn_cv_row)

    print(f"  {'Model':<22} {'CV C-index':>16}  {'CV IBS':>16}")
    print("  " + "-" * 58)
    for r in sorted(all_rows, key=lambda x: x["cv_ibs_mean"]
                     if not np.isnan(x["cv_ibs_mean"]) else float('inf')):
        marker  = " <--" if "GCN" in r["name"] else ""
        ci_str  = f"{r['cv_cindex_mean']:.4f}±{r['cv_cindex_std']:.4f}"
        ibs_str = f"{r['cv_ibs_mean']:.4f}±{r['cv_ibs_std']:.4f}"
        print(f"  {r['name']:<22} {ci_str:>16}  {ibs_str:>16}{marker}")
    print("=" * 70)

    if gcn_cv_row is not None:
        best_baseline_ci = max(cv_results, key=lambda r: r["cv_cindex_mean"])
        gcn_lower  = gcn_cv_row["cv_cindex_mean"]      - gcn_cv_row["cv_cindex_std"]
        base_upper = best_baseline_ci["cv_cindex_mean"] + best_baseline_ci["cv_cindex_std"]
        overlap = gcn_lower <= base_upper

        print(f"\n  Best baseline CV C-index: {best_baseline_ci['name']} "
              f"({best_baseline_ci['cv_cindex_mean']:.4f}±{best_baseline_ci['cv_cindex_std']:.4f})")
        print(f"  GCN CV C-index:           "
              f"{gcn_cv_row['cv_cindex_mean']:.4f}±{gcn_cv_row['cv_cindex_std']:.4f}")
        if overlap:
            print(f"  ⚠ GCN's std band overlaps the best baseline's — the "
                  f"apparent GCN advantage is not clearly distinguishable "
                  f"from CV fold-to-fold noise.")
        else:
            print(f"  ✓ GCN's std band does NOT overlap the best baseline's "
                  f"— GCN's advantage holds up under CV.")
    else:
        print(f"\n  [GCN CV row unavailable — see warning above. Only "
              f"baseline-vs-baseline CV comparison shown; GCN cannot yet "
              f"be placed on this table.]")
    print("=" * 70)


def run_baseline_comparison(pipeline: dict, gcn_results: dict, output_dir: str = "plots"):
    """Run survival baselines against GCN AFT/Cox outputs, both as a single
    train/test split (legacy) and as 5-fold CV (the comparison that should
    actually drive any "model X beats model Y" claim).

    For the CV comparison to include GCN's own row, gcn_results needs
    cv_val_cindex_mean/std and cv_val_ibs_mean/std (CV mean/std across the
    same 5 folds GCN was trained on); otherwise the CV table still runs for
    the baselines alone, with a warning that GCN's row is unavailable.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_train = np.hstack([pipeline["cna_tr_r"],  pipeline["mrna_tr_r"],
                         pipeline["meth_tr_r"], pipeline["clin_tr_arr"]])
    X_test  = np.hstack([pipeline["cna_te"],    pipeline["mrna_te"],
                         pipeline["meth_te"],   pipeline["clin_te"]])

    t_train = pipeline["os_months_train"].values.astype(float)
    e_train = pipeline["os_status_train"].values.astype(float)
    t_test  = pipeline["os_months_test"].values.astype(float)
    e_test  = pipeline["os_status_test"].values.astype(float)

    print("\n" + "=" * 62)
    print("  BASELINE COMPARISON (CONTINUOUS SURVIVAL)")
    print("=" * 62)

    surv_results, time_grid, y_tr_sksurv, y_te_sksurv = run_survival_baselines(
        X_train, t_train, e_train, X_test, t_test, e_test
    )

    gcn_cindex = gcn_results.get("cindex", None)
    gcn_ibs    = None
    if "pred_log_t" in gcn_results and "sigma" in gcn_results:
        try:
            mu_test = np.array(gcn_results["pred_log_t"]).reshape(-1, 1)
            sigma   = float(gcn_results["sigma"])
            log_t   = np.log(time_grid).reshape(1, -1)
            gcn_surv_matrix = norm.sf((log_t - mu_test) / sigma)
            gcn_ibs = integrated_brier_score(y_tr_sksurv, y_te_sksurv, gcn_surv_matrix, time_grid)
        except Exception as e:
            print(f"\n  [Warning] Could not calculate GCN IBS (single split): {e}")

    gcn_entry = {"name": "GCN AFT+Cox (ours)", "cindex": gcn_cindex, "ibs": gcn_ibs}
    all_results = surv_results + [gcn_entry]
    _print_summary(all_results)

    valid_cindex = [r for r in all_results if r["cindex"] is not None]
    valid_ibs    = [r for r in all_results if r["ibs"] is not None]
    if valid_cindex:
        _plot_metrics(valid_cindex, "cindex", "Harrell's C-index (Higher = Better)",
                      "Model Risk Ranking Calibration", f"{output_dir}/baseline_cindex.png")
    if valid_ibs:
        _plot_metrics(valid_ibs, "ibs", "Integrated Brier Score (Lower = Better)",
                      "Time-to-Event Absolute Accuracy", f"{output_dir}/baseline_ibs.png",
                      reverse_sort=True)

    print("\n" + "=" * 62)
    print("  BASELINE COMPARISON — 5-FOLD CV")
    print("=" * 62)
    cv_results = run_survival_baselines_cv(X_train, t_train, e_train, n_folds=N_FOLDS)

    gcn_cv_row = None
    required_keys = ["cv_val_cindex_mean", "cv_val_cindex_std",
                     "cv_val_ibs_mean", "cv_val_ibs_std"]
    if all(k in gcn_results for k in required_keys):
        gcn_cv_row = {
            "name": "GCN AFT+Cox (ours)",
            "cv_cindex_mean": gcn_results["cv_val_cindex_mean"],
            "cv_cindex_std":  gcn_results["cv_val_cindex_std"],
            "cv_ibs_mean":    gcn_results["cv_val_ibs_mean"],
            "cv_ibs_std":     gcn_results["cv_val_ibs_std"],
        }
    else:
        missing = [k for k in required_keys if k not in gcn_results]
        print(f"\n  ⚠ gcn_results is missing {missing} — GCN's own CV "
              f"mean±std cannot be placed alongside the baselines' CV "
              f"numbers yet. gcn_train.py needs to track these four "
              f"values (CV C-index and CV IBS, mean and std across the "
              f"5 folds it already trains) and include them in its "
              f"returned dict. Showing baseline-vs-baseline CV only "
              f"until then.")

    _print_cv_summary(cv_results, gcn_cv_row)
    _plot_cv_metrics(cv_results, gcn_cv_row, output_dir)

    return {
        "single_split": all_results,
        "cv": cv_results,
        "gcn_cv": gcn_cv_row,
    }
