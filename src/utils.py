"""
utils.py
--------
Shared utility functions used across the pipeline.

Imported by: gcn_train.py, ablation_studies.py,
             baseline_comparison.py, km_plot.py

Functions
---------
  concordance_index()            — Harrell C-index (numpy, no torch)
  cox_partial_likelihood_loss()  — Cox partial likelihood (torch)
  aft_loss()                     — log-normal AFT negative log-likelihood (torch)
  normalise_adjacency()          — D^-0.5 A D^-0.5 normalisation
  attach_test_nodes()            — k-NN attach test patients to PSN
  find_best_threshold()          — macro-F1 threshold sweep (legacy, binary-head era)
  compute_class_weights()        — inverse-frequency class weights (legacy, binary-head era)
  significance_stars()           — p-value → *** / ** / * / n.s.
  plot_roc_curves()              — smooth interpolated ROC figure (legacy, binary-head era)
"""

import os
import math
import numpy as np
import torch
import matplotlib
import itertools
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve


# ─────────────────────────────────────────────────────────────────────────────
# SURVIVAL METRICS
# ─────────────────────────────────────────────────────────────────────────────
def concordance_index(risk_scores: np.ndarray,
                      times: np.ndarray,
                      events: np.ndarray) -> float:
    """
    Harrell's C-index.

    Counts concordant comparable pairs where the patient who died sooner
    has the higher predicted risk score.

    Parameters
    ----------
    risk_scores : (N,) predicted risk (higher = more risk)
    times       : (N,) survival times in months
    events      : (N,) 1 = event observed, 0 = censored

    Returns
    -------
    float in [0, 1]. Returns 0.5 (random) if no comparable pairs exist.
    """
    n = len(times)
    concordant = comparable = 0
    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = events[i], events[j]
            if ei == 0 and ej == 0:
                continue
            if ei == 1 and ej == 1:
                if times[i] == times[j]:
                    continue
                comparable += 1
                concordant += (risk_scores[i] > risk_scores[j]) \
                              if times[i] < times[j] \
                              else (risk_scores[i] < risk_scores[j])
            elif ei == 1:
                if times[i] >= times[j]:
                    continue
                comparable += 1
                concordant += (risk_scores[i] > risk_scores[j])
            else:
                if times[j] >= times[i]:
                    continue
                comparable += 1
                concordant += (risk_scores[i] < risk_scores[j])
    return concordant / comparable if comparable > 0 else 0.5


def cox_partial_likelihood_loss(risk_scores: torch.Tensor,
                                times: torch.Tensor,
                                events: torch.Tensor) -> torch.Tensor:
    """
    Breslow approximation of the Cox partial likelihood loss.

    Patients are sorted by descending survival time. For each observed
    event, the loss penalises the model if the event patient does not have
    a higher risk score than the other patients still at risk.

    Parameters
    ----------
    risk_scores : (N,) torch float — predicted log-hazard
    times       : (N,) torch float — survival times
    events      : (N,) torch float — 1 = event observed, 0 = censored

    Returns
    -------
    Scalar torch tensor (differentiable).
    """
    sort_idx    = torch.argsort(times, descending=True)
    risk_scores = risk_scores[sort_idx]
    events      = events[sort_idx]
    log_cumsum  = torch.logcumsumexp(risk_scores, dim=0)
    n_events    = events.sum()
    if n_events == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -(risk_scores - log_cumsum)[events == 1].mean()


def aft_loss(pred_log_t: torch.Tensor,
             times: torch.Tensor,
             events: torch.Tensor,
             log_sigma: torch.Tensor) -> torch.Tensor:
    """
    Log-normal AFT negative log-likelihood, in log-time space.

    Model:  log(T) ~ Normal(mu(x), sigma^2)
        mu(x) = pred_log_t   — per-patient location, output by the network
        sigma = exp(log_sigma) — a single GLOBAL scale parameter, shared
                across all patients, learned jointly with the network
                (NOT predicted per patient — see gcn_model.py).

    For uncensored patients (event = 1), the per-patient term is the
    negative log-density of Normal(mu, sigma^2) evaluated at the observed
    log(t):
        NLL_i = log(sigma) + 0.5*log(2*pi) + (log(t_i) - mu_i)^2 / (2*sigma^2)

    For censored patients (event = 0), the per-patient term is the
    negative log of the survival function S(t_obs) = P(T > t_obs):
        z_i   = (mu_i - log(t_obs_i)) / sigma
        NLL_i = -log(Phi(z_i))
    where Phi is the standard normal CDF (computed via torch.erf). As mu_i
    grows relative to log(t_obs_i), Phi(z_i) -> 1 and the penalty vanishes
    (consistent with what's known about the censored patient — they were
    alive at least until t_obs). As mu_i shrinks below log(t_obs_i),
    Phi(z_i) -> 0 and the penalty grows without bound (the model is
    claiming the patient died before they're known to have been alive).

    Note: the Jacobian term from converting the density of log(T) to the
    density of T itself (a function of t alone) is omitted. It's constant
    with respect to (mu, sigma), so dropping it changes the absolute loss
    value by a constant offset but does not change the gradient or the
    optimal (mu, sigma) — standard practice when treating log(T) directly
    as the regression target.

    This replaces the earlier fixed-variance squared-error approximation
    (which is the sigma=1, no-censoring-likelihood special case of this
    loss). Learning sigma lets the model express how much spread genuinely
    exists in survival times, rather than assuming a fixed amount.

    Parameters
    ----------
    pred_log_t : (N,) predicted location mu(x) = log(t-hat)
    times      : (N,) observed survival / censoring times in months
    events     : (N,) 1 = deceased (uncensored), 0 = censored
    log_sigma  : scalar torch tensor — the model's learnable log(sigma)

    Returns
    -------
    Scalar differentiable tensor (mean NLL over patients).
    """
    sigma = torch.exp(log_sigma).clamp(min=1e-3)
    log_t = torch.log(times.clamp(min=1e-8))

    # Uncensored: negative log-density of Normal(mu, sigma^2) at log_t
    resid          = log_t - pred_log_t
    nll_uncensored = log_sigma + 0.5 * math.log(2 * math.pi) + (resid ** 2) / (2 * sigma ** 2)

    # Censored: negative log-survival, S(t_obs) = Phi((mu - log_t) / sigma)
    z            = (pred_log_t - log_t) / sigma
    phi_z        = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    phi_z        = phi_z.clamp(min=1e-7)          # avoid log(0)
    nll_censored = -torch.log(phi_z)

    per_patient = torch.where(events == 1, nll_uncensored, nll_censored)
    return per_patient.mean()


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def normalise_adjacency(psn: np.ndarray,
                        threshold: float = 0.0) -> torch.Tensor:
    """
    Symmetric normalisation: A_norm = D^-0.5 * (A + I) * D^-0.5

    Edges below `threshold` are zeroed before normalisation.
    Self-loops are added via identity matrix.

    Parameters
    ----------
    psn       : (N, N) raw affinity matrix
    threshold : edges below this value are set to 0

    Returns
    -------
    torch.FloatTensor (N, N)
    """
    A = psn.copy()
    np.fill_diagonal(A, 0)
    A[A < threshold] = 0
    A = A + np.eye(A.shape[0])
    rowsum     = A.sum(axis=1)
    D_inv_sqrt = np.diag(rowsum ** -0.5)
    return torch.tensor(D_inv_sqrt @ A @ D_inv_sqrt, dtype=torch.float)


def attach_test_nodes(psn_train: np.ndarray,
                      X_train: np.ndarray,
                      X_test: np.ndarray,
                      k: int = 10) -> np.ndarray:
    """
    Attach test patients to the training PSN via k-NN cosine similarity.

    Test nodes are connected to their k most similar training patients.
    The resulting (n_train + n_test) × (n_train + n_test) adjacency
    matrix preserves the original training PSN in the top-left block.

    Parameters
    ----------
    psn_train : (n_train, n_train) training PSN
    X_train   : (n_train, F) training feature matrix
    X_test    : (n_test,  F) test feature matrix
    k         : number of training neighbours per test node

    Returns
    -------
    full_adj : (n_train + n_test, n_train + n_test) numpy array
    """
    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]
    n_total = n_train + n_test

    tr_norm   = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    te_norm   = X_test  / (np.linalg.norm(X_test,  axis=1, keepdims=True) + 1e-8)
    sim_te_tr = te_norm @ tr_norm.T

    full_adj = np.zeros((n_total, n_total))
    full_adj[:n_train, :n_train] = psn_train

    for i in range(n_test):
        for j in np.argsort(sim_te_tr[i])[::-1][:k]:
            v = float(sim_te_tr[i, j])
            full_adj[n_train + i, j]         = v
            full_adj[j,           n_train + i] = v

    return full_adj


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION UTILITIES (legacy — binary-head era, retained for
# ablation_studies.py / baseline_comparison.py pending their own update)
# ─────────────────────────────────────────────────────────────────────────────
def find_best_threshold(probs: np.ndarray,
                        y_true: np.ndarray) -> float:
    """
    Sweep decision thresholds from 0.20 to 0.75 and return the value
    that maximises macro-F1 on the given predictions.

    Macro-F1 weights both classes equally, so the threshold is pushed
    toward balanced recall across LTS / non-LTS even under class imbalance.
    The conservative range (0.20–0.75) prevents extreme thresholds on
    small validation folds where a single patient flip dominates the metric.

    Parameters
    ----------
    probs  : (N,) predicted probabilities for the positive class
    y_true : (N,) ground-truth binary labels

    Returns
    -------
    float — best threshold in [0.20, 0.75]
    """
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.20, 0.76, 0.02):
        preds = (probs >= thresh).astype(int)
        f1    = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh


def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training labels.

    w_c = N / (2 * N_c)

    Computed dynamically so weights are correct for every LTS threshold
    experiment (12 / 18 / 24 months) without hardcoding cohort sizes.

    Parameters
    ----------
    y_train : (N,) binary training labels (0 = non-LTS, 1 = LTS)

    Returns
    -------
    torch.FloatTensor([w_nonlts, w_lts])
    """
    n_total  = len(y_train)
    n_lts    = int(y_train.sum())
    n_nonlts = n_total - n_lts
    w_nonlts = n_total / (2 * n_nonlts) if n_nonlts > 0 else 1.0
    w_lts    = n_total / (2 * n_lts)    if n_lts    > 0 else 1.0
    return torch.tensor([w_nonlts, w_lts], dtype=torch.float), n_lts, n_nonlts


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def significance_stars(p_value: float) -> str:
    """
    Convert a p-value to a significance annotation string.

    Returns
    -------
    '***'  if p < 0.001
    '**'   if p < 0.01
    '*'    if p < 0.05
    'n.s.' otherwise
    """
    if   p_value < 0.001: return "***"
    elif p_value < 0.01:  return "**"
    elif p_value < 0.05:  return "*"
    else:                 return "n.s."


def plot_roc_curves(results: list,
                    output_path: str,
                    title: str = "ROC Curve Comparison",
                    colors: list = None) -> str:
    """
    Plot smooth interpolated ROC curves for multiple models on one figure.

    Each entry in `results` must be a dict with keys:
      'label' (or 'name') : str   — legend label
      'probs'             : array — predicted probabilities (positive class)
      'y_true'            : array — ground-truth binary labels
      'auc'               : float — pre-computed AUC for the legend

    Smoothing is done by deduplicating FPR values and interpolating onto
    a dense 400-point grid, matching the ablation study plot style.

    Parameters
    ----------
    results     : list of dicts (one per model/configuration)
    output_path : full path to save the PNG (directory must exist)
    title       : figure title string
    colors      : optional list of hex color strings. If fewer entries than
                  results, the list is cycled so no curve is ever dropped.

    Returns
    -------
    output_path : str — path where the figure was saved
    """
    default_colors = [
        "#ff0000", "#00aa22", "#2244ff", "#ff9900",
        "#f2b6c6", "#66d9ff", "#aa00aa", "#bfef45",
        "#000000", "#8B4513", "#e6194b", "#4363d8",
    ]
    # Cycle so any number of results is handled without silent truncation
    color_cycle = itertools.cycle(colors if colors else default_colors)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="navy",
            linewidth=1.5, alpha=0.8, label="Random  0.50")

    for res, color in zip(results, color_cycle):
        label = res.get("label") or res.get("name", "?")
        fpr, tpr, _ = roc_curve(res["y_true"], res["probs"])

        # Deduplicate FPR before interpolation to avoid numpy warnings
        fpr_u, idx = np.unique(fpr, return_index=True)
        tpr_u      = tpr[idx]

        # Interpolate onto dense grid for smooth curves
        fpr_smooth = np.linspace(0, 1, 400)
        tpr_smooth = np.interp(fpr_smooth, fpr_u, tpr_u)

        ax.plot(fpr_smooth, tpr_smooth, color=color, linewidth=2.0,
                label=f"{label}  {res['auc']:.2f}")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ROC plot saved → {output_path}")
    return output_path