"""Shared utility functions used across the pipeline."""

import os
import math
import numpy as np
import torch
import matplotlib
import itertools
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve


def concordance_index(risk_scores: np.ndarray,
                      times: np.ndarray,
                      events: np.ndarray) -> float:
    """Harrell's C-index: fraction of comparable pairs where the patient who
    died sooner has the higher risk score. Returns 0.5 if no pairs are comparable."""
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
    """Breslow approximation of the Cox partial likelihood loss: penalises
    an event patient for not outscoring the others still at risk."""
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
    """Log-normal AFT negative log-likelihood: log(T) ~ Normal(mu, sigma^2),
    where mu = pred_log_t (per patient) and sigma = exp(log_sigma) is a single
    global scale learned jointly with the network (see gcn_model.py).

    Uncensored patients get the Normal log-density at log(t). Censored
    patients get -log(S(t_obs)) = -log(Phi((mu - log_t)/sigma)), so the
    penalty vanishes as mu grows past log(t_obs) and blows up as it falls
    below it. Drops the log(T)->T Jacobian term since it's constant w.r.t.
    (mu, sigma) and doesn't affect the gradient or optimum.
    """
    sigma = torch.exp(log_sigma).clamp(min=1e-3)
    log_t = torch.log(times.clamp(min=1e-8))

    resid          = log_t - pred_log_t
    nll_uncensored = log_sigma + 0.5 * math.log(2 * math.pi) + (resid ** 2) / (2 * sigma ** 2)

    z            = (pred_log_t - log_t) / sigma
    phi_z        = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    phi_z        = phi_z.clamp(min=1e-7)          # avoid log(0)
    nll_censored = -torch.log(phi_z)

    per_patient = torch.where(events == 1, nll_uncensored, nll_censored)
    return per_patient.mean()


def normalise_adjacency(psn: np.ndarray,
                        threshold: float = 0.0) -> torch.Tensor:
    """Symmetric normalisation A_norm = D^-0.5 (A + I) D^-0.5, zeroing edges below threshold first."""
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
    """Attach test patients to the training PSN via k-NN cosine similarity,
    connecting each test node to its k most similar training patients.
    The original training PSN is preserved in the top-left block of the result."""
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


# legacy binary-head era, kept for ablation_studies.py / baseline_comparison.py
def find_best_threshold(probs: np.ndarray,
                        y_true: np.ndarray) -> float:
    """Sweep thresholds in [0.20, 0.75] and return the one maximising macro-F1
    (the conservative range avoids extreme thresholds dominated by a single
    patient flip on small validation folds)."""
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.20, 0.76, 0.02):
        preds = (probs >= thresh).astype(int)
        f1    = f1_score(y_true, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, float(thresh)
    return best_thresh


def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights w_c = N / (2 * N_c), computed dynamically
    so they're correct for any LTS threshold without hardcoding cohort sizes."""
    n_total  = len(y_train)
    n_lts    = int(y_train.sum())
    n_nonlts = n_total - n_lts
    w_nonlts = n_total / (2 * n_nonlts) if n_nonlts > 0 else 1.0
    w_lts    = n_total / (2 * n_lts)    if n_lts    > 0 else 1.0
    return torch.tensor([w_nonlts, w_lts], dtype=torch.float), n_lts, n_nonlts


def significance_stars(p_value: float) -> str:
    """p-value -> '***' / '**' / '*' / 'n.s.'"""
    if   p_value < 0.001: return "***"
    elif p_value < 0.01:  return "**"
    elif p_value < 0.05:  return "*"
    else:                 return "n.s."


def plot_roc_curves(results: list,
                    output_path: str,
                    title: str = "ROC Curve Comparison",
                    colors: list = None) -> str:
    """Plot smooth interpolated ROC curves for multiple models on one figure.
    Each entry in `results` needs 'label'/'name', 'probs', 'y_true', 'auc'."""
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