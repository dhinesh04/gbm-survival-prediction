"""
survival_aware_psn.py
---------------------
Survival-aware Patient Similarity Network (PSN) construction.

Blends the SNF omics-based PSN with a survival similarity kernel so
that the graph used by the GCN directly encodes time-to-event structure.

Motivation
----------
The current SNF PSN is built solely from omics feature similarity
(CNA, mRNA, methylation). OS_MONTHS and OS_STATUS are never used
during graph construction — they only appear later as GCN supervision
targets. This creates a mismatch: the Cox head is trained on survival
signal that the graph topology knows nothing about.

Making the graph construction survival-aware directly addresses this by:
  1. Injecting continuous survival proximity (|T_i - T_j|) into edge weights.
  2. Giving higher confidence to edges involving patients with observed
     events (E_i=1) and down-weighting uncertain censored-censored pairs.
  3. Improving label smoothness z-score so neighbours more consistently
     share survival outcomes before GCN training even starts.

Primary beneficiary: Cox head — survival time/event were previously
                     completely absent from graph construction.
Secondary beneficiary: Binary head — already partially informed via
                     mRMR, but gains from better neighbourhood alignment.

Strategy (lightweight blend)
-----------------------------
    S_ij = exp(-|T_i - T_j| / sigma) * confidence_ij

    confidence_ij:
      - 1.0  if both patients have observed events (E_i=1, E_j=1)
      - 0.5  if exactly one patient has an observed event
      - 0.0  if both are censored (T values are lower bounds — unreliable)

    P_final = (1 - alpha) * P_omics  +  alpha * S_survival
    P_final is then row-normalised to preserve the scale SNF produces.

Parameters
----------
sigma : float
    Bandwidth for the survival time kernel.
    Default = median of pairwise |T_i - T_j| (data-adaptive).
alpha : float
    Blend weight for survival similarity. 0 = pure omics PSN.
    Recommended range: 0.1 - 0.3 so omics signal is not overwhelmed.
"""

import numpy as np
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# SURVIVAL SIMILARITY KERNEL
# ---------------------------------------------------------------------------

def build_survival_kernel(
    os_months: np.ndarray,
    os_status: np.ndarray,
    sigma: float = None
) -> np.ndarray:
    """
    Build an N x N survival similarity matrix.

    S_ij = exp(-|T_i - T_j| / sigma) * confidence_ij

    The confidence mask down-weights censored-censored pairs because
    both T values are right-censored lower bounds — their similarity
    is unreliable. Pairs where at least one event is observed are
    more trustworthy.

    Parameters
    ----------
    os_months : (N,) float   Survival times in months.
    os_status : (N,) int     1 = deceased (event observed), 0 = censored.
    sigma     : float | None Kernel bandwidth. If None, set to median
                             of all pairwise |T_i - T_j|.

    Returns
    -------
    S : (N, N) float   Survival similarity matrix in [0, 1].
                       Diagonal is 1.0 (perfect self-similarity).
    """
    T = os_months.reshape(-1, 1).astype(np.float64)
    E = os_status.astype(np.float64)

    # Pairwise absolute time differences
    diff = np.abs(cdist(T, T, metric="cityblock"))   # (N, N)

    # Data-adaptive bandwidth: median of all off-diagonal differences
    N = len(T)
    if sigma is None:
        off_diag = diff[np.triu_indices(N, k=1)]
        sigma = float(np.median(off_diag)) + 1e-8
        print(f"  Survival kernel: sigma = {sigma:.2f} months (data-adaptive)")
    else:
        print(f"  Survival kernel: sigma = {sigma:.2f} months (user-specified)")

    # Gaussian similarity kernel
    S_raw = np.exp(-diff / sigma)   # (N, N)

    # Confidence mask based on event status:
    #   E_i=1 AND E_j=1  -> 1.0  (both confirmed events)
    #   E_i=1 XOR E_j=1  -> 0.5  (one confirmed, one censored)
    #   E_i=0 AND E_j=0  -> 0.0  (both censored — unreliable pair)
    E_row = E.reshape(-1, 1)
    E_col = E.reshape(1, -1)
    both_events   = E_row * E_col                          # 1 if both=1
    one_event     = (E_row + E_col - 2 * both_events)     # 1 if exactly one=1
    confidence    = 1.0 * both_events + 0.5 * one_event   # 0 if neither

    S = S_raw * confidence

    # Restore diagonal to 1 (a patient is perfectly similar to itself)
    np.fill_diagonal(S, 1.0)

    return S.astype(np.float32)


# ---------------------------------------------------------------------------
# BLEND SNF PSN WITH SURVIVAL KERNEL
# ---------------------------------------------------------------------------

def build_survival_aware_psn(
    psn_omics: np.ndarray,
    os_months: np.ndarray,
    os_status: np.ndarray,
    alpha: float = 0.2,
    sigma: float = None
) -> tuple:
    """
    Blend an existing omics-based SNF PSN with a survival similarity kernel.

    P_final = (1 - alpha) * P_omics  +  alpha * S_survival

    Each row is then normalised to [0, 1] to preserve SNF's scale.

    Parameters
    ----------
    psn_omics : (N, N)   SNF-fused omics PSN from snf.snf().
    os_months : (N,)     Survival times in months (training patients).
    os_status : (N,)     Event indicators (1=deceased, 0=censored).
    alpha     : float    Weight assigned to survival similarity.
                         Recommended: 0.1 - 0.3.
    sigma     : float|None  Kernel bandwidth (None = data-adaptive).

    Returns
    -------
    psn_sa    : (N, N)  Survival-aware PSN (blended + normalised).
    S         : (N, N)  Raw survival similarity matrix (for diagnostics).
    """
    assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
    N = psn_omics.shape[0]
    assert psn_omics.shape == (N, N), "psn_omics must be square"

    print("\n-- Building Survival-Aware PSN --------------------------------")
    print(f"  Blend weight: alpha={alpha}  "
          f"(omics {(1-alpha)*100:.0f}% | survival {alpha*100:.0f}%)")

    S = build_survival_kernel(os_months, os_status, sigma=sigma)

    # Weighted blend
    psn_blended = (1.0 - alpha) * psn_omics + alpha * S

    # Row-normalise to [0, 1] so magnitude matches what SNF produces
    row_max = psn_blended.max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1.0, row_max)   # avoid divide-by-zero
    psn_sa = psn_blended / row_max

    # Restore symmetry (tiny numerical drift from row-wise division)
    psn_sa = (psn_sa + psn_sa.T) / 2.0

    print(f"  Survival kernel (off-diag): "
          f"mean={float(S[np.triu_indices(N, k=1)].mean()):.4f}, "
          f"max={float(S[np.triu_indices(N, k=1)].max()):.4f}")
    print(f"  Blended PSN range: "
          f"[{psn_sa.min():.4f}, {psn_sa.max():.4f}]")
    print(f"  Symmetric:     {np.allclose(psn_sa, psn_sa.T, atol=1e-6)}")
    print(f"  Non-negative:  {bool((psn_sa >= 0).all())}")
    print("--------------------------------------------------------------\n")

    return psn_sa.astype(np.float32), S


# ---------------------------------------------------------------------------
# ABLATION HELPER — compare omics-only vs survival-aware PSN diagnostics
# ---------------------------------------------------------------------------

def compare_psn_diagnostics(
    psn_omics: np.ndarray,
    psn_sa: np.ndarray,
    y_labels: np.ndarray,
    top_k: int = 20,
    n_permutations: int = 1000,
    random_state: int = 42
) -> dict:
    """
    Run hubness and label-smoothness diagnostics on both PSNs and
    print a side-by-side comparison table.

    Parameters
    ----------
    psn_omics  : (N, N)   Original omics-only SNF PSN.
    psn_sa     : (N, N)   Survival-aware blended PSN.
    y_labels   : (N,)     LTS labels (or event indicators) for smoothness.
    top_k      : int      Neighbourhood size for hubness/agreement.
    n_permutations : int  Permutations for null distribution.
    random_state : int    RNG seed.

    Returns
    -------
    dict with keys 'omics' and 'survival_aware', each containing
    'max_indegree', 'neighbor_agreement', 'z_score'.
    """

    def _run(psn):
        N = psn.shape[0]
        psn_nd = psn.copy()
        np.fill_diagonal(psn_nd, 0)

        # Build top-K neighbour mask
        topk = np.zeros((N, N), dtype=bool)
        for i in range(N):
            top_idx = np.argsort(psn_nd[i])[::-1][:top_k]
            topk[i, top_idx] = True

        max_indeg = int(topk.sum(axis=0).max())

        y_arr = np.array(y_labels)
        ag_list = []
        for i in range(N):
            nbrs = np.where(topk[i])[0]
            if len(nbrs):
                ag_list.append((y_arr[nbrs] == y_arr[i]).mean())
        obs_ag = float(np.mean(ag_list))

        rng = np.random.default_rng(random_state)
        perm_ags = []
        for _ in range(n_permutations):
            yp = rng.permutation(y_arr)
            pa = []
            for i in range(N):
                nbrs = np.where(topk[i])[0]
                if len(nbrs):
                    pa.append((yp[nbrs] == yp[i]).mean())
            perm_ags.append(np.mean(pa))

        pm  = float(np.mean(perm_ags))
        ps  = float(np.std(perm_ags))
        z   = (obs_ag - pm) / (ps + 1e-10)

        return {
            "max_indegree":       max_indeg,
            "neighbor_agreement": obs_ag,
            "perm_mean":          pm,
            "perm_std":           ps,
            "z_score":            z,
        }

    d_omics = _run(psn_omics)
    d_sa    = _run(psn_sa)

    # Print side-by-side table
    print("\n-- PSN Diagnostic Comparison ----------------------------------")
    print(f"  {'Metric':<38} {'Omics-only':>12} {'Surv-aware':>12} {'Delta':>10}")
    print("  " + "-" * 76)

    metrics = [
        ("Max in-degree  (hubness, lower better)",  "max_indegree",       "d"),
        ("Neighbour agreement (higher better)",      "neighbor_agreement", ".4f"),
        ("Permutation null mean",                    "perm_mean",          ".4f"),
        ("Permutation null std",                     "perm_std",           ".4f"),
        ("Label smoothness z-score (higher better)", "z_score",            ".3f"),
    ]

    for label, key, fmt in metrics:
        vo = d_omics[key]
        vs = d_sa[key]
        if fmt == "d":
            delta = f"{vs - vo:+d}"
            print(f"  {label:<38} {vo:>12d} {vs:>12d} {delta:>10}")
        else:
            delta = f"{vs - vo:+.4f}"
            print(f"  {label:<38} {vo:>12{fmt}} {vs:>12{fmt}} {delta:>10}")

    dz = d_sa["z_score"] - d_omics["z_score"]
    print("\n  Interpretation:")
    if dz > 0.5:
        print(f"  [PASS] Survival-aware PSN improves label smoothness "
              f"(delta_z = +{dz:.2f})")
    elif dz > 0:
        print(f"  [WEAK] Marginal improvement (delta_z = +{dz:.2f}). "
              f"Try increasing alpha.")
    else:
        print(f"  [FAIL] No improvement (delta_z = {dz:.2f}). "
              f"Consider tuning sigma or alpha.")
    print("--------------------------------------------------------------\n")

    return {"omics": d_omics, "survival_aware": d_sa}