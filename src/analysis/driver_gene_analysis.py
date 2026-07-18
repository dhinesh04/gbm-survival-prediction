"""Univariate Cox hazard analysis against the 27 GBM driver genes from
Brennan et al. 2013 (Cell), same TCGA-GBM cohort as this study."""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# Brennan et al. 2013 driver gene panel
DRIVER_GENES = [
    "ARID2", "ATRX",  "BRAF",    "CDKN2A", "CIC",     "DNMT3A",
    "EGFR",  "FGFR2", "FUBP1",   "IDH1",   "IDH2",    "KDR",
    "KRAS",  "MDM4",  "MET",     "NF1",    "NOTCH2",  "NTRK1",
    "PDGFRA","PIK3CA","PIK3R1",  "PTEN",   "PTPN11",  "RB1",
    "SETD2", "SMARCB1","TP53",
]

# Genes highlighted as most significant in the paper (bold in professor's list)
BOLD_GENES = {"ATRX", "BRAF", "CDKN2A", "EGFR", "IDH1",
              "IDH2", "KRAS", "PTEN", "TP53"}


def _find_gene_in_columns(gene: str, columns):
    """Match a driver gene to its column name, handling suffixes like 'EGFR|1956' or 'EGFR_mut'."""
    cols = list(columns)
    if gene in cols:
        return gene
    for c in cols:
        if c.startswith(gene + "|") or c.startswith(gene + "_"):
            return c
        if c.upper() == gene.upper():
            return c
    return None


def run_univariate_cox(
    X_cna:       pd.DataFrame,
    X_mrna:      pd.DataFrame,
    X_meth:      pd.DataFrame,
    X_mut: pd.DataFrame,
    os_months:   pd.Series,
    os_status:   pd.Series,
    gene_list:   list = None,
) -> pd.DataFrame:
    """Univariate penalised CoxPH per driver gene across all modalities, fit on
    the full cohort (biological validation, not model evaluation)."""
    from lifelines import CoxPHFitter

    modalities = [
        ("CNA",         X_cna),
        ("mRNA",        X_mrna),
        ("Methylation", X_meth),
        ("Mutation", X_mut)
    ]

    # Align survival data index to DataFrames
    common_idx = (X_cna.index
                  .intersection(X_mrna.index)
                  .intersection(X_meth.index)
                  .intersection(X_mut.index)
                  .intersection(os_months.index))

    t = os_months.loc[common_idx].values.astype(float)
    e = os_status.loc[common_idx].values.astype(float)

    results = []
    for gene in (gene_list if gene_list is not None else DRIVER_GENES):
        for mod_name, X in modalities:
            X_aligned = X.loc[common_idx] if X.index.isin(common_idx).any() else X

            col = _find_gene_in_columns(gene, X_aligned.columns)
            if col is None:
                continue

            feature = X_aligned[col].values.astype(float)

            if np.std(feature) < 1e-8:  # no variation to model
                continue

            df_cox = pd.DataFrame({
                "feature": feature,
                "time":    t,
                "event":   e.astype(int),
            })

            try:
                cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
                cph.fit(df_cox, duration_col="time", event_col="event",
                        show_progress=False)
                s = cph.summary.loc["feature"]
                hr      = float(np.exp(s["coef"]))
                ci_low  = float(np.exp(s["coef lower 95%"]))
                ci_high = float(np.exp(s["coef upper 95%"]))
                pval    = float(s["p"])
                results.append({
                    "gene":        gene,
                    "modality":    mod_name,
                    "col_name":    col,
                    "HR":          hr,
                    "CI_low":      ci_low,
                    "CI_high":     ci_high,
                    "p_value":     pval,
                    "neg_log10_p": -np.log10(pval + 1e-300),
                    "is_bold":     gene in BOLD_GENES,
                    "significant": pval < 0.05,
                })
            except Exception:
                pass  # gene failed to converge, skip it

    df = pd.DataFrame(results)
    if df.empty:
        return df

    return df.sort_values("p_value")


def plot_forest(cox_df: pd.DataFrame, output_dir: str):
    """Forest plot of univariate Cox HRs: significant (p<0.05) genes plus up
    to 5 top non-significant ones for context."""
    if cox_df.empty:
        print("  No Cox results to plot.")
        return

    sig   = cox_df[cox_df["significant"]].copy()
    nonsig = cox_df[~cox_df["significant"]].head(5).copy()
    plot_df = pd.concat([sig, nonsig]).drop_duplicates(
        subset=["gene", "modality"]).copy()
    plot_df = plot_df.sort_values("HR", ascending=True).reset_index(drop=True)

    if plot_df.empty:
        print("  No results to display in forest plot.")
        return

    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.5 + 2)))

    y = np.arange(n)

    for i, row in plot_df.iterrows():
        color  = "#d62728" if row["HR"] > 1 else "#1f77b4"
        alpha  = 1.0 if row["significant"] else 0.4
        marker = "D" if row["is_bold"] else "o"
        ms     = 9  if row["is_bold"] else 7

        # CI bar
        ax.plot([row["CI_low"], row["CI_high"]], [i, i],
                color=color, alpha=alpha, linewidth=1.5, zorder=2)
        # Point estimate
        ax.scatter(row["HR"], i, color=color, alpha=alpha,
                   marker=marker, s=ms**2, zorder=3)

    # Null line (HR = 1)
    ax.axvline(1.0, color="black", linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=1)

    # Y-axis labels: gene (modality) with p-value annotation
    ylabels = []
    for _, row in plot_df.iterrows():
        g = row["gene"]
        pstr = f"p={row['p_value']:.3f}" if row["p_value"] >= 0.001 \
               else f"p={row['p_value']:.2e}"
        sig_star = "*" if row["significant"] else ""
        label = f"$\\bf{{{g}}}$" if row["is_bold"] else g
        ylabels.append(f"{label}  [{row['modality']}]  {pstr}{sig_star}")

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=9.5)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=12)
    ax.set_title(
        "Univariate Cox Analysis — GBM Driver Genes\n"
        "HR > 1: risk gene (shorter survival)   |   HR < 1: protective\n"
        "Bold diamond = most recurrently altered (Brennan et al. 2013)   "
        "  * = p < 0.05",
        fontsize=11
    )

    # Legend
    red_patch  = mpatches.Patch(color="#d62728", label="HR > 1  (risk)")
    blue_patch = mpatches.Patch(color="#1f77b4", label="HR < 1  (protective)")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right", fontsize=10)

    ax.set_xscale("log")
    ax.grid(axis='x', alpha=0.25)

    path = f"{output_dir}/driver_gene_forest.png"
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Forest plot saved → {path}")
    return path


def _print_cox_summary(cox_df: pd.DataFrame):
    if cox_df.empty:
        print("  No Cox results.")
        return

    print("\n" + "=" * 75)
    print("  DRIVER GENE ANALYSIS — UNIVARIATE COX RESULTS (all patients)")
    print("=" * 75)
    print(f"  {'Gene':<10} {'Modality':<13} {'HR':>7}  {'95% CI':<18}  "
          f"{'p-value':>9}  {'Sig':^4}")
    print("  " + "-" * 65)

    for _, row in cox_df.sort_values("p_value").iterrows():
        ci_str = f"[{row['CI_low']:.3f}, {row['CI_high']:.3f}]"
        pstr   = f"{row['p_value']:.4f}" if row["p_value"] >= 0.0001 \
                 else f"{row['p_value']:.2e}"
        sig    = "***" if row["p_value"] < 0.001 else \
                 "**"  if row["p_value"] < 0.01  else \
                 "*"   if row["p_value"] < 0.05  else ""
        mark   = "**" if row["is_bold"] else "  "
        print(f"  {mark}{row['gene']:<8} {row['modality']:<13} "
              f"{row['HR']:>7.3f}  {ci_str:<18}  {pstr:>9}  {sig:^4}")

    n_sig = cox_df["significant"].sum()
    print(f"\n  Significant (p<0.05): {n_sig}/{len(cox_df)} gene-modality pairs")
    print(f"  ** = most recurrently altered (Brennan et al. 2013)")
    print("=" * 75)


def run_driver_gene_analysis(
    raw_data:         dict,
    output_dir:       str
) -> dict:
    """Run the univariate Cox driver-gene analysis and save the forest plot.
    raw_data needs X_cna/X_mrna/X_meth/X_mut (full patient matrices), os_months, os_status."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 68)
    print("  DRIVER GENE ANALYSIS")
    print(f"  Reference: Brennan et al. 2013 — {len(DRIVER_GENES)} GBM driver genes")
    print("=" * 68)

    print("\n  Running univariate Cox for each driver gene "
          "across all modalities...")
    cox_df = run_univariate_cox(
        X_cna      = raw_data["X_cna"],
        X_mrna     = raw_data["X_mrna"],
        X_meth     = raw_data["X_meth"],
        X_mut      = raw_data["X_mut"],
        os_months  = raw_data["os_months"],
        os_status  = raw_data["os_status"],
    )
    _print_cox_summary(cox_df)
    plot_forest(cox_df, output_dir)

    return {"cox_df": cox_df}