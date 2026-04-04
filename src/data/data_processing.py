# data_processing.py
# ------------------
# Processes raw cBioPortal GBM files into clean train-ready CSVs.
#
# Can be run standalone:
#   python data_processing.py                  # uses DEFAULT_THRESHOLD from config
#   python data_processing.py --threshold 12   # override threshold
#   python data_processing.py --threshold 18
#   python data_processing.py --threshold 24
#
# Or called programmatically from run_experiments.py:
#   from data_processing import main as run_data_processing
#   run_data_processing(threshold=12, output_dir="data_12m")

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from config import RAW_DATA_DIR, DEFAULT_DATA_DIR, DEFAULT_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def to_pid(x: str) -> str:
    """Truncate full TCGA barcode to 12-char patient ID."""
    return str(x)[:12]


def os_status_to_binary(x):
    """Convert TCGA OS_STATUS string ('LIVING' / 'DECEASED') to 0 / 1."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if "LIVING"   in s:
        return int(0)
    if "DECEASED" in s:
        return int(1)
    return np.nan


def clean_gene_name(col):
    """Strip Entrez suffixes (e.g. 'TP53|7157') and filter invalid names."""
    col = str(col).strip()
    if col == "PATIENT_ID":
        return col
    if "|" in col:
        col = col.split("|")[0].strip()
    if col in {"", "?", "nan", "NA", "None"}:
        return None
    return col


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(threshold: float = DEFAULT_THRESHOLD,
         raw_data_dir: str = RAW_DATA_DIR,
         output_dir: str = DEFAULT_DATA_DIR,
         use_mutations: bool = False):
    """
    Process raw GBM files and save cleaned CSVs to output_dir.

    Parameters
    ----------
    threshold    : LTS survival threshold in months (e.g. 12, 18, 24).
                   Patients who died ON OR BEFORE this threshold → non-LTS (0).
                   Patients alive AFTER this threshold → LTS (1).
                   Patients censored before the threshold → excluded (fate unknown).
    raw_data_dir : path to raw cBioPortal txt files.
    output_dir   : destination folder for processed CSVs.
    """

    print("=" * 62)
    print(f"  Data Processing — LTS threshold = {threshold} months")
    print(f"  Raw data:   {raw_data_dir}/")
    print(f"  Output dir: {output_dir}/")
    print(f"  Mutation mode: {'ON' if use_mutations else 'OFF'}")
    print("=" * 62)

    os.makedirs(output_dir, exist_ok=True)

    # ── Clinical raw ──────────────────────────────────────────────────────────
    clinical_raw = pd.read_csv(
        f"{raw_data_dir}/data_clinical_patient.txt",
        delimiter="\t",
        skiprows=4
    )

    # ── Labels: survival info + LTS target ───────────────────────────────────
    labels_df = clinical_raw[["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]].copy()

    labels_df = labels_df[labels_df["OS_MONTHS"] != "[Not Available]"]
    labels_df["OS_MONTHS"] = pd.to_numeric(labels_df["OS_MONTHS"], errors="coerce")
    labels_df["PATIENT_ID"] = labels_df["PATIENT_ID"].apply(to_pid)
    labels_df["OS_STATUS"]  = labels_df["OS_STATUS"].apply(os_status_to_binary)
    labels_df = labels_df.dropna(subset=["OS_MONTHS", "OS_STATUS"])
    labels_df = labels_df.drop_duplicates(subset="PATIENT_ID", keep="first").reset_index(drop=True)

    # ── Binary LTS endpoint ───────────────────────────────────────────────────
    #   non-LTS : died ON or BEFORE threshold  (OS_STATUS=1  AND OS_MONTHS <= threshold)
    #   LTS     : still alive AFTER threshold  (OS_MONTHS > threshold, any status)
    #   Excluded: censored BEFORE threshold    (OS_STATUS=0 AND OS_MONTHS <= threshold)
    #             → fate genuinely unknown, removing avoids label noise
    non_lts_mask = (labels_df["OS_STATUS"] == 1) & (labels_df["OS_MONTHS"] <= threshold)
    lts_mask     = labels_df["OS_MONTHS"] > threshold
    excluded_mask = ~(non_lts_mask | lts_mask)

    total      = len(labels_df)
    n_lts      = int(lts_mask.sum())
    n_non_lts  = int(non_lts_mask.sum())
    n_excluded = int(excluded_mask.sum())

    print(f"\nBefore exclusion — total patients: {total}")
    print(f"  LTS     (OS_MONTHS > {threshold}):                    {n_lts}")
    print(f"  non-LTS (OS_STATUS==1 AND OS_MONTHS <= {threshold}): {n_non_lts}")
    print(f"  Excluded (censored before {threshold}m, fate unknown): {n_excluded}")
    print(f"  Exclusion rate: {n_excluded / total * 100:.1f}%")

    keep_mask = non_lts_mask | lts_mask
    labels_df = labels_df[keep_mask].copy().reset_index(drop=True)
    labels_df["LTS"] = lts_mask[keep_mask].astype(int).values

    # Sanity checks
    assert (labels_df["LTS"].isin([0, 1])).all(), "LTS must be binary"
    assert not labels_df["LTS"].isnull().any(),    "LTS must not contain NaN"
    assert (labels_df[labels_df["LTS"] == 0]["OS_STATUS"] == 1).all(), \
        "All non-LTS patients must have observed events"
    assert (labels_df[labels_df["LTS"] == 1]["OS_MONTHS"] > threshold).all(), \
        "All LTS patients must have OS_MONTHS > threshold"

    lts_pct = (labels_df["LTS"] == 1).mean()
    print(f"\nAfter exclusion — patients retained: {len(labels_df)}")
    print(f"  LTS=1 (survived > {threshold}m):  {int((labels_df['LTS']==1).sum())}")
    print(f"  LTS=0 (died <= {threshold}m):      {int((labels_df['LTS']==0).sum())}")
    print(f"  Class balance: {lts_pct*100:.1f}% LTS")

    if lts_pct < 0.25 or lts_pct > 0.75:
        print(f"\n  ⚠️  WARNING: Class imbalance detected ({lts_pct*100:.1f}% LTS).")
        print(f"     Consider: class_weight='balanced' or stratified CV.")

    # ── Clinical features ─────────────────────────────────────────────────────
    clinical_df = clinical_raw[[
        "PATIENT_ID", "AGE", "SEX",
        "KARNOFSKY_PERFORMANCE_SCORE",
        "HISTORY_NEOADJUVANT_TRTYN",
    ]].copy()

    clinical_df["PATIENT_ID"] = clinical_df["PATIENT_ID"].apply(to_pid)
    clinical_df["SEX"] = clinical_df["SEX"].map({"Male": 0, "Female": 1})
    clinical_df["HISTORY_NEOADJUVANT_TRTYN"] = clinical_df["HISTORY_NEOADJUVANT_TRTYN"].map({"Yes": 1, "No": 0})
    clinical_df["AGE"] = pd.to_numeric(clinical_df["AGE"], errors="coerce")
    clinical_df["KARNOFSKY_PERFORMANCE_SCORE"] = pd.to_numeric(
        clinical_df["KARNOFSKY_PERFORMANCE_SCORE"].replace("[Not Available]", np.nan),
        errors="coerce"
    )

    # MICE-style iterative imputation for KPS (fit on full clinical set)
    kps_impute_cols = ["AGE", "SEX", "HISTORY_NEOADJUVANT_TRTYN", "KARNOFSKY_PERFORMANCE_SCORE"]
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20,
        random_state=42,
        sample_posterior=True
    )
    imputed_array = imputer.fit_transform(clinical_df[kps_impute_cols])
    imputed_df    = pd.DataFrame(imputed_array, columns=kps_impute_cols, index=clinical_df.index)
    clinical_df["KARNOFSKY_PERFORMANCE_SCORE"] = (
        imputed_df["KARNOFSKY_PERFORMANCE_SCORE"].clip(0, 100).round(-1).astype(int)
    )
    clinical_df = clinical_df.drop_duplicates(subset="PATIENT_ID", keep="first").reset_index(drop=True)

    # ── CNA ───────────────────────────────────────────────────────────────────
    cna_data = pd.read_csv(f"{raw_data_dir}/data_linear_cna.txt", delimiter="\t").transpose()
    cna_data.columns = cna_data.iloc[0].values
    cna_data = cna_data.iloc[2:, :].reset_index()
    cna_data = cna_data.rename(columns={"index": "PATIENT_ID"})
    cna_data["PATIENT_ID"] = cna_data["PATIENT_ID"].apply(to_pid)
    cna_data = cna_data.drop_duplicates(subset="PATIENT_ID", keep="first")

    # ── mRNA ──────────────────────────────────────────────────────────────────
    mrna_data = pd.read_csv(f"{raw_data_dir}/data_mrna_affymetrix_microarray.txt", delimiter="\t").transpose()
    mrna_data.columns = mrna_data.iloc[0].values
    mrna_data = mrna_data.iloc[2:, :].reset_index()
    mrna_data = mrna_data.rename(columns={"index": "PATIENT_ID"})
    mrna_data["PATIENT_ID"] = mrna_data["PATIENT_ID"].apply(to_pid)
    mrna_data = mrna_data.drop_duplicates(subset="PATIENT_ID", keep="first")

    # ── Methylation (HM27 + HM450) ──────────────────────────────
    met27 = pd.read_csv(f"{raw_data_dir}/data_methylation_hm27.txt", delimiter="\t").transpose()
    met27 = met27.drop(["Entrez_Gene_Id"], axis=0)
    met27.columns = met27.loc["Hugo_Symbol"]
    met27 = met27.drop(index="Hugo_Symbol").reset_index().rename(columns={"index": "PATIENT_ID"})
    met27["PATIENT_ID"] = met27["PATIENT_ID"].apply(to_pid)
    met27.columns.name = None

    met450 = pd.read_csv(f"{raw_data_dir}/data_methylation_hm450.txt", delimiter="\t").transpose()
    met450 = met450.drop(["Entrez_Gene_Id"], axis=0)
    met450.columns = met450.loc["Hugo_Symbol"]
    met450 = met450.drop(index="Hugo_Symbol").reset_index().rename(columns={"index": "PATIENT_ID"})
    met450["PATIENT_ID"] = met450["PATIENT_ID"].apply(to_pid)
    met450.columns.name = None

    # Patients present in both arrays → keep only their HM450 data
    common_patients  = set(met27["PATIENT_ID"]) & set(met450["PATIENT_ID"])
    met27            = met27[~met27["PATIENT_ID"].isin(common_patients)].copy()
    methylation_data = pd.concat([met27, met450], ignore_index=True)
    methylation_data = methylation_data.drop_duplicates(subset="PATIENT_ID", keep="first")

    mut_matrix = None
    keep_genes = []

    # ── Mutation (MAF long format → binary patient × gene matrix) ────────────
    # data_mutations.txt is one row per mutation event per patient.
    # We pivot it to a binary matrix: 1 = gene mutated, 0 = not mutated.
    # Only non-silent mutations are kept (same convention as Brennan et al. 2013).
    # Genes mutated in < MIN_MUT_PATIENTS patients are dropped to remove
    # near-zero-variance columns that mRMR cannot score meaningfully.
    NON_SILENT_TYPES = {
        "Missense_Mutation", "Nonsense_Mutation",
        "Frame_Shift_Del",   "Frame_Shift_Ins",
        "Splice_Site",
        "In_Frame_Del",      "In_Frame_Ins",
        "Nonstop_Mutation",  "Translation_Start_Site",
    }
    MIN_MUT_PATIENTS = 5   # gene must be mutated in ≥ 5 patients to be kept

    mut_raw = pd.read_csv(
        f"{raw_data_dir}/data_mutations.txt",
        delimiter="\t",
        low_memory=False,
        usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"],
    )

    # Keep non-silent mutations only
    mut_raw = mut_raw[mut_raw["Variant_Classification"].isin(NON_SILENT_TYPES)].copy()

    # Normalise patient IDs to 12-char format
    mut_raw["PATIENT_ID"] = mut_raw["Tumor_Sample_Barcode"].apply(to_pid)

    # Pivot to binary patient × gene matrix
    mut_matrix = (
        mut_raw
        .drop_duplicates(subset=["PATIENT_ID", "Hugo_Symbol"])
        .assign(mutated=1)
        .pivot(index="PATIENT_ID", columns="Hugo_Symbol", values="mutated")
        .fillna(0)
        .astype(np.float32)
    )
    mut_matrix.columns.name = None
    mut_matrix = mut_matrix.reset_index()

    # Drop genes mutated in fewer than MIN_MUT_PATIENTS patients
    gene_cols  = [c for c in mut_matrix.columns if c != "PATIENT_ID"]
    freq       = mut_matrix[gene_cols].sum()
    keep_genes = freq[freq >= MIN_MUT_PATIENTS].index.tolist()
    mut_matrix = mut_matrix[["PATIENT_ID"] + keep_genes]


    # ── Intersect patients across ALL modalities ──────────────────────────────
    common_ids = (
        set(labels_df["PATIENT_ID"])
        & set(clinical_df["PATIENT_ID"])
        & set(cna_data["PATIENT_ID"])
        & set(mrna_data["PATIENT_ID"])
        & set(methylation_data["PATIENT_ID"])
    )

    if use_mutations and mut_matrix is not None:
        common_ids = common_ids & set(mut_matrix["PATIENT_ID"])

    print(f"\nBefore modality intersection:")
    print(f"  labels:       {labels_df.shape[0]} patients")
    print(f"  clinical:     {clinical_df.shape[0]} patients")
    print(f"  CNA:          {cna_data.shape[0]} patients")
    print(f"  mRNA:         {mrna_data.shape[0]} patients")
    print(f"  Methylation:  {methylation_data.shape[0]} patients")
    if use_mutations and mut_matrix is not None:
        print(f"  Mutation:     {mut_matrix.shape[0]} patients  "
              f"({len(keep_genes)} genes, mut≥{MIN_MUT_PATIENTS} patients)")
    else:
        print("  Mutation:     SKIPPED (use_mutations=False)")

    labels_df        = labels_df[labels_df["PATIENT_ID"].isin(common_ids)].copy()
    clinical_df      = clinical_df[clinical_df["PATIENT_ID"].isin(common_ids)].copy()
    cna_data         = cna_data[cna_data["PATIENT_ID"].isin(common_ids)].copy()
    mrna_data        = mrna_data[mrna_data["PATIENT_ID"].isin(common_ids)].copy()
    methylation_data = methylation_data[methylation_data["PATIENT_ID"].isin(common_ids)].copy()
    if use_mutations and mut_matrix is not None:
        mut_matrix       = mut_matrix[mut_matrix["PATIENT_ID"].isin(common_ids)].copy()

    # Sort all dataframes by PATIENT_ID for alignment
    sort_dfs = [labels_df, clinical_df, cna_data, mrna_data, methylation_data]
    if use_mutations and mut_matrix is not None:
        sort_dfs.append(mut_matrix)
    for df in sort_dfs:
        df.sort_values("PATIENT_ID", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ── Omics quality filters ─────────────────────────────────────────────────
    # Drop columns with > 10% NaN, fill remaining NaN with 0
    for name, df in [("CNA", cna_data), ("mRNA", mrna_data), ("Meth", methylation_data)]:
        orig = df.shape[1]
        df.dropna(thresh=len(df) * 0.9, axis=1, inplace=True)
        df.fillna(0, inplace=True)
        # Drop near-zero-variance columns (> 90% zeros)
        df.drop(columns=df.columns[(df == 0).mean() >= 0.9], inplace=True)
        print(f"  {name}: {orig} → {df.shape[1]} features after quality filter")

    # Mutation: already binary, re-apply frequency filter on the intersected cohort
    # (some genes may fall below threshold after patient reduction)
    if use_mutations and mut_matrix is not None:
        mut_gene_cols = [c for c in mut_matrix.columns if c != "PATIENT_ID"]
        mut_freq_post = mut_matrix[mut_gene_cols].sum()
        keep_mut = mut_freq_post[mut_freq_post >= MIN_MUT_PATIENTS].index.tolist()
        mut_matrix.drop(
            columns=[c for c in mut_gene_cols if c not in keep_mut],
            inplace=True
        )
        print(f"  Mut: {len(mut_gene_cols)} → {len(keep_mut)} genes after re-filtering "
              f"on intersected cohort")

    print(f"\nAfter modality intersection — common patients: {len(common_ids)}")
    print(f"  LTS=1: {int((labels_df['LTS']==1).sum())}  "
          f"LTS=0: {int((labels_df['LTS']==0).sum())}")

    # ── Save ──────────────────────────────────────────────────────────────────
    labels_df.to_csv(       f"{output_dir}/labels.csv",            index=False)
    clinical_df.to_csv(     f"{output_dir}/clinical_data.csv",     index=False)
    cna_data.to_csv(        f"{output_dir}/cna_data.csv",          index=False)
    mrna_data.to_csv(       f"{output_dir}/mrna_data.csv",         index=False)
    methylation_data.to_csv(f"{output_dir}/methylation_data.csv",  index=False)
    mut_matrix.to_csv(      f"{output_dir}/mutation_data.csv",     index=False)

    print(f"\n  Saved 6 CSVs to {output_dir}/")
    print(f"  clinical_data.csv   : {clinical_df.shape}")
    print(f"  cna_data.csv        : {cna_data.shape}")
    print(f"  mrna_data.csv       : {mrna_data.shape}")
    print(f"  methylation_data.csv: {methylation_data.shape}")
    print(f"  mutation_data.csv   : {mut_matrix.shape}")
    print(f"  labels.csv          : {labels_df.shape}")
    print("=" * 62)

    # Return class counts so run_experiments.py can track them
    return {
        "n_patients": len(labels_df),
        "n_lts":      int((labels_df["LTS"] == 1).sum()),
        "n_non_lts":  int((labels_df["LTS"] == 0).sum()),
        "lts_pct":    float(lts_pct),
        "use_mutations": bool(use_mutations),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process raw GBM cBioPortal files into clean CSVs."
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"LTS survival threshold in months (default: {DEFAULT_THRESHOLD})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: data_{threshold}m/)"
    )
    parser.add_argument(
        "--use_mutations",
        action="store_true",
        help="Include mutation modality (default: False)"
    )
    args = parser.parse_args()

    out = args.output_dir or f"data_{int(args.threshold)}m"
    main(threshold=args.threshold, output_dir=out, use_mutations=args.use_mutations)
