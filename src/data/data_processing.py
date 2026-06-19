# data_processing.py
# ------------------
# Processes raw cBioPortal GBM files into clean train-ready CSVs.
#
# Output labels.csv contains three columns only: PATIENT_ID, OS_MONTHS, OS_STATUS.
# No LTS binary label is computed here. The threshold-based LTS endpoint
# (used for stratification and AUC reporting) is derived inline in the
# training pipeline from OS_MONTHS and OS_STATUS.
#
# This means:
#   - All patients with valid survival data are retained (no exclusions).
#   - The same processed data works for any LTS threshold (12 / 18 / 24 m).
#   - Data processing runs once; threshold sensitivity is handled downstream.
#
# Can be run standalone:
#   python data_processing.py
#
# Or called programmatically:
#   from data_processing import main as run_data_processing
#   run_data_processing(output_dir="data/")

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from config import RAW_DATA_DIR, DEFAULT_DATA_DIR


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
    if "LIVING"   in s: return int(0)
    if "DECEASED" in s: return int(1)
    return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main(raw_data_dir: str = RAW_DATA_DIR,
         output_dir:   str = DEFAULT_DATA_DIR,
         use_mutations: bool = False):
    """
    Process raw GBM files and save cleaned CSVs to output_dir.

    labels.csv columns: PATIENT_ID, OS_MONTHS, OS_STATUS
      OS_STATUS: 1 = deceased (event observed), 0 = censored (alive at last follow-up)

    All patients with valid OS_MONTHS and OS_STATUS are retained.
    No threshold-based exclusion is applied.
    """
    print("=" * 62)
    print(f"  Data Processing")
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

    # ── Survival labels ───────────────────────────────────────────────────────
    labels_df = clinical_raw[["PATIENT_ID", "OS_MONTHS", "OS_STATUS"]].copy()
    labels_df = labels_df[labels_df["OS_MONTHS"] != "[Not Available]"]
    labels_df["OS_MONTHS"] = pd.to_numeric(labels_df["OS_MONTHS"], errors="coerce")
    labels_df["PATIENT_ID"] = labels_df["PATIENT_ID"].apply(to_pid)
    labels_df["OS_STATUS"]  = labels_df["OS_STATUS"].apply(os_status_to_binary)
    labels_df = labels_df.dropna(subset=["OS_MONTHS", "OS_STATUS"])
    labels_df = labels_df.drop_duplicates(subset="PATIENT_ID", keep="first").reset_index(drop=True)

    n_total    = len(labels_df)
    n_deceased = int((labels_df["OS_STATUS"] == 1).sum())
    n_censored = int((labels_df["OS_STATUS"] == 0).sum())

    print(f"\nPatients with valid survival data: {n_total}")
    print(f"  Deceased (OS_STATUS=1): {n_deceased}")
    print(f"  Censored (OS_STATUS=0): {n_censored}")
    print(f"  OS_MONTHS range: [{labels_df['OS_MONTHS'].min():.1f}, "
          f"{labels_df['OS_MONTHS'].max():.1f}] months")
    print(f"  Retaining all {n_total} patients — no exclusions")

    # ── Clinical features ─────────────────────────────────────────────────────
    clinical_df = clinical_raw[[
        "PATIENT_ID", "AGE", "SEX",
        "KARNOFSKY_PERFORMANCE_SCORE",
        "HISTORY_NEOADJUVANT_TRTYN",
    ]].copy()

    clinical_df["PATIENT_ID"] = clinical_df["PATIENT_ID"].apply(to_pid)
    clinical_df["SEX"] = clinical_df["SEX"].map({"Male": 0, "Female": 1})
    clinical_df["HISTORY_NEOADJUVANT_TRTYN"] = (
        clinical_df["HISTORY_NEOADJUVANT_TRTYN"].map({"Yes": 1, "No": 0})
    )
    clinical_df["AGE"] = pd.to_numeric(clinical_df["AGE"], errors="coerce")
    clinical_df["KARNOFSKY_PERFORMANCE_SCORE"] = pd.to_numeric(
        clinical_df["KARNOFSKY_PERFORMANCE_SCORE"].replace("[Not Available]", np.nan),
        errors="coerce"
    )

    # MICE-style iterative imputation for KPS (fit on full clinical set)
    kps_impute_cols = ["AGE", "SEX", "HISTORY_NEOADJUVANT_TRTYN",
                       "KARNOFSKY_PERFORMANCE_SCORE"]
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20,
        random_state=42,
        sample_posterior=True
    )
    imputed_array = imputer.fit_transform(clinical_df[kps_impute_cols])
    imputed_df    = pd.DataFrame(imputed_array, columns=kps_impute_cols,
                                 index=clinical_df.index)
    clinical_df["KARNOFSKY_PERFORMANCE_SCORE"] = (
        imputed_df["KARNOFSKY_PERFORMANCE_SCORE"].clip(0, 100).round(-1).astype(int)
    )
    clinical_df = (clinical_df
                   .drop_duplicates(subset="PATIENT_ID", keep="first")
                   .reset_index(drop=True))

    # ── CNA ───────────────────────────────────────────────────────────────────
    cna_data = pd.read_csv(
        f"{raw_data_dir}/data_linear_cna.txt", delimiter="\t").transpose()
    cna_data.columns = cna_data.iloc[0].values
    cna_data = cna_data.iloc[2:, :].reset_index().rename(columns={"index": "PATIENT_ID"})
    cna_data["PATIENT_ID"] = cna_data["PATIENT_ID"].apply(to_pid)
    cna_data = cna_data.drop_duplicates(subset="PATIENT_ID", keep="first")

    # ── mRNA ──────────────────────────────────────────────────────────────────
    mrna_data = pd.read_csv(
        f"{raw_data_dir}/data_mrna_affymetrix_microarray.txt", delimiter="\t").transpose()
    mrna_data.columns = mrna_data.iloc[0].values
    mrna_data = mrna_data.iloc[2:, :].reset_index().rename(columns={"index": "PATIENT_ID"})
    mrna_data["PATIENT_ID"] = mrna_data["PATIENT_ID"].apply(to_pid)
    mrna_data = mrna_data.drop_duplicates(subset="PATIENT_ID", keep="first")

    # ── Methylation (HM27 + HM450) ────────────────────────────────────────────
    met27 = pd.read_csv(
        f"{raw_data_dir}/data_methylation_hm27.txt", delimiter="\t").transpose()
    met27 = met27.drop(["Entrez_Gene_Id"], axis=0)
    met27.columns = met27.loc["Hugo_Symbol"]
    met27 = (met27.drop(index="Hugo_Symbol")
             .reset_index().rename(columns={"index": "PATIENT_ID"}))
    met27["PATIENT_ID"] = met27["PATIENT_ID"].apply(to_pid)
    met27.columns.name  = None

    met450 = pd.read_csv(
        f"{raw_data_dir}/data_methylation_hm450.txt", delimiter="\t").transpose()
    met450 = met450.drop(["Entrez_Gene_Id"], axis=0)
    met450.columns = met450.loc["Hugo_Symbol"]
    met450 = (met450.drop(index="Hugo_Symbol")
              .reset_index().rename(columns={"index": "PATIENT_ID"}))
    met450["PATIENT_ID"] = met450["PATIENT_ID"].apply(to_pid)
    met450.columns.name  = None

    # Patients in both arrays → keep HM450 data only
    common_patients  = set(met27["PATIENT_ID"]) & set(met450["PATIENT_ID"])
    met27            = met27[~met27["PATIENT_ID"].isin(common_patients)].copy()
    methylation_data = (pd.concat([met27, met450], ignore_index=True)
                        .drop_duplicates(subset="PATIENT_ID", keep="first"))

    # ── Mutation (MAF → binary patient × gene matrix) ─────────────────────────
    NON_SILENT_TYPES = {
        "Missense_Mutation", "Nonsense_Mutation",
        "Frame_Shift_Del",   "Frame_Shift_Ins",
        "Splice_Site",
        "In_Frame_Del",      "In_Frame_Ins",
        "Nonstop_Mutation",  "Translation_Start_Site",
    }
    MIN_MUT_PATIENTS = 5

    mut_raw = pd.read_csv(
        f"{raw_data_dir}/data_mutations.txt",
        delimiter="\t", low_memory=False,
        usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification"],
    )
    mut_raw = mut_raw[mut_raw["Variant_Classification"].isin(NON_SILENT_TYPES)].copy()
    mut_raw["PATIENT_ID"] = mut_raw["Tumor_Sample_Barcode"].apply(to_pid)
    mut_matrix = (
        mut_raw
        .drop_duplicates(subset=["PATIENT_ID", "Hugo_Symbol"])
        .assign(mutated=1)
        .pivot(index="PATIENT_ID", columns="Hugo_Symbol", values="mutated")
        .fillna(0).astype(np.float32)
    )
    mut_matrix.columns.name = None
    mut_matrix = mut_matrix.reset_index()
    gene_cols  = [c for c in mut_matrix.columns if c != "PATIENT_ID"]
    keep_genes = gene_cols[mut_matrix[gene_cols].sum() >= MIN_MUT_PATIENTS].tolist() \
        if False else \
        [c for c in gene_cols if mut_matrix[c].sum() >= MIN_MUT_PATIENTS]
    mut_matrix = mut_matrix[["PATIENT_ID"] + keep_genes]

    # ── Intersect patients across modalities ──────────────────────────────────
    common_ids = (
        set(labels_df["PATIENT_ID"])
        & set(clinical_df["PATIENT_ID"])
        & set(cna_data["PATIENT_ID"])
        & set(mrna_data["PATIENT_ID"])
        & set(methylation_data["PATIENT_ID"])
    )
    if use_mutations:
        common_ids = common_ids & set(mut_matrix["PATIENT_ID"])

    print(f"\nBefore modality intersection:")
    print(f"  labels:       {len(labels_df)} patients")
    print(f"  clinical:     {len(clinical_df)} patients")
    print(f"  CNA:          {len(cna_data)} patients")
    print(f"  mRNA:         {len(mrna_data)} patients")
    print(f"  Methylation:  {len(methylation_data)} patients")
    if use_mutations:
        print(f"  Mutation:     {len(mut_matrix)} patients  "
              f"({len(keep_genes)} genes, mut≥{MIN_MUT_PATIENTS})")
    else:
        print("  Mutation:     SKIPPED (use_mutations=False)")

    labels_df        = labels_df[labels_df["PATIENT_ID"].isin(common_ids)].copy()
    clinical_df      = clinical_df[clinical_df["PATIENT_ID"].isin(common_ids)].copy()
    cna_data         = cna_data[cna_data["PATIENT_ID"].isin(common_ids)].copy()
    mrna_data        = mrna_data[mrna_data["PATIENT_ID"].isin(common_ids)].copy()
    methylation_data = methylation_data[methylation_data["PATIENT_ID"].isin(common_ids)].copy()
    if use_mutations:
        mut_matrix   = mut_matrix[mut_matrix["PATIENT_ID"].isin(common_ids)].copy()

    # Align all dataframes by PATIENT_ID
    all_dfs = [labels_df, clinical_df, cna_data, mrna_data, methylation_data]
    if use_mutations:
        all_dfs.append(mut_matrix)
    for df in all_dfs:
        df.sort_values("PATIENT_ID", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ── Omics quality filters ─────────────────────────────────────────────────
    for name, df in [("CNA", cna_data), ("mRNA", mrna_data), ("Meth", methylation_data)]:
        orig = df.shape[1]
        df.dropna(thresh=len(df) * 0.9, axis=1, inplace=True)
        df.fillna(0, inplace=True)
        df.drop(columns=df.columns[(df == 0).mean() >= 0.9], inplace=True)
        print(f"  {name}: {orig} → {df.shape[1]} features after quality filter")

    if use_mutations:
        mut_gene_cols = [c for c in mut_matrix.columns if c != "PATIENT_ID"]
        keep_mut = [c for c in mut_gene_cols
                    if mut_matrix[c].sum() >= MIN_MUT_PATIENTS]
        mut_matrix.drop(
            columns=[c for c in mut_gene_cols if c not in keep_mut], inplace=True)
        print(f"  Mut: {len(mut_gene_cols)} → {len(keep_mut)} genes "
              f"after re-filtering on intersected cohort")

    n_final    = len(labels_df)
    n_dec_fin  = int((labels_df["OS_STATUS"] == 1).sum())
    n_cens_fin = int((labels_df["OS_STATUS"] == 0).sum())

    print(f"\nAfter modality intersection — common patients: {n_final}")
    print(f"  Deceased: {n_dec_fin}   Censored: {n_cens_fin}")

    # ── Save ──────────────────────────────────────────────────────────────────
    labels_df.to_csv(        f"{output_dir}/labels.csv",            index=False)
    clinical_df.to_csv(      f"{output_dir}/clinical_data.csv",     index=False)
    cna_data.to_csv(         f"{output_dir}/cna_data.csv",          index=False)
    mrna_data.to_csv(        f"{output_dir}/mrna_data.csv",         index=False)
    methylation_data.to_csv( f"{output_dir}/methylation_data.csv",  index=False)
    mut_matrix.to_csv(       f"{output_dir}/mutation_data.csv",     index=False)

    print(f"\n  Saved 6 CSVs to {output_dir}/")
    print(f"  labels.csv          : {labels_df.shape}  "
          f"[PATIENT_ID, OS_MONTHS, OS_STATUS]")
    print(f"  clinical_data.csv   : {clinical_df.shape}")
    print(f"  cna_data.csv        : {cna_data.shape}")
    print(f"  mrna_data.csv       : {mrna_data.shape}")
    print(f"  methylation_data.csv: {methylation_data.shape}")
    print(f"  mutation_data.csv   : {mut_matrix.shape}")
    print("=" * 62)

    return {
        "n_patients": n_final,
        "n_deceased": n_dec_fin,
        "n_censored": n_cens_fin,
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
        "--output_dir", type=str, default=DEFAULT_DATA_DIR,
        help=f"Output directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--use_mutations", action="store_true",
        help="Include mutation modality (default: False)"
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, use_mutations=args.use_mutations)