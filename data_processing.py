# Data processing of raw files
import pandas as pd
import numpy as np
import os

def to_pid(x: str) -> str:
    return str(x)[:12]

def os_status_to_binary(x):
    """
    You want: living=1, deceased=0
    TCGA often has values like '0:LIVING' and '1:DECEASED'
    """
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()

    # If it contains text
    if "LIVING" in s:
        return 1
    if "DECEASED" in s:
        return 0

    # If it’s numeric-like
    # Many TCGA files use 0=LIVING, 1=DECEASED -> invert to match your requirement
    try:
        v = int(float(s.split(":")[0]))
        # v==0 means living -> 1
        # v==1 means deceased -> 0
        return 1 if v == 0 else 0
    except:
        return np.nan

def clean_mrna(df, id_col="PATIENT_ID", max_na_frac=0.10, fill_value=0):
    """
    mRNA: drop features with >10% NA, then fill remaining NA with 0.
    (no aggressive zero-filter by default)
    """
    feats = df.drop(columns=[id_col])
    feats = feats.dropna(thresh=len(feats) * (1 - max_na_frac), axis=1)
    feats = feats.fillna(fill_value)
    return pd.concat([df[[id_col]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

def clean_cna_keep_frequent_alterations(df, id_col="PATIENT_ID",
                                       max_na_frac=0.10,
                                       fill_value=0,
                                       min_nonzero_frac=0.05):
    """
    CNA: drop features with >10% NA, fill NA with 0, then KEEP genes that are
    altered (non-zero) in at least `min_nonzero_frac` of patients.
    Example: min_nonzero_frac=0.05 keeps genes with CNA != 0 in >=5% of patients.
    """
    feats = df.drop(columns=[id_col])

    # 1) drop high-NA columns
    feats = feats.dropna(thresh=len(feats) * (1 - max_na_frac), axis=1)

    # 2) fill remaining NA with 0
    feats = feats.fillna(fill_value)

    # 3) keep columns with enough non-zero values
    nonzero_frac = (feats != 0).mean(axis=0)
    feats = feats.loc[:, nonzero_frac >= min_nonzero_frac]

    return pd.concat([df[[id_col]].reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

def main():
    raw_data_path = "raw_data"
    processed_data_path = "data"
    os.makedirs(processed_data_path, exist_ok=True)

    # -------- Clinical --------
    clinical_data = pd.read_csv(
        f"{raw_data_path}/data_clinical_patient.txt",
        delimiter="\t",
        skiprows=4
    )

    clinical_data = clinical_data[
        [
            "PATIENT_ID",
            "AGE",
            "SEX",
            "RACE",
            "ETHNICITY",
            "OS_STATUS",
            "OS_MONTHS",
        ]
    ].copy()

    # Normalize patient IDs
    clinical_data["PATIENT_ID"] = clinical_data["PATIENT_ID"].apply(to_pid)

    # Convert OS_STATUS to your binary definition
    clinical_data["OS_STATUS"] = clinical_data["OS_STATUS"].apply(os_status_to_binary)

    # Ensure OS_MONTHS is numeric
    clinical_data["OS_MONTHS"] = pd.to_numeric(clinical_data["OS_MONTHS"], errors="coerce")

    # LTS/STS label: >24 -> 1 else 0 (this makes exactly 24 count as STS=0)
    clinical_data["LTS"] = (clinical_data["OS_MONTHS"] >= 24).astype(int)

    # Drop duplicate patients (after truncation)
    clinical_data = clinical_data.drop_duplicates(subset="PATIENT_ID", keep="first")

    # -------- CNA --------
    cna_data = pd.read_csv(
        f"{raw_data_path}/data_linear_cna.txt",
        delimiter="\t"
    ).transpose()

    cna_data.columns = cna_data.iloc[0].values
    cna_data = cna_data.iloc[2:, :].reset_index()

    # rename index column to PATIENT_ID
    cna_data = cna_data.rename(columns={"index": "PATIENT_ID"})

    # Normalize patient IDs then drop duplicates
    cna_data["PATIENT_ID"] = cna_data["PATIENT_ID"].apply(to_pid)
    cna_data = cna_data.drop_duplicates(subset="PATIENT_ID", keep="first")

    # -------- mRNA --------
    mrna_data = pd.read_csv(
        f"{raw_data_path}/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt",
        delimiter="\t"
    ).transpose()

    mrna_data.columns = mrna_data.iloc[0].values
    mrna_data = mrna_data.iloc[2:, :].reset_index()
    
    # rename index column to PATIENT_ID
    mrna_data = mrna_data.rename(columns={"index": "PATIENT_ID"})

    # Normalize patient IDs then drop duplicates
    mrna_data["PATIENT_ID"] = mrna_data["PATIENT_ID"].apply(to_pid)
    mrna_data = mrna_data.drop_duplicates(subset="PATIENT_ID", keep="first")

    # -------- Keep only common patients across all 3 --------
    common_ids = (
        set(clinical_data["PATIENT_ID"])
        & set(cna_data["PATIENT_ID"])
        & set(mrna_data["PATIENT_ID"])
    )

    clinical_data = clinical_data[clinical_data["PATIENT_ID"].isin(common_ids)].copy()
    cna_data = cna_data[cna_data["PATIENT_ID"].isin(common_ids)].copy()
    mrna_data = mrna_data[mrna_data["PATIENT_ID"].isin(common_ids)].copy()

    # Sort so all three are aligned in the same order
    clinical_data = clinical_data.sort_values("PATIENT_ID").reset_index(drop=True)
    cna_data = cna_data.sort_values("PATIENT_ID").reset_index(drop=True)
    mrna_data = mrna_data.sort_values("PATIENT_ID").reset_index(drop=True)

    # CNA: keep genes altered in >=5% of patients (tune min_nonzero_frac if needed)
    cna_data = clean_cna_keep_frequent_alterations(
        cna_data,
        id_col="PATIENT_ID",
        max_na_frac=0.10,
        fill_value=0,
        min_nonzero_frac=0.05
    )

    # mRNA: drop high-NA genes and fill NA with 0 (no non-zero filter by default)
    mrna_data = clean_mrna(
        mrna_data,
        id_col="PATIENT_ID",
        max_na_frac=0.10,
        fill_value=0
    )

    # Quick sanity check
    print("CNA matches mRNA:", np.array_equal(cna_data["PATIENT_ID"].values, mrna_data["PATIENT_ID"].values))
    print("mRNA matches Clinical:", np.array_equal(mrna_data["PATIENT_ID"].values, clinical_data["PATIENT_ID"].values))
    print("Common patients kept:", len(common_ids))

    # Optional: save
    clinical_data.to_csv(f"{processed_data_path}/clinical_data.csv", index=False)
    cna_data.to_csv(f"{processed_data_path}/cna_data.csv", index=False)
    mrna_data.to_csv(f"{processed_data_path}/mrna_data.csv", index=False)

    print('the shape for cna data is: '+str(cna_data.shape))
    print('the shape for mrna data is: '+str(mrna_data.shape))

if __name__ == "__main__":
    main()