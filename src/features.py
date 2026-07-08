"""
DeepEHR Feature Engineering

Creates baseline machine learning features from Synthea EHR data.

Input
-----
data/processed/encounters_with_labels.csv
data/raw/patients.csv

Output
------
data/processed/encounter_level_dataset.csv
"""

from pathlib import Path
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_data(path: Path) -> pd.DataFrame:
    """Load a CSV file."""
    print(f"Loading {path}...")
    return pd.read_csv(path)


def create_age_feature(
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge patient demographics with encounters and calculate age at encounter.
    """

    patients = patients.copy()
    encounters = encounters.copy()

    patients["BIRTHDATE"] = pd.to_datetime(
        patients["BIRTHDATE"],
        utc=True
    )

    encounters["START"] = pd.to_datetime(
        encounters["START"],
        utc=True
    )

    encounters = encounters.rename(columns={"Id": "ENCOUNTER_ID"})
    patients = patients.rename(columns={"Id": "PATIENT_ID"})

    
    merged_df = encounters.merge(
        patients[["PATIENT_ID", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"]],
        left_on="PATIENT",
        right_on="PATIENT_ID",
        how="left",
    )

    merged_df["AGE_AT_ENCOUNTER"] = (
        (merged_df["START"] - merged_df["BIRTHDATE"]).dt.days // 365
    )

    return merged_df


def create_prior_visit_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many encounters each patient had before the current encounter.
    """

    df = df.copy()

    df = df.sort_values(
        by=["PATIENT", "START"]
    ).reset_index(drop=True)

    df["NUM_PRIOR_ENCOUNTERS"] = (
        df.groupby("PATIENT").cumcount()
    )

    return df


def clean_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean the baseline features for the first ML dataset.
    """

    feature_cols = [
        "ENCOUNTER_ID",
        "PATIENT",
        "START",
        "STOP",
        "AGE_AT_ENCOUNTER",
        "GENDER",
        "RACE",
        "ETHNICITY",
        "ENCOUNTERCLASS",
        "REASONDESCRIPTION",
        "TOTAL_CLAIM_COST",
        "NUM_PRIOR_ENCOUNTERS",
        "DAYS_TO_NEXT_ENCOUNTER",
        "READMITTED_30D",
    ]

    dataset = df[feature_cols].copy()

    dataset = dataset.rename(
        columns={
            "REASONDESCRIPTION": "REASON_FOR_VISIT",
        }
    )

    categorical_cols = [
        "GENDER",
        "RACE",
        "ETHNICITY",
        "ENCOUNTERCLASS",
        "REASON_FOR_VISIT",
    ]

    for col in categorical_cols:
        dataset[col] = dataset[col].fillna("Unknown")

    dataset["TOTAL_CLAIM_COST"] = dataset["TOTAL_CLAIM_COST"].fillna(0)

    return dataset


def main():
    patients = load_data(RAW_DATA_DIR / "patients.csv")
    encounters = load_data(PROCESSED_DATA_DIR / "encounters_with_labels.csv")

    dataset = create_age_feature(
        patients=patients,
        encounters=encounters,
    )

    dataset = create_prior_visit_feature(dataset)

    dataset = clean_baseline_features(dataset)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "encounter_level_dataset.csv"

    dataset.to_csv(output_path, index=False)

    print("\nFeature dataset created successfully.")
    print(f"Saved to: {output_path}")
    print(f"Shape: {dataset.shape}")

    print("\nFirst 5 rows:")
    print(dataset.head())

    print("\nReadmission label distribution:")
    print(dataset["READMITTED_30D"].value_counts())

    print("\nMissing values:")
    print(dataset.isnull().sum())

    print("\nColumns:")
    print(dataset.columns.tolist())

    print("\nNumeric summary:")
    print(dataset.describe())


if __name__ == "__main__":
    main()