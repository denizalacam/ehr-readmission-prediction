"""
DeepEHR Preprocessing Pipeline

Author: Deniz Alacam

Purpose
-------
Prepare the raw Synthea Electronic Health Record (EHR)
data for machine learning.

This script will eventually:
1. Load raw CSV files
2. Convert dates to datetime
3. Sort encounters chronologically
4. Create 30-day readmission labels
5. Save the processed dataset
"""

from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Project directories
# ------------------------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

def load_data(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from the raw data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    filepath = RAW_DATA_DIR / filename

    print(f"Loading {filename}...")

    df = pd.read_csv(filepath)

    print(f"✓ Loaded {len(df):,} rows")

    return df


def main():
    patients = load_data("patients.csv")
    encounters = load_data("encounters.csv")

    encounters = preprocess_encounters(encounters)

    print("\nPatients shape:", patients.shape)
    print("Encounters shape:", encounters.shape)

    print("\nFirst 5 sorted encounters:")
    print(encounters[["PATIENT", "Id", "START", "STOP", "ENCOUNTERCLASS"]].head())
    print("\nReadmission Summary")
    print(encounters["READMITTED_30D"].value_counts())

    print("\nSample:")
    print(
        encounters[
            [
                "PATIENT",
                "START",
                "STOP",
                "NEXT_ENCOUNTER_START",
                "DAYS_TO_NEXT_ENCOUNTER",
                "READMITTED_30D",
            ]
        ].head(10)
    )
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    encounters.to_csv(
        PROCESSED_DATA_DIR / "encounters_with_labels.csv",
        index=False,
    )

    print("\nSaved: data/processed/encounters_with_labels.csv")


def preprocess_encounters(encounters: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the encounters table.
    """

    encounters = encounters.copy()

    encounters["START"] = pd.to_datetime(encounters["START"])
    encounters["STOP"] = pd.to_datetime(encounters["STOP"])

    encounters = encounters.sort_values(
        by=["PATIENT", "START"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Find the next encounter for each patient
    # ------------------------------------------------------------------

    encounters["NEXT_ENCOUNTER_START"] = (
        encounters
        .groupby("PATIENT")["START"]
        .shift(-1)
    )

    # ------------------------------------------------------------------
    # Days until the next encounter
    # ------------------------------------------------------------------

    encounters["DAYS_TO_NEXT_ENCOUNTER"] = (
        encounters["NEXT_ENCOUNTER_START"] - encounters["START"]
    ).dt.days


    # ------------------------------------------------------------------
    # 30-day readmission label
    # ------------------------------------------------------------------

    encounters["READMITTED_30D"] = (
    encounters["DAYS_TO_NEXT_ENCOUNTER"]
        .le(30)
        .fillna(False)
        .astype(int)
    )

    return encounters


if __name__ == "__main__":
    main()