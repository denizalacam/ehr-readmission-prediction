from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw")

FILES = [
    "patients.csv",
    "encounters.csv",
    "conditions.csv",
    "medications.csv",
    "observations.csv",
    "procedures.csv",
]


def summarize_file(filename: str) -> None:
    path = DATA_DIR/filename

    if not path.exists():
        print(f"\n❌ Missing file: {filename}")
        return

    df = pd.read_csv(path)

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")

    # -------------------------
    # Basic information
    # -------------------------
    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # -------------------------
    # Column names
    # -------------------------
    print("\nColumns:")
    print(df.columns.tolist())

    # -------------------------
    # Preview
    # -------------------------
    print("\nFirst 5 rows:")
    print(df.head())

    # -------------------------
    # Missing values
    # -------------------------
    print("\nMissing values:")

    missing = df.isnull().sum()

    print(missing[missing > 0].sort_values(ascending=False))

    # -------------------------
    # Data types
    # -------------------------
    print("\nData types:")

    print(df.dtypes)

    # -------------------------
    # Unique patients
    # -------------------------
    if "PATIENT" in df.columns:

        print(f"\nUnique patients: {df['PATIENT'].nunique():,}")

    # -------------------------
    # Memory usage
    # -------------------------
    memory = df.memory_usage(deep=True).sum() / 1024**2

    print(f"\nMemory usage: {memory:.2f} MB")


def main():
    print("DeepEHR Data Exploration")
    print(f"Looking for data in: {DATA_DIR.resolve()}")

    for file in FILES:
        summarize_file(file)


if __name__ == "__main__":
    main()