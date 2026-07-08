from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = Path("data/processed/encounter_level_dataset.csv")

NUMERICAL_FEATURES = [
    "AGE_AT_ENCOUNTER",
    "TOTAL_CLAIM_COST",
    "NUM_PRIOR_ENCOUNTERS",
]

CATEGORICAL_FEATURES = [
    "GENDER",
    "RACE",
    "ETHNICITY",
    "ENCOUNTERCLASS",
]

# ---------------------------------------------------
# Functions
# ---------------------------------------------------

def load_dataset() -> pd.DataFrame:

    print("Loading feature dataset...")

    return pd.read_csv(DATASET_PATH)


def create_X_y(df: pd.DataFrame):

    feature_columns = [
        "AGE_AT_ENCOUNTER",
        "GENDER",
        "RACE",
        "ETHNICITY",
        "ENCOUNTERCLASS",
        "TOTAL_CLAIM_COST",
        "NUM_PRIOR_ENCOUNTERS",
    ]

    X = df[feature_columns].copy()

    y = df["READMITTED_30D"].copy()

    return X, y


def split_data(X, y):
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test

def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing pipeline for numerical and categorical features.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor



def create_processed_feature_matrix(X_train, X_test):
    """
    Fit the preprocessing pipeline on the training data
    and transform both the training and testing data.
    """

    # Build the preprocessing pipeline
    preprocessor = build_preprocessor()

    # Learn preprocessing from training data and transform it
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply the same preprocessing to the test data
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    return (
    X_train_processed_df,
    X_test_processed_df,
    preprocessor,
    )
