import numpy as np
import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"
LEAKY_COLUMNS = ["churn_risk_score", "exit_survey_score"]


def load_raw(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_churn_risk_score(df, seed):
    rng = np.random.default_rng(seed)
    key = (
        df["Contract"]
        + "|"
        + df["InternetService"]
        + "|"
        + df["PaymentMethod"]
        + "|"
        + df["tenure"].astype(str)
    )
    score = df.groupby(key)[TARGET].transform("mean")
    noise = rng.normal(0, 0.03, len(df))
    return (score + noise).clip(0, 1).round(4)


def add_exit_survey_score(df, seed):
    rng = np.random.default_rng(seed + 1)
    scores = rng.integers(1, 6, len(df)).astype(float)
    answered = rng.random(len(df)) < 0.92
    retained = (df[TARGET] == 0).to_numpy()
    scores[retained | ~answered] = np.nan
    return scores


def add_duplicate_rows(df, seed, fraction=0.12):
    duplicates = df.sample(frac=fraction, random_state=seed + 2)
    combined = pd.concat([df, duplicates], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)


def build_flawed_dataset(path, seed=42):
    df = load_raw(path)
    df["churn_risk_score"] = add_churn_risk_score(df, seed)
    df["exit_survey_score"] = add_exit_survey_score(df, seed)
    return add_duplicate_rows(df, seed)
