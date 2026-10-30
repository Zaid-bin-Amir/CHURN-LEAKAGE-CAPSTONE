import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from common import K_FEATURES, SEED, TEST_SIZE, compute_metrics, make_model, make_preprocessor
from flawed_data import ID_COL, TARGET, build_flawed_dataset

VARIANTS = [
    ("Clean baseline", {}),
    ("+ churn_risk_score", {"include_score": True}),
    ("+ exit_survey_score", {"include_survey": True}),
    ("+ duplicate rows", {"keep_duplicates": True}),
    ("+ preprocessing on full data", {"global_prep": True}),
    ("+ feature selection on full data", {"global_select": True}),
    (
        "All five bugs together",
        {
            "include_score": True,
            "include_survey": True,
            "keep_duplicates": True,
            "global_prep": True,
            "global_select": True,
        },
    ),
]


def evaluate_variant(
    path,
    include_score=False,
    include_survey=False,
    keep_duplicates=False,
    global_prep=False,
    global_select=False,
):
    df = build_flawed_dataset(path)
    if not keep_duplicates:
        df = df.drop_duplicates(subset=ID_COL, keep="first").reset_index(drop=True)
    y = df[TARGET].to_numpy()
    drop_cols = [TARGET, ID_COL]
    if not include_score:
        drop_cols.append("churn_risk_score")
    if not include_survey:
        drop_cols.append("exit_survey_score")
    X = df.drop(columns=drop_cols)
    if include_survey:
        X["exit_survey_score"] = X["exit_survey_score"].fillna(0)

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    preprocessor = make_preprocessor(X)
    preprocessor.fit(X if global_prep else X.iloc[train_idx])
    A_train = preprocessor.transform(X.iloc[train_idx])
    A_test = preprocessor.transform(X.iloc[test_idx])

    selector = SelectKBest(f_classif, k=K_FEATURES)
    if global_select:
        selector.fit(preprocessor.transform(X), y)
    else:
        selector.fit(A_train, y[train_idx])

    model = make_model().fit(selector.transform(A_train), y[train_idx])
    proba = model.predict_proba(selector.transform(A_test))[:, 1]
    pred = (proba >= 0.5).astype(int)
    return compute_metrics(y[test_idx], pred, proba)


def run_ablation(path):
    rows = []
    for label, flags in VARIANTS:
        metrics = evaluate_variant(path, **flags)
        rows.append({"variant": label, **metrics})
    return pd.DataFrame(rows)
