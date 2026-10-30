import sys

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from common import (
    K_FEATURES,
    SEED,
    TEST_SIZE,
    clean_feature_names,
    compute_metrics,
    make_model,
    make_preprocessor,
)
from flawed_data import ID_COL, LEAKY_COLUMNS, TARGET, build_flawed_dataset


def run_fixed_pipeline(path):
    df = build_flawed_dataset(path)
    df = df.drop_duplicates(subset=ID_COL, keep="first").reset_index(drop=True)
    ids = df[ID_COL]
    y = df[TARGET].to_numpy()
    X = df.drop(columns=[TARGET, ID_COL] + LEAKY_COLUMNS)

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = Pipeline(
        [
            ("prep", make_preprocessor(X_train)),
            ("select", SelectKBest(f_classif, k=K_FEATURES)),
            ("model", make_model()),
        ]
    )

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=folds, scoring="roc_auc", n_jobs=1
    )

    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    train_pred = pipeline.predict(X_train)

    names = clean_feature_names(pipeline.named_steps["prep"].get_feature_names_out())
    names = names[pipeline.named_steps["select"].get_support()]

    return {
        "name": "Fixed",
        "metrics": compute_metrics(y_test, pred, proba),
        "train_accuracy": float((train_pred == y_train).mean()),
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "y_test": y_test,
        "proba": proba,
        "pred": pred,
        "features": names,
        "importances": pipeline.named_steps["model"].feature_importances_,
        "rows": len(df),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "test_overlap": float(ids.iloc[test_idx].isin(ids.iloc[train_idx]).mean()),
    }


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    result = run_fixed_pipeline(data_path)
    print("FIXED PIPELINE")
    for metric, value in result["metrics"].items():
        print(f"{metric:>10}: {value:.4f}")
    print(f"cv roc_auc (train only): {result['cv_roc_auc_mean']:.4f} +/- {result['cv_roc_auc_std']:.4f}")
    print(f"test rows also present in train: {result['test_overlap']:.1%}")
