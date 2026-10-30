import sys

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from common import (
    K_FEATURES,
    SEED,
    TEST_SIZE,
    clean_feature_names,
    compute_metrics,
    make_model,
    make_preprocessor,
)
from flawed_data import ID_COL, TARGET, build_flawed_dataset


def run_leaky_pipeline(path):
    df = build_flawed_dataset(path)
    y = df[TARGET].to_numpy()
    ids = df[ID_COL]
    X = df.drop(columns=[TARGET, ID_COL])
    X["exit_survey_score"] = X["exit_survey_score"].fillna(0)

    preprocessor = make_preprocessor(X)
    X_all = preprocessor.fit_transform(X)
    feature_names = clean_feature_names(preprocessor.get_feature_names_out())

    selector = SelectKBest(f_classif, k=K_FEATURES)
    X_all = selector.fit_transform(X_all, y)
    feature_names = feature_names[selector.get_support()]

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)
    model = make_model().fit(X_all[train_idx], y[train_idx])

    proba = model.predict_proba(X_all[test_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    train_pred = model.predict(X_all[train_idx])

    return {
        "name": "Leaky",
        "metrics": compute_metrics(y[test_idx], pred, proba),
        "train_accuracy": float((train_pred == y[train_idx]).mean()),
        "y_test": y[test_idx],
        "proba": proba,
        "pred": pred,
        "features": feature_names,
        "importances": model.feature_importances_,
        "rows": len(df),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "test_overlap": float(ids.iloc[test_idx].isin(ids.iloc[train_idx]).mean()),
    }


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    result = run_leaky_pipeline(data_path)
    print("LEAKY PIPELINE")
    for metric, value in result["metrics"].items():
        print(f"{metric:>10}: {value:.4f}")
    print(f"test rows also present in train: {result['test_overlap']:.1%}")
