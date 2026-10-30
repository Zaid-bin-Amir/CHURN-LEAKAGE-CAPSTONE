# LEAKAGE AUDIT: Telco Customer Churn

NeuroFive Capstone, Machine Learning. Target: `Churn` (binary classification).
Dataset: Telco Customer Churn (7,043 customers, 20 features), extended with the deliberately flawed columns and rows described below. All numbers come from `python src/compare.py` (seed 42, 80/20 split, Random Forest with 300 trees in both pipelines).

## 1. Summary

The leaky pipeline reports a ROC-AUC of **0.985** and an F1 of **0.959**. After removing every leakage source the honest numbers are a ROC-AUC of **0.833** and an F1 of **0.568**. The drop is not a regression: it is the first valid estimate of how the model behaves on customers it has never seen. Five leakage sources were found, one more than the four listed in the brief.

## 2. Bugs found

### Bug 1: Preprocessing fit on the full dataset before splitting

* **Where:** `leaky_pipeline.py`, `preprocessor.fit_transform(X)` is called on all rows, then the data is split.
* **Why it is leakage:** the median used by the imputer, the mean and variance used by the scaler, and the category list of the encoder are all computed using test rows. Information about the test set flows into training.
* **Real-world problem:** the offline score is slightly optimistic, and in production the model receives data scaled with statistics it was never validated against.
* **Measured impact on its own:** negligible here (ROC-AUC 0.8332 to 0.8334), because standard scaling and median imputation carry little information. It is still a pattern that causes serious damage with target encoding, PCA or aggressive imputation.

### Bug 2: Feature built from the target (`churn_risk_score`)

* **Where:** `flawed_data.py`, `add_churn_risk_score`. The score is the average churn label of customers who share contract, internet service, payment method and tenure, computed with every row's own label included.
* **Why it is leakage:** the feature is a function of the target. Small groups mean a customer's own label barely hides behind the average.
* **Real-world problem:** for a new customer the outcome is unknown, so the score cannot be computed honestly. A deployed model would see a different, weaker feature than the one it was trained on.
* **Measured impact on its own:** ROC-AUC 0.833 to **0.889**, recall 0.503 to 0.623.

### Bug 3: Duplicate customer rows in both train and test

* **Where:** `flawed_data.py`, `add_duplicate_rows` appends 845 copies of existing customers (7,043 to 7,888 rows) before the split.
* **Why it is leakage:** after a random split, **17.3%** of the test rows belong to a customer that also sits in the training set. The model is graded on rows it has already memorised.
* **Real-world problem:** the reported score measures memory instead of generalisation and hides overfitting.
* **Measured impact on its own:** ROC-AUC 0.833 to **0.882**, F1 0.568 to 0.647.

### Bug 4: Post-outcome feature (`exit_survey_score`)

* **Where:** `flawed_data.py`, `add_exit_survey_score`. The column is filled only for customers who already churned, and empty (then replaced by 0) for everyone else.
* **Why it is leakage:** a score from an exit survey can only be recorded after the customer leaves. Whether the value exists is the label.
* **Real-world problem:** at prediction time every active customer has an empty value, so the model loses the feature it relies on most and its real performance collapses.
* **Measured impact on its own:** ROC-AUC 0.833 to **0.988**, precision 0.653 to 1.000. This is the single largest source of inflation and the first feature in the importance chart.

### Bug 5: Feature selection fit on all rows and the full target (found beyond the listed examples)

* **Where:** `leaky_pipeline.py`, `SelectKBest(f_classif, k=20)` is fit on the whole matrix and the whole target before the split.
* **Why it is leakage:** the features are chosen by their correlation with test labels, so selection is tuned to the very rows used for evaluation.
* **Real-world problem:** the selected set looks stronger than it really is and the estimate is biased upward, a silent error because nothing in the code looks wrong.
* **Measured impact on its own:** small here (ROC-AUC 0.8332 to 0.8351) because only 20 of roughly 45 candidates are chosen, but the effect grows quickly with wider feature sets.

Other checks that passed: `customerID` is dropped in both pipelines because an identifier carries no generalisable signal and would let a model memorise rows; `TotalCharges` has 11 blank values, which are converted to missing and imputed inside the pipeline rather than before the split.

## 3. The fix (`fixed_pipeline.py`)

1. Remove duplicate customers (`drop_duplicates` on `customerID`).
2. Drop `customerID`, `churn_risk_score` and `exit_survey_score`.
3. Stratified 80/20 split **before** any learned step.
4. A single scikit-learn `Pipeline`: `ColumnTransformer` (median imputation and scaling for numeric columns, one-hot encoding for categorical columns), `SelectKBest(k=20)`, then the Random Forest. Every step is fit on training data only.
5. 5-fold stratified cross-validation on the training portion alone for a second, independent estimate.

## 4. Before and after metrics (held-out test set)

| Metric | Leaky pipeline | Fixed pipeline | Gap |
|--------|---------------:|---------------:|----:|
| Accuracy | 0.9785 | 0.7970 | 0.1814 |
| Precision | 1.0000 | 0.6528 | 0.3472 |
| Recall | 0.9217 | 0.5027 | 0.4190 |
| F1 | 0.9592 | 0.5680 | 0.3913 |
| ROC-AUC | 0.9853 | 0.8332 | 0.1521 |

Supporting numbers: fixed-pipeline cross-validated ROC-AUC is 0.8284 +/- 0.0150, which agrees with the test ROC-AUC of 0.8332, a sign that the estimate is stable. Test rows that also exist in train: 17.3% (leaky) versus 0.0% (fixed).

## 5. Impact of each bug (added one at a time to the clean pipeline)

| Variant | Accuracy | Recall | F1 | ROC-AUC |
|---------|---------:|-------:|---:|--------:|
| Clean baseline | 0.7970 | 0.5027 | 0.5680 | 0.8332 |
| + churn_risk_score | 0.8126 | 0.6230 | 0.6384 | 0.8886 |
| + exit_survey_score | 0.9844 | 0.9412 | 0.9697 | 0.9875 |
| + duplicate rows | 0.8295 | 0.5843 | 0.6465 | 0.8823 |
| + preprocessing on full data | 0.7963 | 0.5027 | 0.5671 | 0.8334 |
| + feature selection on full data | 0.7956 | 0.5080 | 0.5689 | 0.8351 |
| All five bugs together | 0.9835 | 0.9406 | 0.9682 | 0.9933 |

## 6. Interpreting the gap

* The lower fixed score does not mean something went wrong. A valid pipeline is supposed to score below a leaky one, because the leaky one is allowed to see answers the real world will never provide.
* The gap is driven mostly by the post-outcome column, then by the target-derived score and the duplicate rows. The two procedural bugs (preprocessing and feature selection) add only small optimism on this data but are the most common leaks in real projects.
* A suspiciously perfect validation score is a red flag. A ROC-AUC near 0.83 for churn is realistic and useful; one near 0.99 here only proves that the data contained the answer.
* Leakage makes real-world performance unpredictable, not just slightly optimistic: the leaky model depends on inputs that are absent or different in production, so the live error could be far larger than the 15-point AUC gap shown here.

## 7. Charts

All charts are in `outputs/charts/` and are reproduced in `Churn_Leakage_Capstone.ipynb`.
