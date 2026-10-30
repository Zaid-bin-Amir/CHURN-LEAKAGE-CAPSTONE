# Data Leakage Audit: Telco Customer Churn

NeuroFive Capstone (Machine Learning). A deliberately flawed churn pipeline is audited, fixed and compared with charts.

## Contents

| Path | Purpose |
|------|---------|🔍 Data Leakage Audit – Telco Customer Churn Prediction

Machine Learning | Data Science | Model Evaluation

A comprehensive Machine Learning project focused on identifying, analyzing, and eliminating Data Leakage in a Telco Customer Churn Prediction system.

This project demonstrates how data leakage can artificially inflate model performance and why proper data preprocessing, feature selection, and evaluation techniques are essential for building reliable Machine Learning models.

🚀 Project Overview

The project compares two Machine Learning pipelines:

- Leaky Pipeline: A deliberately flawed model containing multiple data leakage issues.
- Fixed Pipeline: A corrected model designed to ensure fair evaluation and reliable predictions.

By investigating both pipelines, the project highlights how seemingly minor mistakes can lead to misleading performance results.

🎯 Key Features

- Telco Customer Churn Prediction using Machine Learning.
- Identification of five major data leakage issues.
- Comparison between flawed and corrected ML pipelines.
- Data preprocessing using Scikit-learn.
- Feature selection and Random Forest classification.
- Stratified train-test splitting.
- 5-Fold Cross-Validation.
- Confusion Matrix and ROC Curve visualization.
- Feature importance analysis.
- Leakage impact analysis through ablation experiments.

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook
- Google Colab

📊 Model Performance Comparison

The project evaluates both pipelines using standard classification metrics.

Metric| Leaky Pipeline| Fixed Pipeline
Accuracy| 97.85%| 79.70%
Precision| 100%| 65.28%
Recall| 92.17%| 50.27%
F1-Score| 95.92%| 56.80%
ROC-AUC| 98.53%| 83.32%

Key Finding: The leaky model achieves artificially high performance, while the corrected pipeline provides a more realistic evaluation of how the model generalizes to unseen customer data.

🔬 Data Leakage Issues Identified

The project investigates five major sources of data leakage:

1. Preprocessing performed before train-test splitting.
2. Target-derived feature ("churn_risk_score").
3. Duplicate customer records across training and testing sets.
4. Post-outcome feature ("exit_survey_score").
5. Feature selection performed using the complete dataset.

Each issue is analyzed individually, and its impact on model performance is measured.

📁 Project Structure

churn_leakage_project/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── outputs/
│   ├── charts/
│   ├── metrics_comparison.csv
│   ├── summary.json
│   └── ablation.csv
│
├── src/
│   ├── common.py
│   ├── flawed_data.py
│   ├── leaky_pipeline.py
│   ├── fixed_pipeline.py
│   ├── compare.py
│   ├── ablation.py
│   └── plots.py
│
├── Churn_Leakage_Capstone.ipynb
├── LEAKAGE_AUDIT.md
├── requirements.txt
└── README.md

⚙️ Installation & Usage

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the project directory

cd churn_leakage_project

3. Install dependencies

pip install -r requirements.txt

4. Run the comparison

python src/compare.py

The project will generate performance comparison results, evaluation metrics, and visualizations inside the "outputs/" directory.

Google Colab

You can also run the complete project using Google Colab:

1. Open "Churn_Leakage_Capstone.ipynb".
2. Upload it to Google Colab.
3. Upload the dataset from the "data/" folder when requested.
4. Execute all notebook cells.

📈 Project Outcomes

This project demonstrates the importance of:

- Preventing data leakage in Machine Learning workflows.
- Building reliable preprocessing pipelines.
- Evaluating models on genuinely unseen data.
- Understanding the difference between artificial accuracy and real-world performance.
- Using cross-validation and ablation studies for deeper model analysis.

👨‍💻 Author

Zaid Bin Aamir

Computer Science | Machine Learning | Artificial Intelligence | Python

🔗 GitHub: https://github.com/Zaid-bin-Amir

🔗 LinkedIn: https://www.linkedin.com/in/zaid-bin-aamir-9342483a2

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback.

Built with Python and Machine Learning.
| `Churn_Leakage_Capstone.ipynb` | Full project with charts, ready for Google Colab |
| `src/leaky_pipeline.py` | The flawed pipeline (as given) |
| `src/fixed_pipeline.py` | The corrected pipeline |
| `LEAKAGE_AUDIT.md` | Each bug, why it is leakage, before/after metrics |
| `src/flawed_data.py` | Builds the flawed dataset (injected leakage) |
| `src/ablation.py` | Impact of every bug, one at a time |
| `src/plots.py`, `src/common.py` | Chart helpers, shared model and metrics |
| `src/compare.py` | Runs everything and writes `outputs/` |
| `data/` | Telco Customer Churn CSV |
| `outputs/` | Charts (PNG), `metrics_comparison.csv`, `ablation.csv`, `summary.json` |

## Run in Google Colab

1. Upload `Churn_Leakage_Capstone.ipynb` to Colab.
2. Choose Runtime > Run all.
3. When the setup cell asks, upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` (found in the `data/` folder).

## Run locally

```
pip install -r requirements.txt
python src/leaky_pipeline.py
python src/fixed_pipeline.py
python src/compare.py
```

Run the commands from the project root. `compare.py` regenerates every chart and table in `outputs/`.
