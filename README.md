# Data Leakage Audit: Telco Customer Churn

NeuroFive Capstone (Machine Learning). A deliberately flawed churn pipeline is audited, fixed and compared with charts.

## Contents

| Path | Purpose |
|------|---------|
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
