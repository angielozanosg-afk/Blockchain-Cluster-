# Tokenized Assets – Respondent Segmentation (K-Means)

Interactive Streamlit app for the research project *User Trust and Willingness to Adopt
Blockchain-Based Financial Platforms in Digital Assets in Singapore*.

It segments survey respondents with K-Means on the constructs of the extended UTAUT2 model
(PE, EE, FC, DL, UT) and compares the segments on Willingness to Invest (WTI).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `requirements.txt` | Python dependencies |
| `Clean Data Digital Assets .xls` | Survey data (98 responses, 24 Likert items) – loaded automatically |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What the app shows

1. **Cluster visualisation** – PCA in 2D and 3D, coloured by cluster.
2. **Cluster profiling** – grouped bar chart, radar chart and summary statistics (mean, SD, N, %).
3. **Construct comparison & heatmaps** – correlations by cluster, boxplots, and drivers of WTI by cluster.
4. **Data & export** – dataset with construct scores and cluster assignment, downloadable as CSV.

Sidebar settings: number of clusters (k = 2–5), constructs used for clustering, standardisation,
and the item set (all items or the purified set of the final PLS-SEM model).

Method: K-Means (scikit-learn, `n_init=10`, `random_state=42`) on construct mean scores.
Exploratory and descriptive; it complements the confirmatory PLS-SEM analysis.
