"""
Tokenized Assets - Respondent Segmentation (K-Means)
====================================================
Interactive Streamlit app that segments survey respondents on the constructs of
the extended UTAUT2 model:

    PE  Performance Expectancy      (PE1-PE4)
    EE  Effort Expectancy           (EE1-EE4)
    FC  Facilitating Conditions     (FC1-FC4)
    DL  Digital Financial Literacy  (DL1-DL5)
    UT  User Trust                  (UT1-UT4)
    WTI Willingness to Invest       (WTI1-WTI3)  -> dependent variable

Run locally:   streamlit run app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
st.set_page_config(page_title="Tokenized Assets - K-Means Segmentation", page_icon="📊", layout="wide")

CONSTRUCTS = {
    "PE": ["PE1", "PE2", "PE3", "PE4"],
    "EE": ["EE1", "EE2", "EE3", "EE4"],
    "FC": ["FC1", "FC2", "FC3", "FC4"],
    "DL": ["DL1", "DL2", "DL3", "DL4", "DL5"],
    "UT": ["UT1", "UT2", "UT3", "UT4"],
    "WTI": ["WTI1", "WTI2", "WTI3"],
}
# Items retained in the final purified PLS-SEM measurement model
PURIFIED = {
    "PE": ["PE1", "PE2", "PE4"],
    "EE": ["EE1", "EE2", "EE4"],
    "FC": ["FC1", "FC3", "FC4"],
    "DL": ["DL2", "DL3", "DL4"],
    "UT": ["UT1", "UT2", "UT4"],
    "WTI": ["WTI1", "WTI2", "WTI3"],
}
LABELS = {
    "PE": "Performance Expectancy",
    "EE": "Effort Expectancy",
    "FC": "Facilitating Conditions",
    "DL": "Digital Financial Literacy",
    "UT": "User Trust",
    "WTI": "Willingness to Invest",
}
PREDICTORS = ["PE", "EE", "FC", "DL", "UT"]
ALL_CONSTRUCTS = PREDICTORS + ["WTI"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # one fixed colour per cluster
RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent


# --------------------------------------------------------------------------
# Data loading and preprocessing
# --------------------------------------------------------------------------
def find_default_file():
    """Return the first Excel file in the repository root (prefers 'Clean Data...')."""
    files = sorted(list(DATA_DIR.glob("*.xls")) + list(DATA_DIR.glob("*.xlsx")))
    files = [f for f in files if not f.name.startswith("~$")]
    preferred = [f for f in files if "clean data" in f.name.lower()]
    return (preferred or files or [None])[0]


@st.cache_data(show_spinner=False)
def read_excel(source, name):
    """Read .xls (xlrd) or .xlsx (openpyxl). `name` is only used to pick the engine."""
    engine = "xlrd" if str(name).lower().endswith(".xls") else "openpyxl"
    return pd.read_excel(source, engine=engine)


def clean_items(raw):
    """Standardise column names, keep the survey items, coerce to numbers and drop incomplete rows."""
    df = raw.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    # accept the alternative item names used in earlier versions of the questionnaire
    df = df.rename(columns=lambda c: c.replace("DFL", "DL") if c.startswith("DFL") else c)
    df = df.rename(columns=lambda c: c.replace("WI", "WTI", 1) if c.startswith("WI") and not c.startswith("WTI") else c)
    items = [i for group in CONSTRUCTS.values() for i in group]
    missing = [i for i in items if i not in df.columns]
    df = df[[i for i in items if i in df.columns]].apply(pd.to_numeric, errors="coerce")
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    return df, missing, n_before - len(df)


def add_construct_scores(items, item_map):
    """Mean of the items of each construct, per respondent."""
    scores = pd.DataFrame(index=items.index)
    for construct, cols in item_map.items():
        cols = [c for c in cols if c in items.columns]
        scores[construct] = items[cols].mean(axis=1)
    return scores


# --------------------------------------------------------------------------
# Clustering helpers
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_kmeans(X, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    # relabel so that Cluster 1 is always the largest one (stable naming and colours)
    order = pd.Series(labels).value_counts().index.tolist()
    mapping = {old: new + 1 for new, old in enumerate(order)}
    labels = np.array([mapping[l] for l in labels])
    sil = silhouette_score(X, labels) if 1 < k < len(X) else np.nan
    return labels, float(model.inertia_), float(sil)


@st.cache_data(show_spinner=False)
def k_diagnostics(X, k_values):
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sil = silhouette_score(X, model.labels_) if k > 1 else np.nan
        rows.append({"k": k, "Inertia": model.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)


def standardized_betas(df, predictors, target="WTI"):
    """Standardised OLS coefficients of `target` on `predictors` (descriptive, not a hypothesis test)."""
    if len(df) <= len(predictors) + 1:
        return pd.Series(np.nan, index=predictors)
    Z = (df[predictors + [target]] - df[predictors + [target]].mean()) / df[predictors + [target]].std(ddof=0)
    Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0)
    model = LinearRegression().fit(Z[predictors], Z[target])
    return pd.Series(model.coef_, index=predictors)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

st.sidebar.subheader("1 · Data")
uploaded = st.sidebar.file_uploader("Upload survey file (.xls / .xlsx)", type=["xls", "xlsx"])
default_file = find_default_file()

if uploaded is not None:
    raw, source_name = read_excel(uploaded, uploaded.name), uploaded.name
elif default_file is not None:
    raw, source_name = read_excel(str(default_file), default_file.name), default_file.name
else:
    st.title("Tokenized Assets - Respondent Segmentation")
    st.info("No data file was found in the repository. Upload the survey file (.xls or .xlsx) in the sidebar to start.")
    st.stop()

items, missing_items, dropped_rows = clean_items(raw)
if missing_items:
    st.error("The file is missing these item columns: " + ", ".join(missing_items))
    st.stop()
st.sidebar.caption(f"Loaded **{source_name}** · {len(items)} respondents · {items.shape[1]} items")
if dropped_rows:
    st.sidebar.warning(f"{dropped_rows} incomplete response(s) were removed.")

st.sidebar.subheader("2 · Construct scores")
use_purified = st.sidebar.checkbox(
    "Use purified item set (final PLS-SEM model)", value=False,
    help="Off: mean of all items of each construct. On: only the items retained in the final measurement model "
         "(PE1, PE2, PE4 · EE1, EE2, EE4 · FC1, FC3, FC4 · DL2, DL3, DL4 · UT1, UT2, UT4 · WTI1-WTI3).")
item_map = PURIFIED if use_purified else CONSTRUCTS
scores = add_construct_scores(items, item_map)

st.sidebar.subheader("3 · K-Means")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=5, value=2, step=1)
cluster_vars = st.sidebar.multiselect(
    "Constructs used to form the clusters", options=ALL_CONSTRUCTS, default=PREDICTORS,
    format_func=lambda c: f"{c} - {LABELS[c]}",
    help="By default the clusters are formed on the five independent constructs. WTI is left out so that the "
         "segments can be compared on the dependent variable without circularity.")
scale = st.sidebar.checkbox("Standardise constructs (StandardScaler)", value=True,
                            help="Recommended: puts all constructs on the same scale (mean 0, SD 1) before clustering.")

if len(cluster_vars) < 2:
    st.warning("Select at least two constructs to form the clusters.")
    st.stop()

X = scores[cluster_vars].to_numpy(dtype=float)
if scale:
    X = StandardScaler().fit_transform(X)

labels, inertia, silhouette = run_kmeans(X, k)
cluster_names = [f"Cluster {i}" for i in range(1, k + 1)]
color_map = {name: PALETTE[i] for i, name in enumerate(cluster_names)}
data = pd.concat([items, scores], axis=1)
data["Cluster"] = pd.Categorical([f"Cluster {l}" for l in labels], categories=cluster_names, ordered=True)

# --------------------------------------------------------------------------
# Header and key metrics
# --------------------------------------------------------------------------
st.title("📊 Tokenized Assets - Respondent Segmentation")
st.caption("K-Means clustering on the constructs of the extended UTAUT2 model · User willingness to adopt "
           "blockchain-based platforms for digital assets in Singapore")

sizes = data["Cluster"].value_counts().reindex(cluster_names)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Respondents", len(data))
m2.metric("Clusters (k)", k)
m3.metric("Silhouette score", f"{silhouette:.3f}", help="From -1 to 1. Higher = better separated clusters.")
m4.metric("Inertia", f"{inertia:,.1f}", help="Within-cluster sum of squares. Lower = tighter clusters.")
st.caption("Cluster sizes: " + " · ".join(f"**{n}** n = {int(sizes[n])} ({sizes[n] / len(data):.0%})" for n in cluster_names))

with st.expander("How to choose k: Elbow method and Silhouette score"):
    diag = k_diagnostics(X, tuple(range(1, 9)))
    c1, c2 = st.columns(2)
    fig = px.line(diag, x="k", y="Inertia", markers=True, title="Elbow method (inertia)")
    fig.update_traces(line_color=PALETTE[0])
    fig.add_vline(x=k, line_dash="dash", line_color="gray")
    c1.plotly_chart(fig, use_container_width=True)
    fig = px.line(diag.dropna(), x="k", y="Silhouette", markers=True, title="Silhouette score by k")
    fig.update_traces(line_color=PALETTE[0])
    fig.add_vline(x=k, line_dash="dash", line_color="gray")
    c2.plotly_chart(fig, use_container_width=True)
    st.caption("Look for the 'elbow' where inertia stops falling quickly, and for the k with the highest silhouette. "
               "The dashed line marks the k currently selected.")

tab1, tab2, tab3, tab4 = st.tabs(["🔵 Cluster visualisation (PCA)", "📋 Cluster profiling",
                                  "🔥 Construct comparison & heatmaps", "💾 Data & export"])

# --------------------------------------------------------------------------
# Tab 1 - PCA 2D / 3D
# --------------------------------------------------------------------------
with tab1:
    st.subheader("Clusters in the principal component space")
    n_comp = min(3, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    comps = pca.fit_transform(X)
    pcs = [f"PC{i + 1}" for i in range(n_comp)]
    pca_df = pd.DataFrame(comps, columns=pcs)
    pca_df["Cluster"] = data["Cluster"].astype(str)
    pca_df["Respondent"] = np.arange(1, len(pca_df) + 1)
    for c in ALL_CONSTRUCTS:
        pca_df[c] = scores[c].round(2)
    var = pca.explained_variance_ratio_
    axis_title = {p: f"{p} ({v:.0%} of variance)" for p, v in zip(pcs, var)}

    c1, c2 = st.columns(2)
    fig2d = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", color_discrete_map=color_map,
                       category_orders={"Cluster": cluster_names}, hover_data=["Respondent"] + ALL_CONSTRUCTS,
                       labels=axis_title, title="PCA - 2D")
    fig2d.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    c1.plotly_chart(fig2d, use_container_width=True)

    if n_comp >= 3:
        fig3d = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Cluster", color_discrete_map=color_map,
                              category_orders={"Cluster": cluster_names}, hover_data=["Respondent"] + ALL_CONSTRUCTS,
                              labels=axis_title, title="PCA - 3D (drag to rotate, scroll to zoom)")
        fig3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color="white")))
        fig3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        c2.plotly_chart(fig3d, use_container_width=True)
    else:
        c2.info("Select at least three constructs to see the 3D view.")

    st.caption(f"The first {n_comp} components explain {var.sum():.0%} of the variance of the clustering variables. "
               "PCA is used only to draw the clusters; K-Means runs on all the selected constructs.")
    with st.expander("PCA loadings - what each component represents"):
        loadings = pd.DataFrame(pca.components_.T, index=[f"{c} - {LABELS[c]}" for c in cluster_vars], columns=pcs)
        st.dataframe(loadings.style.format("{:.2f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                     use_container_width=True)

# --------------------------------------------------------------------------
# Tab 2 - Profiling
# --------------------------------------------------------------------------
with tab2:
    st.subheader("Mean construct scores by cluster")
    means = data.groupby("Cluster", observed=False)[ALL_CONSTRUCTS].mean()
    long = means.reset_index().melt(id_vars="Cluster", var_name="Construct", value_name="Mean")
    long["Cluster"] = long["Cluster"].astype(str)

    c1, c2 = st.columns(2)
    bar = px.bar(long, x="Construct", y="Mean", color="Cluster", barmode="group", color_discrete_map=color_map,
                 category_orders={"Cluster": cluster_names, "Construct": ALL_CONSTRUCTS}, text_auto=".2f",
                 title="Grouped bar chart (1 = strongly disagree, 5 = strongly agree)")
    bar.update_layout(yaxis_range=[1, 5])
    c1.plotly_chart(bar, use_container_width=True)

    radar = go.Figure()
    for name in cluster_names:
        vals = means.loc[name, ALL_CONSTRUCTS].tolist()
        radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=ALL_CONSTRUCTS + ALL_CONSTRUCTS[:1],
                                        name=name, fill="toself", opacity=0.6, line=dict(color=color_map[name])))
    radar.update_layout(title="Radar chart", polar=dict(radialaxis=dict(range=[1, 5], visible=True)))
    c2.plotly_chart(radar, use_container_width=True)

    st.subheader("Summary statistics")
    summary = pd.DataFrame({"N": sizes.to_numpy(dtype=int),
                            "% of sample": (sizes.to_numpy(dtype=float) / len(data) * 100).round(1)},
                           index=pd.Index(cluster_names, name="Cluster"))
    stats = data.groupby("Cluster", observed=False)[ALL_CONSTRUCTS].agg(["mean", "std"])
    for c in ALL_CONSTRUCTS:
        summary[f"{c} mean"] = stats[(c, "mean")].round(2).to_numpy()
        summary[f"{c} SD"] = stats[(c, "std")].round(2).to_numpy()
    total = {"N": len(data), "% of sample": 100.0}
    for c in ALL_CONSTRUCTS:
        total[f"{c} mean"] = round(data[c].mean(), 2)
        total[f"{c} SD"] = round(data[c].std(), 2)
    summary.loc["Total sample"] = total
    summary["N"] = summary["N"].astype(int)
    st.dataframe(summary, use_container_width=True)

    st.subheader("What distinguishes each cluster")
    overall = data[ALL_CONSTRUCTS].mean()
    sd = data[ALL_CONSTRUCTS].std()
    for name in cluster_names:
        z = ((means.loc[name] - overall) / sd).sort_values()
        low, high = z.index[0], z.index[-1]
        st.markdown(f"**{name}** (n = {int(sizes[name])}, {sizes[name] / len(data):.0%}): highest relative score in "
                    f"**{LABELS[high]}** ({means.loc[name, high]:.2f}); lowest in **{LABELS[low]}** "
                    f"({means.loc[name, low]:.2f}).")
    st.caption("Relative scores compare each cluster mean with the mean of the total sample.")

# --------------------------------------------------------------------------
# Tab 3 - Comparison and heatmaps
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Correlations between constructs, by cluster")
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 4.2), squeeze=False)
    for ax, name in zip(axes[0], cluster_names):
        sub = data.loc[data["Cluster"] == name, ALL_CONSTRUCTS]
        sns.heatmap(sub.corr(), annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
                    cbar=name == cluster_names[-1], ax=ax, annot_kws={"size": 8}, linewidths=0.5, linecolor="white")
        ax.set_title(f"{name} (n = {len(sub)})", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Pearson correlations of the construct scores inside each cluster. The last row/column (WTI) shows "
               "which constructs move together with Willingness to Invest in that segment.")

    st.subheader("Distribution of a construct across clusters")
    target = st.selectbox("Construct", ALL_CONSTRUCTS, index=ALL_CONSTRUCTS.index("WTI"),
                          format_func=lambda c: f"{c} - {LABELS[c]}")
    box_df = data[[target, "Cluster"]].copy()
    box_df["Cluster"] = box_df["Cluster"].astype(str)
    box = px.box(box_df, x="Cluster", y=target, color="Cluster", color_discrete_map=color_map, points="all",
                 category_orders={"Cluster": cluster_names}, title=f"{LABELS[target]} by cluster")
    box.update_layout(showlegend=False, yaxis_range=[0.8, 5.2])
    st.plotly_chart(box, use_container_width=True)

    st.subheader("Drivers of Willingness to Invest inside each cluster")
    betas = pd.DataFrame({name: standardized_betas(data[data["Cluster"] == name], PREDICTORS) for name in cluster_names})
    betas["Total sample"] = standardized_betas(data, PREDICTORS)
    beta_long = betas[cluster_names].reset_index().melt(id_vars="index", var_name="Cluster", value_name="Beta")
    beta_fig = px.bar(beta_long, x="index", y="Beta", color="Cluster", barmode="group", color_discrete_map=color_map,
                      category_orders={"Cluster": cluster_names, "index": PREDICTORS}, text_auto=".2f",
                      labels={"index": "Construct", "Beta": "Standardised β on WTI"},
                      title="Standardised regression of WTI on the five constructs, by cluster")
    st.plotly_chart(beta_fig, use_container_width=True)
    st.dataframe(betas.rename(index=LABELS).style.format("{:.3f}"), use_container_width=True)
    st.caption("Exploratory and descriptive: ordinary least squares on standardised construct scores within each "
               "cluster. Small clusters give unstable estimates; this does not replace the PLS-SEM hypothesis tests.")

# --------------------------------------------------------------------------
# Tab 4 - Data and export
# --------------------------------------------------------------------------
with tab4:
    st.subheader("Dataset with construct scores and cluster assignment")
    export = data.copy()
    export[ALL_CONSTRUCTS] = export[ALL_CONSTRUCTS].round(3)
    export["Cluster"] = export["Cluster"].astype(str)
    export.insert(0, "Respondent", np.arange(1, len(export) + 1))
    st.dataframe(export, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download as CSV", data=export.to_csv(index=False).encode("utf-8"),
                       file_name=f"digital_assets_clusters_k{k}.csv", mime="text/csv")
    st.caption(f"Settings used: k = {k}; clustering variables = {', '.join(cluster_vars)}; "
               f"standardised = {'yes' if scale else 'no'}; item set = {'purified' if use_purified else 'all items'}; "
               f"random_state = {RANDOM_STATE}.")

st.divider()
st.caption("Method: K-Means (scikit-learn, n_init = 10, random_state = 42) on construct mean scores; PCA for "
           "visualisation. This is an exploratory, descriptive analysis that complements the confirmatory PLS-SEM model.")
