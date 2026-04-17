import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.manifold import TSNE

st.title("Dimensionality Reduction (PCA)")

df = pd.read_csv("data/crime_500k_cleaned.csv")

le = LabelEncoder()
df["Primary Type"] = le.fit_transform(df["Primary Type"])

with open("models/PCA_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/pca.pkl", "rb") as f:
    pca = pickle.load(f)

features = [
    'Primary Type',
    'Arrest',
    'Domestic',
    'Latitude',
    'Longitude',
    'Hour',
    'Month',
    'Crime_Severity_Score'
]

X = df[features]
X_scaled = scaler.transform(X)

X_pca = pca.transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=[f"PCA{i+1}" for i in range(X_pca.shape[1])]
)

pca_df = pca_df.sample(20000, random_state=42)

fig = px.scatter(
    pca_df,
    x="PCA1",
    y="PCA2",
    opacity=0.7,
    title="Crime Data in PCA Space"
)

st.plotly_chart(fig, use_container_width=True)

variance_df = pd.DataFrame({
    "Component": [f"PCA{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "Variance": pca.explained_variance_ratio_
})

st.subheader("Explained Variance Ratio")

st.write(variance_df)

importance = np.abs(pca.components_[0])
top_features = pd.Series(importance, index=X.columns).sort_values(ascending=False)
st.subheader("Top 5 features of PCA")
st.dataframe(top_features.head(5).reset_index().rename(columns={"index": "Feature", 0: "Importance"}))

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

fig1 = px.line(
    x=range(1, len(cumulative_variance) + 1),
    y=cumulative_variance,
    markers=True,
    labels={
        "x": "Number of Components",
        "y": "Cumulative Explained Variance"
    },
    title="PCA Scree Plot"
)

st.plotly_chart(fig1)

tsne_sample = X_scaled[:5000]

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=1000
)

X_tsne = tsne.fit_transform(tsne_sample)

tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:,0],
    "TSNE2": X_tsne[:,1],
    "Crime Type": df["Primary Type"][:5000].astype(str)
})

fig_tsne = px.scatter(
    tsne_df,
    x="TSNE1",
    y="TSNE2",
    color="Crime Type",
    opacity=0.7,
    title="t-SNE Crime Type Clusters"
)

st.plotly_chart(fig_tsne)