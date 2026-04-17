import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.title("Crime Hotspot Analysis")

df = pd.read_csv("data/crime_500k_cleaned.csv")

with open("models/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("models/dbscan.pkl", "rb") as f:
    dbscan = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/dbscan_scaler.pkl", "rb") as f:
    dbscan_scaler = pickle.load(f)

X_kmeans = df[["Latitude","Longitude"]]
X_kmeans_scaled = scaler.transform(X_kmeans)

df["KMeans Cluster"] = kmeans.predict(X_kmeans_scaled).astype(str)

st.subheader("KMeans Crime Hotspots")

fig1 = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="KMeans Cluster",
    height=600,
    title="Crime Hotspots using KMeans",
    mapbox_style="carto-positron"
)

st.plotly_chart(fig1)

df_db = df.sample(100000, random_state=42)

X_db = df_db[["Latitude","Longitude"]]
X_db_scaled = dbscan_scaler.transform(X_db)

db_labels = dbscan.fit_predict(X_db_scaled)

df_db["DBSCAN Cluster"] = db_labels.astype(str)

st.subheader("DBSCAN Crime Hotspots")

fig2 = px.scatter_mapbox(
    df_db,
    lat="Latitude",
    lon="Longitude",
    color="DBSCAN Cluster",
    height=600,
    title="Crime Hotspots using DBSCAN",
    mapbox_style="carto-positron"
)

st.plotly_chart(fig2)