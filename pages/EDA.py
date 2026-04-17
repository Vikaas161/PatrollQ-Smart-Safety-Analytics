import streamlit as st
import pandas as pd
import plotly.express as px


st.title("Chicago Crime Data Overview")

df = pd.read_csv("data/crime_500k_cleaned.csv")
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dateset Shape")
st.write(df.shape)

st.subheader("Crime Type Distribution")

num = st.selectbox(
    "Select N Top Crimes",
    [5,10,15,20,25,30]
)
crime_counts = df["Primary Type"].value_counts().head(num)

fig = px.bar(
    x=crime_counts.index,
    y=crime_counts.values,
    title="Top 10 Crime Types",
    labels={"x": "Crime Type", "y": "Count"}
)

st.plotly_chart(fig)