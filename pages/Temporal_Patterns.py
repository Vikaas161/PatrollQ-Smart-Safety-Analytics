import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Temporal Crime Pattern Analysis")

df = pd.read_csv("data/crime_500k_cleaned.csv")

hours_count = df.groupby("Hour").size().reset_index(name="Count")

fig = px.line(
    hours_count,
    x="Hour",
    y="Count",
    markers=True,
    title="Crime Distribution by Hour"
)

st.plotly_chart(fig)


days_count = df["Day_of_Week"].value_counts().reset_index()
days_count.columns = ["Day", "Count"]

fig1 = px.bar(
    days_count,
    x="Day",
    y="Count",
    title="Crime Distribution by Day",
)

st.plotly_chart(fig1)


month_counts = df["Month"].value_counts().reset_index(name="Count")

fig2 = px.bar(
    month_counts,
    x="Month",
    y="Count",
    title="Crime Distribution by Month"
)

st.plotly_chart(fig2)


weekend_counts = df.groupby("Is_Weekend").size().reset_index(name="Count")

fig3 = px.bar(
    weekend_counts,
    x="Is_Weekend",
    y="Count",
    title="Weekend vs Weekday Crime Distribution"
)

st.plotly_chart(fig3)

seasons = df.groupby("Season").size().reset_index(name="Count")

fig4 = px.line(
    seasons,
    x="Season",
    y="Count",
    title="Seasonal Crime Trend"
)

st.plotly_chart(fig4)

seasons = df.groupby("Domestic").size().reset_index(name="Count")

fig4 = px.bar(
    seasons,
    x="Domestic",
    y="Count",
    title="Domestic Crime Trend"
)

st.plotly_chart(fig4)

year_counts = df.groupby("Year").size().reset_index(name="Count")

fig5 = px.line(
    year_counts,
    x="Year",
    y="Count",
    markers=True,
    title="Crime Trend by Year"
)

st.plotly_chart(fig5)

crime_hour = df.groupby(["Hour","Primary Type"]).size().reset_index(name="Count")

fig = px.line(
    crime_hour,
    x="Hour",
    y="Count",
    color="Primary Type",
    title="Crime Type vs Hour"
)

st.plotly_chart(fig)