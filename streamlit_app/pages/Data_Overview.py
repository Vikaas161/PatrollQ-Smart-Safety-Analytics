"""
Data Overview Page
Complete statistics and exploratory data analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.title("📊 Crime Data Overview")
st.markdown("Comprehensive analysis of Chicago crime dataset")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("data/processed/crime_data_final.parquet")

        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is not None:
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Basic Statistics",
        "🕐 Temporal Analysis",
        "📍 Geographic Distribution",
        "⚖️ Crime Severity"
    ])
    
    # ========================================================================
    # TAB 1: BASIC STATISTICS
    # ========================================================================
    with tab1:
        st.markdown("## Basic Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Dataset Dimensions")
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Features", df.shape[1])
            st.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
        
        with col2:
            st.markdown("### 🚔 Crime Statistics")
            st.metric("Crime Types", df['Primary Type'].nunique())
            st.metric("Arrest Rate", f"{df['Arrest'].mean()*100:.1f}%")
            st.metric("Domestic Incidents", f"{df['Domestic'].mean()*100:.1f}%")
        
        with col3:
            st.markdown("### 📍 Geographic Coverage")
            if 'District' in df.columns:
                st.metric("Police Districts", df['District'].nunique())
            if 'Ward' in df.columns:
                st.metric("City Wards", df['Ward'].nunique())
            st.metric("Unique Locations", df['Location Description'].nunique())
        
        st.markdown("---")
        
        # Crime type distribution
        st.markdown("### 🎯 Crime Type Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            top_n = st.slider("Show top N crime types", 5, 20, 10)
            crime_counts = df['Primary Type'].value_counts().head(top_n)
            
            fig = px.bar(
                x=crime_counts.values,
                y=crime_counts.index,
                orientation='h',
                title=f"Top {top_n} Crime Types",
                labels={'x': 'Number of Incidents', 'y': 'Crime Type'},
                color=crime_counts.values,
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Top 10 Details")
            top_10 = df['Primary Type'].value_counts().head(10)
            for crime, count in top_10.items():
                pct = (count / len(df)) * 100
                st.markdown(f"**{crime}**")
                st.progress(pct / 100)
                st.caption(f"{count:,} incidents ({pct:.1f}%)")
        
        # Location analysis
        st.markdown("### 📍 Top Crime Locations")
        
        location_counts = df['Location Description'].value_counts().head(15)
        
        fig_loc = px.bar(
            x=location_counts.values,
            y=location_counts.index,
            orientation='h',
            title="Top 15 Crime Locations",
            labels={'x': 'Number of Incidents', 'y': 'Location Type'},
            color=location_counts.values,
            color_continuous_scale='Blues'
        )
        fig_loc.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_loc, use_container_width=True)
    
    # ========================================================================
    # TAB 2: TEMPORAL ANALYSIS
    # ========================================================================
    with tab2:
        st.markdown("## ⏰ Temporal Crime Patterns")
        
        # Hourly distribution
        st.markdown("### 🕐 Crimes by Hour of Day")
        
        hourly_crimes = df.groupby('Hour').size().reset_index(name='Count')
        
        fig_hour = px.line(
            hourly_crimes,
            x='Hour',
            y='Count',
            title='Crime Distribution by Hour (24-hour format)',
            markers=True
        )
        fig_hour.update_traces(line_color='#EF4444', line_width=3, marker_size=8)
        fig_hour.update_layout(
            xaxis_title='Hour of Day',
            yaxis_title='Number of Crimes',
            plot_bgcolor='white',
            height=400
        )
        fig_hour.add_vrect(x0=22, x1=4, fillcolor="red", opacity=0.1, 
                           annotation_text="High Risk Hours", annotation_position="top left")
        st.plotly_chart(fig_hour, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Day of week
            st.markdown("### 📅 Crimes by Day of Week")
            
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_crimes = df.groupby('Day_of_Week').size().reindex(day_order).reset_index(name='Count')
            
            fig_day = px.bar(
                daily_crimes,
                x='Day_of_Week',
                y='Count',
                title='Weekly Crime Distribution',
                color='Count',
                color_continuous_scale='Viridis'
            )
            fig_day.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_day, use_container_width=True)
        
        with col2:
            # Monthly distribution
            st.markdown("### 📆 Crimes by Month")
            
            monthly_crimes = df.groupby('Month').size().reset_index(name='Count')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_crimes['Month_Name'] = monthly_crimes['Month'].apply(lambda x: month_names[x-1])
            
            fig_month = px.bar(
                monthly_crimes,
                x='Month_Name',
                y='Count',
                title='Seasonal Crime Patterns',
                color='Count',
                color_continuous_scale='YlOrRd'
            )
            fig_month.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_month, use_container_width=True)
        
        # Heatmap: Hour vs Day of Week
        st.markdown("### 🔥 Crime Heatmap: Hour × Day of Week")
        
        heatmap_data = df.groupby(['Hour', 'Day_of_Week']).size().reset_index(name='Count')
        heatmap_pivot = heatmap_data.pivot(index='Hour', columns='Day_of_Week', values='Count')
        heatmap_pivot = heatmap_pivot[day_order]
        
        fig_heatmap = px.imshow(
            heatmap_pivot,
            labels=dict(x="Day of Week", y="Hour of Day", color="Crime Count"),
            x=day_order,
            y=list(range(24)),
            color_continuous_scale='Reds',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=500, title="When Do Crimes Occur Most?")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Seasonal analysis
        st.markdown("### 🌤️ Seasonal Crime Patterns")
        
        seasonal_crimes = df.groupby('Season').size().reindex(
            ['Winter', 'Spring', 'Summer', 'Fall']
        ).reset_index(name='Count')
        
        fig_season = px.pie(
            seasonal_crimes,
            names='Season',
            values='Count',
            title='Crime Distribution by Season',
            color='Season',
            color_discrete_map={
                'Winter': '#3B82F6',
                'Spring': '#10B981',
                'Summer': '#F59E0B',
                'Fall': '#EF4444'
            }
        )
        fig_season.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_season, use_container_width=True)
    
    # ========================================================================
    # TAB 3: GEOGRAPHIC DISTRIBUTION
    # ========================================================================
    with tab3:
        st.markdown("## 📍 Geographic Crime Distribution")
        
        # Scatter plot of crime locations
        st.markdown("### 🗺️ Crime Density Map")
        
        # Sample for performance
        sample_size = min(10000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        fig_scatter = px.scatter_mapbox(
            df_sample,
            lat='Latitude',
            lon='Longitude',
            color='Crime_Severity',
            size='Crime_Severity',
            hover_data=['Primary Type', 'Date'],
            color_continuous_scale='Reds',
            zoom=10,
            height=600,
            title=f'Crime Locations (Sample of {sample_size:,} records)'
        )
        
        fig_scatter.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":40,"l":0,"b":0}
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Geographic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Latitude Distribution")
            fig_lat = px.histogram(
                df,
                x='Latitude',
                nbins=50,
                title='Crime Distribution by Latitude',
                color_discrete_sequence=['#3B82F6']
            )
            fig_lat.update_layout(height=300)
            st.plotly_chart(fig_lat, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Longitude Distribution")
            fig_lon = px.histogram(
                df,
                x='Longitude',
                nbins=50,
                title='Crime Distribution by Longitude',
                color_discrete_sequence=['#EF4444']
            )
            fig_lon.update_layout(height=300)
            st.plotly_chart(fig_lon, use_container_width=True)
    
    # ========================================================================
    # TAB 4: CRIME SEVERITY
    # ========================================================================
    with tab4:
        st.markdown("## ⚖️ Crime Severity Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Severity distribution
            st.markdown("### 📊 Severity Distribution")
            
            severity_counts = df['Crime_Severity'].value_counts().sort_index()
            
            fig_sev = px.bar(
                x=severity_counts.index,
                y=severity_counts.values,
                title='Crime Count by Severity Level',
                labels={'x': 'Severity Level (1=Low, 5=High)', 'y': 'Number of Crimes'},
                color=severity_counts.index,
                color_continuous_scale='RdYlGn_r'
            )
            fig_sev.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_sev, use_container_width=True)
            
            # Severity by crime type
            st.markdown("### 🎯 Top Crime Types by Severity")
            
            severity_by_type = df.groupby('Primary Type')['Crime_Severity'].agg(['mean', 'count']).reset_index()
            severity_by_type = severity_by_type[severity_by_type['count'] > 1000].sort_values('mean', ascending=False).head(15)
            
            fig_sev_type = px.bar(
                severity_by_type,
                x='mean',
                y='Primary Type',
                orientation='h',
                title='Average Severity by Crime Type (min 1000 incidents)',
                labels={'mean': 'Average Severity', 'Primary Type': 'Crime Type'},
                color='mean',
                color_continuous_scale='Reds'
            )
            fig_sev_type.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_sev_type, use_container_width=True)
        
        with col2:
            # Arrest rate by severity
            st.markdown("### 🚔 Arrest Rate by Severity")
            
            arrest_by_severity = df.groupby('Crime_Severity')['Arrest'].mean() * 100
            
            fig_arr = px.line(
                x=arrest_by_severity.index,
                y=arrest_by_severity.values,
                title='Arrest Rate vs Crime Severity',
                labels={'x': 'Severity Level', 'y': 'Arrest Rate (%)'},
                markers=True
            )
            fig_arr.update_traces(line_color='#10B981', line_width=3, marker_size=10)
            fig_arr.update_layout(height=400)
            st.plotly_chart(fig_arr, use_container_width=True)
            
            # Domestic vs Non-domestic
            st.markdown("### 🏠 Domestic vs Non-Domestic Crimes")
            
            domestic_data = pd.DataFrame({
                'Type': ['Domestic', 'Non-Domestic'],
                'Count': [
                    df['Domestic'].sum(),
                    (~df['Domestic']).sum()
                ]
            })
            
            fig_dom = px.pie(
                domestic_data,
                names='Type',
                values='Count',
                title='Domestic vs Non-Domestic Incidents',
                color='Type',
                color_discrete_map={
                    'Domestic': '#EF4444',
                    'Non-Domestic': '#3B82F6'
                }
            )
            fig_dom.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_dom, use_container_width=True)
            
            # Severity correlation
            st.markdown("### 📈 Severity Correlations")
            
            corr_data = df[['Crime_Severity', 'Arrest', 'Domestic']].corr()
            
            fig_corr = px.imshow(
                corr_data,
                text_auto='.2f',
                color_continuous_scale='RdBu',
                aspect='auto',
                title='Feature Correlations'
            )
            fig_corr.update_layout(height=300)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Download section
    st.markdown("---")
    st.markdown("### 💾 Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name="crime_data_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
        summary_stats = df.describe().to_csv().encode('utf-8')
        st.download_button(
            label="📊 Download Summary Statistics",
            data=summary_stats,
            file_name="crime_summary_stats.csv",
            mime="text/csv"
        )
    
    with col3:
        crime_counts = df['Primary Type'].value_counts().to_csv().encode('utf-8')
        st.download_button(
            label="📈 Download Crime Counts",
            data=crime_counts,
            file_name="crime_type_counts.csv",
            mime="text/csv"
        )

else:
    st.error("⚠️ Data file not found. Please run the preprocessing pipeline first.")
    st.code("""
cd PatrolIQ
python src/data_preprocessing.py
python src/feature_engineering.py
python src/clustering.py
python src/dimensionality.py
    """, language="bash")