"""
Crime Hotspots Page
Interactive map showing geographic crime clusters
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np

st.set_page_config(page_title="Crime Hotspots", page_icon="🗺️", layout="wide")

st.title("🗺️ Crime Hotspots Analysis")
st.markdown("Geographic clustering and hotspot identification")

# Load data
@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/crime_data_final.parquet")

    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Map Controls")
    
    clustering_method = st.sidebar.selectbox(
        "Select Clustering Method",
        ["K-Means", "DBSCAN", "Hierarchical"],
        help="Choose which clustering algorithm to visualize"
    )
    
    cluster_col_map = {
        "K-Means": "KMeans_Cluster",
        "DBSCAN": "DBSCAN_Cluster",
        "Hierarchical": "Hierarchical_Cluster"
    }
    
    cluster_col = cluster_col_map[clustering_method]
    
    if cluster_col not in df.columns:
        st.error(f"⚠️ {clustering_method} clustering not found. Please run clustering.py first.")
        st.stop()
    
    visualization_type = st.sidebar.radio(
        "Visualization Type",
        ["Heatmap", "Cluster Map", "Individual Points"]
    )
    
    sample_size = st.sidebar.slider(
        "Sample Size (for performance)",
        1000, 50000, 10000, 1000
    )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Interactive Map",
        "📊 Cluster Analysis",
        "📈 Hotspot Statistics"
    ])
    
    with tab1:
        st.markdown(f"### {clustering_method} Clustering Visualization")
        
        # Sample data
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        if visualization_type == "Heatmap":
            # Create heatmap
            st.markdown("#### 🔥 Crime Density Heatmap")
            
            # Create base map
            chicago_coords = [41.8781, -87.6298]
            m = folium.Map(
                location=chicago_coords,
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # Prepare heat data
            heat_data = [[row['Latitude'], row['Longitude']] 
                        for idx, row in df_sample.iterrows()]
            
            # Add heatmap
            HeatMap(
                heat_data,
                radius=15,
                blur=25,
                max_zoom=13,
                gradient={0.0: 'blue', 0.5: 'yellow', 0.75: 'orange', 1.0: 'red'}
            ).add_to(m)
            
            # Display map
            st_folium(m, width=1200, height=600)
            
        elif visualization_type == "Cluster Map":
            # Create cluster map
            st.markdown(f"#### 🎯 {clustering_method} Cluster Map")
            
            # Filter out noise in DBSCAN
            if clustering_method == "DBSCAN":
                df_sample = df_sample[df_sample[cluster_col] != -1]
            
            # Create plotly map
            fig = px.scatter_mapbox(
                df_sample,
                lat='Latitude',
                lon='Longitude',
                color=cluster_col,
                hover_data=['Primary Type', 'Date', 'Crime_Severity'],
                color_continuous_scale='Rainbow',
                zoom=10,
                height=600,
                title=f'{clustering_method} Crime Clusters'
            )
            
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0,"t":40,"l":0,"b":0}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster statistics
            st.markdown("#### 📊 Cluster Summary")
            
            cluster_stats = df_sample.groupby(cluster_col).agg({
                'ID': 'count',
                'Arrest': 'mean',
                'Crime_Severity': 'mean',
                'Latitude': 'mean',
                'Longitude': 'mean'
            }).reset_index()
            
            cluster_stats.columns = ['Cluster', 'Crime Count', 'Arrest Rate', 
                                    'Avg Severity', 'Center Lat', 'Center Lon']
            cluster_stats['Arrest Rate'] = (cluster_stats['Arrest Rate'] * 100).round(1)
            cluster_stats['Avg Severity'] = cluster_stats['Avg Severity'].round(2)
            
            st.dataframe(cluster_stats, use_container_width=True)
            
        else:  # Individual Points
            # Create scatter map with individual points
            st.markdown("#### 📍 Individual Crime Locations")
            
            # Create map with Folium
            chicago_coords = [41.8781, -87.6298]
            m = folium.Map(
                location=chicago_coords,
                zoom_start=11,
                tiles='OpenStreetMap'
            )
            
            # Add marker cluster
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add points (sample for performance)
            for idx, row in df_sample.head(1000).iterrows():
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=3,
                    popup=f"{row['Primary Type']}<br>{row['Date']}",
                    color='red',
                    fill=True,
                    fillOpacity=0.6
                ).add_to(marker_cluster)
            
            # Display map
            st_folium(m, width=1200, height=600)
    
    with tab2:
        st.markdown("## 📊 Cluster Analysis")
        
        # Cluster comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Crimes per Cluster")
            
            cluster_counts = df[cluster_col].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']
            
            # Remove noise cluster for DBSCAN
            if clustering_method == "DBSCAN":
                cluster_counts = cluster_counts[cluster_counts['Cluster'] != -1]
            
            fig_bar = px.bar(
                cluster_counts.head(20),
                x='Cluster',
                y='Count',
                title=f'Crime Distribution Across {clustering_method} Clusters',
                color='Count',
                color_continuous_scale='Reds'
            )
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Cluster Characteristics")
            
            # Get cluster centers and characteristics
            cluster_chars = df.groupby(cluster_col).agg({
                'Crime_Severity': 'mean',
                'Arrest': 'mean',
                'Domestic': 'mean'
            }).reset_index()
            
            cluster_chars.columns = ['Cluster', 'Avg Severity', 'Arrest Rate', 'Domestic Rate']
            cluster_chars['Arrest Rate'] *= 100
            cluster_chars['Domestic Rate'] *= 100
            
            # Remove noise for DBSCAN
            if clustering_method == "DBSCAN":
                cluster_chars = cluster_chars[cluster_chars['Cluster'] != -1]
            
            fig_scatter = px.scatter(
                cluster_chars,
                x='Avg Severity',
                y='Arrest Rate',
                size='Domestic Rate',
                color='Cluster',
                title='Cluster Risk Profile',
                labels={'Avg Severity': 'Average Crime Severity', 
                       'Arrest Rate': 'Arrest Rate (%)'},
                color_continuous_scale='Viridis'
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top crime types per cluster
        st.markdown("### 🔝 Top Crime Types by Cluster")
        
        selected_cluster = st.selectbox(
            "Select Cluster to Analyze",
            sorted(df[cluster_col].unique())
        )
        
        cluster_data = df[df[cluster_col] == selected_cluster]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Crimes", f"{len(cluster_data):,}")
            st.metric("Arrest Rate", f"{cluster_data['Arrest'].mean()*100:.1f}%")
        
        with col2:
            st.metric("Avg Severity", f"{cluster_data['Crime_Severity'].mean():.2f}")
            st.metric("Domestic Rate", f"{cluster_data['Domestic'].mean()*100:.1f}%")
        
        with col3:
            center_lat = cluster_data['Latitude'].mean()
            center_lon = cluster_data['Longitude'].mean()
            st.metric("Center Latitude", f"{center_lat:.4f}")
            st.metric("Center Longitude", f"{center_lon:.4f}")
        
        # Top crimes in selected cluster
        top_crimes = cluster_data['Primary Type'].value_counts().head(10)
        
        fig_crimes = px.bar(
            x=top_crimes.values,
            y=top_crimes.index,
            orientation='h',
            title=f'Top 10 Crimes in Cluster {selected_cluster}',
            labels={'x': 'Number of Incidents', 'y': 'Crime Type'},
            color=top_crimes.values,
            color_continuous_scale='Blues'
        )
        fig_crimes.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_crimes, use_container_width=True)
        
        # Time analysis for selected cluster
        st.markdown("### ⏰ Temporal Patterns in Selected Cluster")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hourly = cluster_data.groupby('Hour').size()
            fig_hour = px.line(
                x=hourly.index,
                y=hourly.values,
                title='Crimes by Hour',
                labels={'x': 'Hour', 'y': 'Crime Count'},
                markers=True
            )
            fig_hour.update_traces(line_color='#EF4444', line_width=2)
            fig_hour.update_layout(height=300)
            st.plotly_chart(fig_hour, use_container_width=True)
        
        with col2:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
            daily = cluster_data.groupby('Day_of_Week').size().reindex(day_order)
            fig_day = px.bar(
                x=daily.index,
                y=daily.values,
                title='Crimes by Day of Week',
                labels={'x': 'Day', 'y': 'Crime Count'},
                color=daily.values,
                color_continuous_scale='Viridis'
            )
            fig_day.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_day, use_container_width=True)
    
    with tab3:
        st.markdown("## 📈 Hotspot Statistics")
        
        # Overall hotspot summary
        st.markdown("### 🎯 Hotspot Summary")
        
        n_clusters = df[cluster_col].nunique()
        if clustering_method == "DBSCAN":
            n_clusters -= 1  # Exclude noise cluster
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Hotspots", n_clusters)
        
        with col2:
            avg_cluster_size = len(df) / n_clusters
            st.metric("Avg Crimes/Hotspot", f"{avg_cluster_size:,.0f}")
        
        with col3:
            # Most dangerous cluster
            cluster_severity = df.groupby(cluster_col)['Crime_Severity'].mean()
            most_dangerous = cluster_severity.idxmax()
            st.metric("Highest Risk Cluster", int(most_dangerous))
        
        with col4:
            # Best arrest rate cluster
            cluster_arrests = df.groupby(cluster_col)['Arrest'].mean()
            best_arrests = cluster_arrests.idxmax()
            st.metric("Best Arrest Rate Cluster", int(best_arrests))
        
        # Detailed cluster ranking
        st.markdown("### 🏆 Cluster Rankings")
        
        ranking_metric = st.selectbox(
            "Rank clusters by:",
            ["Crime Count", "Severity", "Arrest Rate", "Domestic Rate"]
        )
        
        if ranking_metric == "Crime Count":
            rankings = df.groupby(cluster_col).size().sort_values(ascending=False).reset_index()
            rankings.columns = ['Cluster', 'Crime Count']
        elif ranking_metric == "Severity":
            rankings = df.groupby(cluster_col)['Crime_Severity'].mean().sort_values(ascending=False).reset_index()
            rankings.columns = ['Cluster', 'Avg Severity']
        elif ranking_metric == "Arrest Rate":
            rankings = df.groupby(cluster_col)['Arrest'].mean().sort_values(ascending=False).reset_index()
            rankings.columns = ['Cluster', 'Arrest Rate']
            rankings['Arrest Rate'] = (rankings['Arrest Rate'] * 100).round(1)
        else:  # Domestic Rate
            rankings = df.groupby(cluster_col)['Domestic'].mean().sort_values(ascending=False).reset_index()
            rankings.columns = ['Cluster', 'Domestic Rate']
            rankings['Domestic Rate'] = (rankings['Domestic Rate'] * 100).round(1)
        
        rankings['Rank'] = range(1, len(rankings) + 1)
        rankings = rankings[['Rank', 'Cluster'] + [col for col in rankings.columns if col not in ['Rank', 'Cluster']]]
        
        st.dataframe(rankings.head(20), use_container_width=True)
        
        # Download hotspot data
        st.markdown("### 💾 Export Hotspot Analysis")
        
        csv = rankings.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hotspot Rankings",
            data=csv,
            file_name=f"{clustering_method}_hotspot_rankings.csv",
            mime="text/csv"
        )

except FileNotFoundError:
    st.error("⚠️ Data file not found. Please run the preprocessing and clustering pipeline first.")

# Sidebar info
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Clustering Methods")
    
    st.markdown("""
    **K-Means**
    - Creates circular zones
    - Easy to understand
    - Fixed number of clusters
    
    **DBSCAN**
    - Finds natural clusters
    - Handles noise
    - Variable cluster shapes
    
    **Hierarchical**
    - Nested structure
    - Shows relationships
    - Flexible granularity
    """)