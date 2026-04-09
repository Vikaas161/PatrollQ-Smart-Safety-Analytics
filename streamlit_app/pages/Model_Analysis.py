"""
Model Analysis Page
Combines dimensionality reduction visualizations and model performance metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(page_title="Model Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Model Analysis & Dimensionality Reduction")
st.markdown("Explore complex crime patterns in simple 2D visualizations and compare model performance")

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
        "📊 Dimensionality Reduction",
        "🏆 Model Performance",
        "🎯 Feature Importance",
        "📈 Cluster Evaluation"
    ])
    
    # ========================================================================
    # TAB 1: DIMENSIONALITY REDUCTION VISUALIZATIONS
    # ========================================================================
    with tab1:
        st.markdown("## 📊 Dimensionality Reduction Visualizations")
        
        st.info("""
        **What is Dimensionality Reduction?**  
        We have 22+ features (hour, day, location, severity, etc). Humans can't visualize 
        22 dimensions. Dimensionality reduction simplifies this to 2D plots while preserving 
        patterns. Think of it as taking the best photo angle of a 3D sculpture.
        """)
        
        # Method selection
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("### ⚙️ Visualization Settings")
            
            method = st.radio(
                "Select Method",
                ["PCA", "t-SNE", "UMAP"],
                help="Different algorithms for dimensionality reduction"
            )
            
            color_by = st.selectbox(
                "Color Points By",
                ["Primary Type", "Crime_Severity", "KMeans_Cluster", 
                 "Time_of_Day", "Season", "Arrest"],
                help="Feature to color-code the visualization"
            )
            
            sample_size = st.slider(
                "Sample Size",
                1000, 50000, 10000, 1000,
                help="Number of points to display (more = slower)"
            )
            
            show_legend = st.checkbox("Show Legend", value=True)
        
        with col2:
            st.markdown(f"### 🎨 {method} Visualization")
            
            # Check if method columns exist
            if method == "PCA":
                x_col, y_col = 'PCA_1', 'PCA_2'
                method_desc = """
                **PCA (Principal Component Analysis)**
                - Finds directions of maximum variance
                - PC1 explains most patterns, PC2 second most
                - Linear transformation
                - Fast and interpretable
                """
            elif method == "t-SNE":
                x_col, y_col = 'tSNE_1', 'tSNE_2'
                method_desc = """
                **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
                - Creates beautiful cluster visualizations
                - Preserves local structure (nearby points stay nearby)
                - Non-linear transformation
                - Best for presentations
                """
            else:  # UMAP
                x_col, y_col = 'UMAP_1', 'UMAP_2'
                method_desc = """
                **UMAP (Uniform Manifold Approximation and Projection)**
                - Faster than t-SNE
                - Preserves both local and global structure
                - Good for large datasets
                - More reproducible results
                """
            
            if x_col in df.columns and y_col in df.columns:
                # Sample data
                df_plot = df.dropna(subset=[x_col, y_col]).sample(
                    n=min(sample_size, len(df)),
                    random_state=42
                )
                
                # Create visualization
                fig = px.scatter(
                    df_plot,
                    x=x_col,
                    y=y_col,
                    color=color_by,
                    title=f'{method} Visualization - Crime Patterns in 2D',
                    opacity=0.6,
                    height=600,
                    hover_data=['Date', 'Primary Type', 'Crime_Severity']
                )
                
                fig.update_layout(
                    plot_bgcolor='white',
                    xaxis=dict(showgrid=True, gridcolor='lightgray', title=f'{method} Component 1'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray', title=f'{method} Component 2'),
                    showlegend=show_legend
                )
                
                fig.update_traces(marker=dict(size=5, line=dict(width=0)))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Method description
                with st.expander(f"ℹ️ About {method}", expanded=False):
                    st.markdown(method_desc)
                
                # Statistics
                st.markdown("#### 📊 Visualization Statistics")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.metric("Points Displayed", f"{len(df_plot):,}")
                with col_b:
                    st.metric("Total Points", f"{len(df):,}")
                with col_c:
                    n_unique = df_plot[color_by].nunique()
                    st.metric(f"Unique {color_by}", n_unique)
                with col_d:
                    coverage = len(df_plot) / len(df) * 100
                    st.metric("Coverage", f"{coverage:.1f}%")
            else:
                st.warning(f"⚠️ {method} data not found. Please run dimensionality.py first.")
                st.code(f"""
# Run dimensionality reduction:
python src/dimensionality.py
                """)
        
        # Comparison of methods
        st.markdown("---")
        st.markdown("### 🔄 Method Comparison")
        
        comparison_data = {
            'Method': ['PCA', 't-SNE', 'UMAP'],
            'Speed': ['⚡⚡⚡ Fast', '⚡ Slow', '⚡⚡ Medium'],
            'Interpretability': ['High', 'Low', 'Medium'],
            'Preserves': ['Global Structure', 'Local Structure', 'Both'],
            'Best For': ['Understanding features', 'Beautiful plots', 'Large datasets'],
            'Reproducible': ['✓ Yes', '✗ No', '✓ Yes']
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # TAB 2: MODEL PERFORMANCE COMPARISON
    # ========================================================================
    with tab2:
        st.markdown("## 🏆 Clustering Model Performance")
        
        st.info("""
        **What are we comparing?**  
        We tested 3 clustering algorithms: K-Means, DBSCAN, and Hierarchical.
        Each has different strengths. We use metrics to objectively compare performance.
        """)
        
        # Check if cluster columns exist
        cluster_cols = ['KMeans_Cluster', 'DBSCAN_Cluster', 'Hierarchical_Cluster']
        available_methods = [col.replace('_Cluster', '') for col in cluster_cols if col in df.columns]
        
        if available_methods:
            # Calculate metrics for each method
            metrics_data = []
            
            for method in available_methods:
                cluster_col = f'{method}_Cluster'
                
                # Filter out noise for DBSCAN
                if method == 'DBSCAN':
                    df_eval = df[df[cluster_col] != -1]
                else:
                    df_eval = df
                
                # Prepare features for evaluation
                feature_cols = ['Latitude', 'Longitude']
                X = df_eval[feature_cols].values
                labels = df_eval[cluster_col].values
                
                # Calculate metrics
                try:
                    silhouette = silhouette_score(X, labels)
                    davies_bouldin = davies_bouldin_score(X, labels)
                except:
                    silhouette = 0
                    davies_bouldin = 0
                
                n_clusters = df[cluster_col].nunique()
                if method == 'DBSCAN' and -1 in df[cluster_col].values:
                    n_clusters -= 1
                    n_noise = (df[cluster_col] == -1).sum()
                else:
                    n_noise = 0
                
                metrics_data.append({
                    'Algorithm': method,
                    'Silhouette Score': silhouette,
                    'Davies-Bouldin': davies_bouldin,
                    'N Clusters': n_clusters,
                    'Noise Points': n_noise,
                    'Quality': '✓ Excellent' if silhouette > 0.7 else '✓ Good' if silhouette > 0.5 else '○ Fair'
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Display metrics comparison
            st.markdown("### 📊 Performance Metrics")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Metrics table
                st.dataframe(
                    metrics_df.style.format({
                        'Silhouette Score': '{:.4f}',
                        'Davies-Bouldin': '{:.4f}',
                        'N Clusters': '{:.0f}',
                        'Noise Points': '{:,.0f}'
                    }).background_gradient(subset=['Silhouette Score'], cmap='RdYlGn', vmin=0, vmax=1)
                    .background_gradient(subset=['Davies-Bouldin'], cmap='RdYlGn_r', vmin=0, vmax=2),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Metrics explanation