"""
PatrolIQ - Smart Safety Analytics Platform
MAIN ENTRY POINT - This is Home.py (not app.py)

To run: streamlit run streamlit_app/Home.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION - Must be first Streamlit command
# ============================================================================
st.set_page_config(
    page_title="PatrolIQ - Home",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar open by default
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E40AF;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        text-align: center;
        padding-bottom: 2rem;
    }
    .feature-card {
        background: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    #st.image("https://via.placeholder.com/150x50/1E40AF/FFFFFF?text=PatrolIQ", 
         #    use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## 📍 Navigation")
    st.info("""
    **Main Pages:**
    - 🏠 **Home** (Current)
    - 📊 **Data Overview**
    - 🗺️ **Crime Hotspots**
    - ⏰ **Temporal Patterns**
    - 🔍 **Model Analysis**
    
    Use the page selector above ☝️
    """)
    
    st.markdown("---")
    
    st.markdown("## ℹ️ About")
    st.markdown("""
    **PatrolIQ** analyzes 500,000 crime records 
    using machine learning to identify patterns 
    and optimize police deployment.
    
    **Technologies:**
    - Python 🐍
    - Streamlit 🎈
    - Scikit-learn 🤖
    - MLflow 📊
    - Plotly 📈
    """)
    
    st.markdown("---")
    
    # Quick stats in sidebar
    try:
        df = pd.read_parquet("data/processed/crime_data_final.parquet")

        st.markdown("## 📊 Quick Stats")
        st.metric("Total Crimes", f"{len(df):,}")
        st.metric("Date Range", f"{df['Date'].min()[:4]}-{df['Date'].max()[:4]}")
        st.metric("Crime Types", df['Primary Type'].nunique())
    except:
        st.warning("⚠️ Load data to see stats")
    
    st.markdown("---")
    
    st.markdown("### 🔗 Links")
    st.markdown("""
    - [📖 Documentation](#)
    - [💻 GitHub Repo](#)
    - [📧 Contact](#)
    """)
    
    st.markdown("---")
    st.caption("© 2024 PatrolIQ | v1.0.0")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<div class="main-header">🚔 PatrolIQ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Smart Safety Analytics Platform for Law Enforcement</div>', unsafe_allow_html=True)

# Welcome message
st.markdown("""
Welcome to **PatrolIQ** - an intelligent crime analysis system that helps law enforcement 
agencies make data-driven decisions. Explore crime patterns, identify hotspots, and optimize 
resource allocation using machine learning.
""")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("data/processed/crime_data_final.parquet")
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


df = load_data()

if df is not None:
    # ========================================================================
    # KEY METRICS ROW
    # ========================================================================
    st.markdown("## 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Crimes Analyzed",
            value=f"{len(df):,}",
            delta="500K sample"
        )
    
    with col2:
        arrest_rate = df['Arrest'].mean() * 100
        st.metric(
            label="Arrest Rate",
            value=f"{arrest_rate:.1f}%",
            delta=f"{arrest_rate - 30:.1f}% vs avg"
        )
    
    with col3:
        crime_types = df['Primary Type'].nunique()
        st.metric(
            label="Crime Categories",
            value=f"{crime_types}",
            delta="33 types"
        )
    
    with col4:
        if 'KMeans_Cluster' in df.columns:
            hotspots = df['KMeans_Cluster'].nunique()
            st.metric(
                label="Crime Hotspots",
                value=f"{hotspots}",
                delta="Identified"
            )
    
    st.markdown("---")
    
    # ========================================================================
    # MAIN CONTENT - TWO COLUMNS
    # ========================================================================
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("## 🎯 Platform Features")
        
        st.markdown("""
        <div class="feature-card">
            <h3>🗺️ Crime Hotspot Detection</h3>
            <p>Identify high-risk geographic zones using K-Means, DBSCAN, and Hierarchical clustering</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>⏰ Temporal Pattern Analysis</h3>
            <p>Discover when crimes occur most frequently - hourly, daily, and seasonal trends</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔍 Dimensionality Reduction</h3>
            <p>Visualize complex crime patterns in 2D using PCA, t-SNE, and UMAP</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📈 MLflow Integration</h3>
            <p>Track experiments, compare models, and ensure reproducible results</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("## 📈 Crime Distribution")
        
        # Top crime types
        top_crimes = df['Primary Type'].value_counts().head(10)
        
        fig = px.bar(
            x=top_crimes.values,
            y=top_crimes.index,
            orientation='h',
            title="Top 10 Crime Types",
            labels={'x': 'Number of Incidents', 'y': 'Crime Type'},
            color=top_crimes.values,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Crime trend over time
        st.markdown("### 📅 Crime Trend Over Time")
        
        df_trend = df.groupby(df['Date'].dt.to_period('M')).size().reset_index()
        df_trend.columns = ['Month', 'Count']
        df_trend['Month'] = df_trend['Month'].dt.to_timestamp()
        
        fig_trend = px.line(
            df_trend,
            x='Month',
            y='Count',
            title='Monthly Crime Incidents',
            labels={'Count': 'Number of Crimes', 'Month': 'Date'}
        )
        
        fig_trend.update_traces(line_color='#3B82F6', line_width=3)
        fig_trend.update_layout(
            plot_bgcolor='white',
            height=300
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("---")
    
    # ========================================================================
    # HOW TO USE
    # ========================================================================
    
    st.markdown("## 🚀 How to Use PatrolIQ")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Explore Data
        Navigate to **Data Overview** to:
        - View dataset statistics
        - Understand crime distributions
        - Check data quality metrics
        - Download reports
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ Analyze Patterns
        Use **Crime Hotspots** and **Temporal Patterns** to:
        - Identify high-risk zones
        - Study time-based patterns
        - Compare algorithms
        - Generate insights
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ Validate Models
        Check **Model Analysis** to:
        - Explore 2D visualizations
        - Compare performance metrics
        - Understand feature importance
        - Export results
        """)
    
    st.markdown("---")
    
    # ========================================================================
    # USE CASES
    # ========================================================================
    
    st.markdown("## 💼 Real-World Applications")
    
    use_cases = st.tabs([
        "🚔 Police Departments",
        "🏛️ City Administration",
        "🔍 Analytics Firms",
        "🚑 Emergency Response"
    ])
    
    with use_cases[0]:
        st.markdown("""
        ### Police Department Use Cases
        
        **Patrol Optimization**
        - Allocate officers to high-risk zones during peak crime hours
        - Reduce response time by 60% with data-driven deployment
        - Proactive crime prevention in identified hotspots
        
        **Resource Planning**
        - Evidence-based budget allocation for public safety
        - Identify areas requiring increased police presence
        - Track effectiveness of intervention strategies
        
        **Performance Metrics**
        - Monitor arrest rates across different districts
        - Evaluate crime reduction initiatives
        - Generate comprehensive crime analysis reports
        """)
    
    with use_cases[1]:
        st.markdown("""
        ### City Administration Applications
        
        **Urban Planning**
        - Design safer neighborhoods with data-driven insights
        - Strategic placement of surveillance systems
        - Optimize street lighting in high-crime areas
        
        **Budget Justification**
        - Provide concrete evidence for public safety funding
        - Track ROI of safety initiatives
        - Communicate effectively with stakeholders
        """)
    
    with use_cases[2]:
        st.markdown("""
        ### Law Enforcement Analytics Firms
        
        **Multi-Jurisdiction Services**
        - Provide crime intelligence to multiple cities
        - Benchmark safety performance across regions
        - Develop predictive policing solutions
        
        **Advanced Analytics**
        - Build custom models for client cities
        - Integrate with existing law enforcement systems
        - Deliver comprehensive crime analysis reports
        """)
    
    with use_cases[3]:
        st.markdown("""
        ### Emergency Response Systems
        
        **Risk Assessment**
        - Prioritize emergency calls based on area risk levels
        - Optimize ambulance and fire department placement
        - Coordinate multi-agency response in high-crime zones
        
        **Situational Awareness**
        - Real-time crime data for first responders
        - Historical pattern analysis for better preparedness
        - Integration with 911 dispatch systems
        """)
    
    st.markdown("---")
    
    # ========================================================================
    # GETTING STARTED
    # ========================================================================
    
    st.markdown("## 🎓 Getting Started")
    
    st.success("""
    **👉 Ready to explore?**
    
    Use the **sidebar navigation** (left side) to access different pages:
    
    1. **📊 Data Overview** - Start here to understand the dataset
    2. **🗺️ Crime Hotspots** - View geographic crime clusters
    3. **⏰ Temporal Patterns** - Analyze time-based crime trends
    4. **🔍 Model Analysis** - Validate ML models and visualizations
    """)
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Data Statistics", use_container_width=True):
            st.info("Navigate to 📊 Data Overview in the sidebar")
    
    with col2:
        if st.button("🗺️ Explore Hotspots", use_container_width=True):
            st.info("Navigate to 🗺️ Crime Hotspots in the sidebar")
    
    with col3:
        if st.button("⏰ See Time Patterns", use_container_width=True):
            st.info("Navigate to ⏰ Temporal Patterns in the sidebar")

else:
    # ========================================================================
    # ERROR STATE - NO DATA
    # ========================================================================
    
    st.error("""
    ### ⚠️ Data Not Found
    
    The processed data file is missing. Please run the data pipeline first.
    """)
    
    st.code("""
# Run these commands in order:
cd PatrolIQ
python src/data_preprocessing.py
python src/feature_engineering.py
python src/clustering.py
python src/dimensionality.py

# Then restart Streamlit:
streamlit run streamlit_app/Home.py
    """, language="bash")
    
    st.markdown("---")
    
    st.markdown("## 📚 Setup Instructions")
    
    with st.expander("🔧 Complete Setup Guide", expanded=True):
        st.markdown("""
        ### Step 1: Environment Setup
        ```bash
        python -m venv venv
        source venv/bin/activate  # Windows: venv\\Scripts\\activate
        pip install -r requirements.txt
        ```
        
        ### Step 2: Download Data
        Visit [Chicago Crime Dataset](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
        and download to `data/raw/chicago_crimes.csv`
        
        ### Step 3: Run Pipeline
        ```bash
        python src/data_preprocessing.py
        python src/feature_engineering.py
        python src/clustering.py
        python src/dimensionality.py
        ```
        
        ### Step 4: Launch App
        ```bash
        streamlit run streamlit_app/Home.py
        ```
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📚 Resources**
    - [Documentation](#)
    - [GitHub Repository](#)
    - [Tutorial Videos](#)
    """)

with col2:
    st.markdown("""
    **🔗 Connect**
    - [LinkedIn](#)
    - [Twitter](#)
    - [Email Support](#)
    """)

with col3:
    st.markdown("""
    **ℹ️ About**
    - Version: 1.0.0
    - Last Updated: 2024
    - License: MIT
    """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280;'>
    Made with ❤️ for Public Safety | Powered by Machine Learning
</div>
""", unsafe_allow_html=True)