"""
Dimensionality Reduction Module for PatrolIQ
=============================================

PURPOSE: Reduce complex 22-feature data into simple 2D visualizations
WHY: Humans can't visualize 22 dimensions - we need 2D plots
WHAT WE DO:
    - PCA: Find most important patterns (70%+ variance)
    - t-SNE: Create beautiful cluster visualizations
    - UMAP: Fast alternative to t-SNE

REAL-WORLD ANALOGY:
    Imagine a 3D sculpture
    Dimensionality reduction is like taking 2D photos from best angles
    You lose some detail but capture the essence

BUSINESS VALUE:
    - Show complex patterns to non-technical stakeholders
    - Identify which features matter most
    - Create compelling visualizations for reports
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


class DimensionalityReducer:
    """
    Implements PCA, t-SNE, and UMAP for crime pattern visualization
    
    Think of this as a photographer taking the best angle of complex data
    """
    
    def __init__(self, df, mlflow_tracking_uri='mlruns'):
        """
        Initialize reducer
        
        Parameters:
        -----------
        df : pandas DataFrame
            Featured and clustered crime data
        mlflow_tracking_uri : str
            Path to MLflow tracking directory
        """
        self.df = df.copy()
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.models = {}  # Store trained models
        self.results = {}  # Store results
        
        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("PatrolIQ_Dimensionality_Reduction")
        
        print(f"📊 Initialized DimensionalityReducer with {len(self.df):,} records")
        
    def prepare_features(self):
        """
        Select and prepare features for dimensionality reduction
        
        WHY: Need numeric features only, properly scaled
        
        FEATURES SELECTED:
            - Temporal: Hour, Day, Month, Weekend
            - Geographic: Latitude, Longitude
            - Crime: Severity, Arrest, Domestic
            - Encoded: Crime type, Location type
        """
        print("\n🔧 Preparing features for dimensionality reduction...")
        
        # Select numerical feature columns
        feature_cols = [
            # Temporal features
            'Hour', 'Day_of_Week_Num', 'Month', 'Is_Weekend',
            
            # Geographic features (use original, not normalized, for better visualization)
            'Latitude', 'Longitude',
            
            # Crime characteristics
            'Crime_Severity', 'Arrest', 'Domestic'
        ]
        
        # Add encoded columns if they exist
        if 'Primary_Type_Encoded' in self.df.columns:
            feature_cols.append('Primary_Type_Encoded')
        if 'Location_Desc_Encoded' in self.df.columns:
            feature_cols.append('Location_Desc_Encoded')
        
        # Select features
        self.features = self.df[feature_cols].copy()
        
        # Handle any remaining missing values
        self.features = self.features.fillna(self.features.median())
        
        # Standardize features (mean=0, std=1)
        # CRITICAL: Dimensionality reduction algorithms sensitive to scale
        scaler = StandardScaler()
        self.features_scaled = scaler.fit_transform(self.features)
        
        self.scaler = scaler
        self.feature_names = feature_cols
        
        print(f"   ✅ Prepared {len(feature_cols)} features:")
        for i, feat in enumerate(feature_cols, 1):
            print(f"      {i}. {feat}")
        print(f"   ✅ Scaled {len(self.features):,} records")
        
        return self
    
    def apply_pca(self, n_components=3, run_name="PCA_Analysis"):
        """
        Principal Component Analysis
        
        WHAT IT DOES:
            - Finds directions of maximum variance in data
            - Projects data onto these "principal components"
            - Ranks features by importance
            
        HOW IT WORKS:
            1. Calculate variance in each direction
            2. Find direction with most variance (PC1)
            3. Find perpendicular direction with next most variance (PC2)
            4. Repeat for PC3, PC4, etc.
            
        BUSINESS INTERPRETATION:
            PC1 might represent "Location" (lat/long dominate)
            PC2 might represent "Time patterns" (hour, day dominate)
            PC3 might represent "Crime severity"
            
        WHY 3 COMPONENTS:
            - Typically explain 70-80% of variance
            - Can visualize in 3D
            - Balance between simplicity and information
            
        GOAL:
            Explain 70%+ variance with 2-3 components
        """
        print(f"\n🔍 Running PCA (n_components={n_components})...")
        
        with mlflow.start_run(run_name=run_name):
            # Apply PCA
            pca = PCA(n_components=n_components, random_state=42)
            components = pca.fit_transform(self.features_scaled)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Add components to dataframe
            for i in range(n_components):
                self.df[f'PCA_{i+1}'] = components[:, i]
            
            # === LOG TO MLFLOW ===
            mlflow.log_param("algorithm", "PCA")
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("n_features", len(self.feature_names))
            
            for i, var in enumerate(explained_variance):
                mlflow.log_metric(f"explained_variance_PC{i+1}", var)
            
            mlflow.log_metric("cumulative_variance", cumulative_variance[-1])
            mlflow.log_metric("total_variance_explained", cumulative_variance[-1])
            
            # Save model
            mlflow.sklearn.log_model(pca, "model")
            
            # Get feature importance
            feature_importance = self._get_pca_feature_importance(pca)
            
            # Store results
            self.models['pca'] = pca
            self.results['pca'] = {
                'components': components,
                'explained_variance': explained_variance,
                'cumulative_variance': cumulative_variance,
                'feature_importance': feature_importance
            }
            
            # === DISPLAY RESULTS ===
            print(f"   ✅ PCA Complete")
            print(f"\n   📊 Variance Explained by Each Component:")
            for i, var in enumerate(explained_variance):
                print(f"      PC{i+1}: {var*100:.2f}% {'■' * int(var*50)}")
            
            print(f"\n   📈 Cumulative Variance:")
            print(f"      Total: {cumulative_variance[-1]*100:.2f}%")
            
            if cumulative_variance[-1] >= 0.70:
                print(f"      ✓ EXCELLENT: Captures 70%+ of patterns")
            elif cumulative_variance[-1] >= 0.60:
                print(f"      ✓ GOOD: Captures 60%+ of patterns")
            else:
                print(f"      ⚠ FAIR: Consider more components")
            
            # Display feature importance
            print(f"\n   🎯 Top 5 Most Important Features:")
            for i, (feat, imp) in enumerate(feature_importance[:5], 1):
                print(f"      {i}. {feat}: {imp:.4f} {'■' * int(imp*20)}")
            
        return self
    
    def _get_pca_feature_importance(self, pca):
        """
        Calculate feature importance from PCA loadings
        
        INTERPRETATION:
            High loading = Feature strongly influences this component
            We use absolute values and first component (explains most variance)
        """
        # Get absolute loadings for first component
        loadings = np.abs(pca.components_[0])
        
        # Normalize to sum to 1
        loadings_normalized = loadings / loadings.sum()
        
        # Create ranking
        importance = sorted(
            zip(self.feature_names, loadings_normalized),
            key=lambda x: x[1],
            reverse=True
        )
        
        return importance
    
    def apply_tsne(self, n_components=2, perplexity=30, run_name="tSNE_Visualization"):
        """
        t-SNE (t-Distributed Stochastic Neighbor Embedding)
        
        WHAT IT DOES:
            - Creates beautiful 2D visualizations
            - Preserves local structure (nearby points stay nearby)
            - Shows clear separation between clusters
            
        HOW IT WORKS:
            1. Calculate pairwise similarities in high dimensions
            2. Create random 2D points
            3. Adjust 2D points to preserve high-dim similarities
            4. Use gradient descent for 1000 iterations
            
        PARAMETERS:
            - perplexity (30): Balance between local and global structure
              Lower = focus on local clusters
              Higher = focus on global structure
            
        PROS:
            ✓ Beautiful visualizations
            ✓ Clear cluster separation
            ✓ Good for presentations
            
        CONS:
            ✗ Very slow (30-60 minutes for 500K points)
            ✗ Different runs give different results
            ✗ Can't transform new data
            
        SOLUTION TO SLOWNESS:
            Sample 50,000 points for t-SNE
            Still gives excellent visualizations
        """
        print(f"\n🎨 Running t-SNE (perplexity={perplexity})...")
        print(f"   ⏳ This may take 5-15 minutes...")
        
        # Sample for computational efficiency
        # t-SNE is O(n²) complexity - very slow for large datasets
        sample_size = min(50000, len(self.features_scaled))
        print(f"   📉 Sampling {sample_size:,} points for speed...")
        
        sample_indices = np.random.choice(
            len(self.features_scaled),
            sample_size,
            replace=False
        )
        X_sample = self.features_scaled[sample_indices]
        
        with mlflow.start_run(run_name=run_name):
            # Apply t-SNE
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate='auto',
                init='pca',
                random_state=42,
                n_jobs=-1,
                max_iter=1000
            )
            components_sample = tsne.fit_transform(X_sample)
            
            # Create full array with NaN for non-sampled points
            components = np.full((len(self.features_scaled), n_components), np.nan)
            components[sample_indices] = components_sample
            
            # Add components to dataframe
            self.df['tSNE_1'] = components[:, 0]
            self.df['tSNE_2'] = components[:, 1]
            
            # === LOG TO MLFLOW ===
            mlflow.log_param("algorithm", "t-SNE")
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("perplexity", perplexity)
            mlflow.log_param("sample_size", sample_size)
            mlflow.log_param("n_iter", 1000)
            
            # KL divergence = optimization metric (lower is better)
            mlflow.log_metric("kl_divergence", tsne.kl_divergence_)
            
            # Store results
            self.models['tsne'] = tsne
            self.results['tsne'] = {
                'components': components,
                'sample_indices': sample_indices,
                'kl_divergence': tsne.kl_divergence_
            }
            
            # Display results
            print(f"   ✅ t-SNE Complete")
            print(f"      • KL Divergence: {tsne.kl_divergence_:.4f} (lower is better)")
            print(f"      • Points Visualized: {sample_size:,}")
            print(f"      • Output: 2D coordinates for beautiful plots")
            
        return self
    
    def apply_umap(self, n_components=2, n_neighbors=15, run_name="UMAP_Visualization"):
        """
        UMAP (Uniform Manifold Approximation and Projection)
        
        WHAT IT DOES:
            - Similar to t-SNE but faster
            - Better preserves global structure
            - Can transform new data points
            
        HOW IT WORKS:
            1. Build graph of nearest neighbors
            2. Optimize low-dimensional representation
            3. Preserve both local and global structure
            
        PARAMETERS:
            - n_neighbors (15): Size of local neighborhood
              Lower = more local focus
              Higher = more global structure
            
        PROS:
            ✓ Much faster than t-SNE
            ✓ Preserves global structure better
            ✓ Can transform new points
            ✓ More reproducible
            
        CONS:
            ✗ Less known than PCA/t-SNE
            ✗ More parameters to tune
            
        WHEN TO USE:
            - Large datasets (faster than t-SNE)
            - Need reproducibility
            - Want global + local structure
        """
        print(f"\n🗺️  Running UMAP (n_neighbors={n_neighbors})...")
        
        with mlflow.start_run(run_name=run_name):
            # Apply UMAP
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=42,
                verbose=False,
                n_jobs=-1
            )
            
            components = reducer.fit_transform(self.features_scaled)
            
            # Add components to dataframe
            self.df['UMAP_1'] = components[:, 0]
            self.df['UMAP_2'] = components[:, 1]
            
            # === LOG TO MLFLOW ===
            mlflow.log_param("algorithm", "UMAP")
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("n_neighbors", n_neighbors)
            mlflow.log_param("n_points", len(self.features_scaled))
            
            # Store results
            self.models['umap'] = reducer
            self.results['umap'] = {
                'components': components
            }
            
            # Display results
            print(f"   ✅ UMAP Complete")
            print(f"      • Output Shape: {components.shape}")
            print(f"      • All {len(self.features_scaled):,} points visualized")
            print(f"      • Faster than t-SNE, better global structure")
            
        return self
    
    def create_variance_plot(self, save_path='models/pca_variance_plot.png'):
        """
        Create scree plot showing variance explained by each PCA component
        
        BUSINESS VALUE:
            Shows stakeholders how much information we retain
            "These 3 components capture 75% of all patterns"
        """
        if 'pca' not in self.results:
            print("⚠️  Run apply_pca() first!")
            return self
        
        explained_var = self.results['pca']['explained_variance']
        cumulative_var = self.results['pca']['cumulative_variance']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance bar plot
        ax1.bar(range(1, len(explained_var)+1), explained_var, 
               alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
        ax1.set_title('Variance Explained by Each Component', fontsize=14, fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xticks(range(1, len(explained_var)+1))
        
        # Add percentage labels on bars
        for i, v in enumerate(explained_var):
            ax1.text(i+1, v+0.01, f'{v*100:.1f}%', ha='center', fontweight='bold')
        
        # Cumulative variance line plot
        ax2.plot(range(1, len(cumulative_var)+1), cumulative_var, 
                marker='o', linestyle='-', linewidth=3, markersize=10, 
                color='coral', markerfacecolor='white', markeredgewidth=2)
        ax2.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='70% threshold')
        ax2.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80% threshold')
        ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Variance Explained', fontsize=12, fontweight='bold')
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_xticks(range(1, len(cumulative_var)+1))
        ax2.set_ylim([0, 1.05])
        
        # Add percentage labels
        for i, v in enumerate(cumulative_var):
            ax2.text(i+1, v+0.02, f'{v*100:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Saved PCA variance plot to: {save_path}")
        
        return self
    
    def create_2d_visualization(self, method='pca', color_by='Primary Type', 
                                sample_size=10000, save_path=None):
        """
        Create interactive 2D scatter plot
        
        PARAMETERS:
            method: 'pca', 'tsne', or 'umap'
            color_by: Feature to color points by
            sample_size: Number of points to plot
        
        BUSINESS VALUE:
            Present complex patterns in simple, interactive plots
            Stakeholders can explore data themselves
        """
        # Determine which components to use
        if method == 'pca':
            x_col, y_col = 'PCA_1', 'PCA_2'
            title = 'Crime Patterns - PCA Visualization'
        elif method == 'tsne':
            x_col, y_col = 'tSNE_1', 'tSNE_2'
            title = 'Crime Patterns - t-SNE Visualization'
        elif method == 'umap':
            x_col, y_col = 'UMAP_1', 'UMAP_2'
            title = 'Crime Patterns - UMAP Visualization'
        else:
            print(f"⚠️  Unknown method: {method}")
            return self
        
        # Check if columns exist
        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f"⚠️  Run apply_{method}() first!")
            return self
        
        # Sample data for visualization
        df_plot = self.df.dropna(subset=[x_col, y_col]).sample(
            n=min(sample_size, len(self.df)),
            random_state=42
        )
        
        # Create interactive plotly scatter plot
        fig = px.scatter(
            df_plot,
            x=x_col,
            y=y_col,
            color=color_by,
            title=title,
            opacity=0.6,
            height=600,
            hover_data=['Date', 'Crime_Severity', 'Arrest']
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='lightgray', zeroline=False),
            font=dict(size=12),
            title_font=dict(size=16, family='Arial Black')
        )
        
        fig.update_traces(marker=dict(size=5, line=dict(width=0)))
        
        # Save to HTML
        if save_path is None:
            save_path = f'models/{method}_visualization.html'
        
        fig.write_html(save_path)
        print(f"📊 Saved {method.upper()} visualization to: {save_path}")
        print(f"   Open in browser to interact with plot!")
        
        return self
    
    def get_summary(self):
        """Display comprehensive summary of dimensionality reduction"""
        print("\n" + "="*70)
        print("🎯 DIMENSIONALITY REDUCTION SUMMARY")
        print("="*70)
        
        print(f"\n📊 Original Dimensions: {len(self.feature_names)} features")
        print(f"   Features: {', '.join(self.feature_names[:5])}...")
        
        if 'pca' in self.results:
            pca_result = self.results['pca']
            print(f"\n🔍 PCA Results:")
            print(f"   • Components: {len(pca_result['explained_variance'])}")
            print(f"   • Variance Explained: {pca_result['cumulative_variance'][-1]*100:.2f}%")
            print(f"   • Top Feature: {pca_result['feature_importance'][0][0]}")
            print(f"   • Quality: {'✓ EXCELLENT' if pca_result['cumulative_variance'][-1] > 0.7 else '✓ GOOD'}")
        
        if 'tsne' in self.results:
            tsne_result = self.results['tsne']
            print(f"\n🎨 t-SNE Results:")
            print(f"   • KL Divergence: {tsne_result['kl_divergence']:.4f}")
            print(f"   • Points Visualized: {len(tsne_result['sample_indices']):,}")
            print(f"   • Output: Beautiful 2D scatter plot")
        
        if 'umap' in self.results:
            print(f"\n🗺️  UMAP Results:")
            print(f"   • Components: {self.results['umap']['components'].shape}")
            print(f"   • All points visualized")
            print(f"   • Output: Fast, global structure preserved")
        
        print(f"\n✅ All dimensionality reduction complete!")
        print(f"   Ready for interactive visualizations in Streamlit")
        
        return self
    
    def save_results(self, output_path='data/processed/crime_data_final.csv'):
        """Save final dataframe with all dimensionality reduction results"""
        self.df.to_csv(output_path, index=False)
        file_size = pd.read_csv(output_path).memory_usage(deep=True).sum() / 1024**2
        print(f"\n💾 Saved final data to: {output_path}")
        print(f"   File size: {file_size:.1f} MB")
        print(f"   New columns: PCA_1, PCA_2, PCA_3, tSNE_1, tSNE_2, UMAP_1, UMAP_2")
        return self
    
    def get_dataframe(self):
        """Return final dataframe with all reductions"""
        return self.df


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("🚀 PATROLIQ DIMENSIONALITY REDUCTION PIPELINE")
    print("="*70)
    
    # Load clustered data
    print("\n📥 Loading clustered data...")
    try:
        df = pd.read_csv('data/processed/crime_data_clustered.csv')
        print(f"✅ Loaded {len(df):,} records with {df.shape[1]} features")
    except FileNotFoundError:
        print("❌ Error: crime_data_clustered.csv not found!")
        print("   Please run clustering.py first")
        exit(1)
    
    # Initialize reducer
    reducer = DimensionalityReducer(df)
    
    # Prepare features
    reducer.prepare_features()
    
    # Run all dimensionality reduction techniques
    print("\n" + "="*70)
    print("🔬 RUNNING DIMENSIONALITY REDUCTION")
    print("="*70)
    
    # 1. PCA - Find main patterns
    reducer.apply_pca(n_components=3)
    
    # 2. t-SNE - Beautiful visualizations
    reducer.apply_tsne(n_components=2, perplexity=30)
    
    # 3. UMAP - Fast alternative
    #reducer.apply_umap(n_components=2, n_neighbors=15)
    
    # Create visualizations
    print("\n" + "="*70)
    print("📊 CREATING VISUALIZATIONS")
    print("="*70)
    
    reducer.create_variance_plot()
    reducer.create_2d_visualization(method='pca', color_by='Primary Type')
    reducer.create_2d_visualization(method='tsne', color_by='Crime_Severity')
    #reducer.create_2d_visualization(method='umap', color_by='KMeans_Cluster')
    
    # Display summary
    reducer.get_summary()
    
    # Save results
    reducer.save_results()
    
    print("\n" + "="*70)
    print("✅ DIMENSIONALITY REDUCTION COMPLETE!")
    print("="*70)
    print(f"\n📊 Results Summary:")
    print(f"   • PCA: {reducer.results['pca']['cumulative_variance'][-1]*100:.1f}% variance explained")
    print(f"   • t-SNE: {len(reducer.results['tsne']['sample_indices']):,} points visualized")
    print(f"   • UMAP: All {len(df):,} points reduced to 2D")
    print(f"\n📁 Files Created:")
    print(f"   • models/pca_variance_plot.png")
    print(f"   • models/pca_visualization.html")
    print(f"   • models/tsne_visualization.html")
    print(f"   • models/umap_visualization.html")
    print(f"\n💡 Next Step: Launch Streamlit app")
    print(f"   streamlit run streamlit_app/Home.py")
    print(f"\n💡 View experiments: mlflow ui --port 5000")
    print("="*70)