import os
import mlflow
import time
import math
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')


class CrimeClusterer:
    def __init__(self, df, mlflow_tracking_uri='file:///C:/Users/vikaa/Downloads/PatrolIQ-main (1)/PatrolIQ-main/mlruns', sample_size=5000):
        self.df = df.copy()
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.models = {}
        self.results = {}
        self.sample_size = 2000

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("PatrolIQ_Clustering")

        print(f"📊 Initialized CrimeClusterer with {len(self.df):,} records")
        print(f"📁 MLflow tracking: {mlflow_tracking_uri}")
        print(f"🔢 Sampling size (for metrics / sampled algorithms): {self.sample_size:,}")

    def prepare_geographic_features(self):
        print("\n📍 Preparing geographic features...")
        self.geo_features = self.df[['Latitude', 'Longitude']].copy()
        scaler = StandardScaler()
        self.geo_features_scaled = scaler.fit_transform(self.geo_features)
        self.geo_scaler = scaler
        print(f"   ✅ Prepared {self.geo_features_scaled.shape[0]:,} geographic points (scaled)")
        return self

    def _get_sample_indices(self, random_state=42):
        n = len(self.geo_features_scaled)
        s = min(self.sample_size, n)
        rng = np.random.RandomState(random_state)
        return rng.choice(n, size=s, replace=False)

    def _safe_silhouette(self, X, labels, sample_for_metric=True):
        """Compute silhouette on a sample if requested (safe for big data)."""
        try:
            if sample_for_metric and len(X) > self.sample_size:
                idx = self._get_sample_indices()
                s = silhouette_score(X[idx], labels[idx])
                return float(s)
            else:
                return float(silhouette_score(X, labels))
        except Exception as e:
            print(f"   ⚠ Warning: silhouette computation failed or too heavy: {e}")
            return float('nan')

    def kmeans_clustering(self, n_clusters=8, run_name="KMeans_Geographic"):
        print(f"\n🎯 Running K-Means Clustering (k={n_clusters}) on FULL dataset...")
        start = time.time()
        mlflow.end_run()
  # ensure no active run
        with mlflow.start_run(run_name=run_name):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300, verbose=0)
            clusters = kmeans.fit_predict(self.geo_features_scaled)

            # compute metrics safely (silhouette on sample)
            silhouette = self._safe_silhouette(self.geo_features_scaled, clusters, sample_for_metric=True)
            davies_bouldin = float('nan')
            calinski = float('nan')
            try:
                # Use sample for DB and Calinski as well (they are also pairwise-ish heavy)
                idx = self._get_sample_indices()
                davies_bouldin = float(davies_bouldin_score(self.geo_features_scaled[idx], clusters[idx]))
                calinski = float(calinski_harabasz_score(self.geo_features_scaled[idx], clusters[idx]))
            except Exception as e:
                print(f"   ⚠ Warning: DB/Calinski computation failed on sample: {e}")

            self.df['KMeans_Cluster'] = clusters

            mlflow.log_param("algorithm", "KMeans")
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.log_metric("silhouette_score_sampled", float(silhouette) if not math.isnan(silhouette) else -1.0)
            mlflow.log_metric("davies_bouldin_score_sampled", float(davies_bouldin) if not math.isnan(davies_bouldin) else -1.0)
            mlflow.log_metric("calinski_harabasz_score_sampled", float(calinski) if not math.isnan(calinski) else -1.0)
            mlflow.log_metric("inertia", float(kmeans.inertia_))

            self.models['kmeans'] = kmeans
            self.results['kmeans'] = {
                'silhouette': float(silhouette) if not math.isnan(silhouette) else -1.0,
                'davies_bouldin': float(davies_bouldin) if not math.isnan(davies_bouldin) else np.inf,
                'calinski': float(calinski) if not math.isnan(calinski) else -1.0,
                'n_clusters': n_clusters
            }

            print(f"   ✅ K-Means completed in {time.time() - start:.1f}s")
            print(f"      • Silhouette (sample): {silhouette:.4f}" if not math.isnan(silhouette) else "      • Silhouette: N/A")
            print(f"      • Inertia: {kmeans.inertia_:.2f}")

            cluster_sizes = pd.Series(clusters).value_counts().sort_index()
            print("      • Cluster sizes (top):")
            for cluster_id, size in cluster_sizes.head(10).items():
                print(f"         - Cluster {cluster_id}: {size:,} ({size/len(clusters)*100:.1f}%)")

        return self

    def dbscan_clustering(self, eps=0.01, min_samples=50, run_name="DBSCAN_Geographic"):

        print(f"\n🔍 Running DBSCAN on a sample (eps={eps}, min_samples={min_samples})...")
        start = time.time()
        mlflow.end_run()

        with mlflow.start_run(run_name=run_name):
            # sample indices & sample data
            sample_idx = self._get_sample_indices()
            X_sample = self.geo_features_scaled[sample_idx]

            print(f"   • Using sample of {len(X_sample):,} points for DBSCAN")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            labels_sample = dbscan.fit_predict(X_sample)

            # sample metrics (only if cluster count > 1)
            mask = labels_sample != -1
            if mask.sum() > 1 and len(set(labels_sample[mask])) > 0:
                try:
                    silhouette_s = float(silhouette_score(X_sample[mask], labels_sample[mask]))
                    davies_s = float(davies_bouldin_score(X_sample[mask], labels_sample[mask]))
                except Exception as e:
                    print(f"   ⚠ Warning: sample silhouette/db computation failed: {e}")
                    silhouette_s, davies_s = float('nan'), float('inf')
            else:
                silhouette_s, davies_s = float('nan'), float('inf')

            # compute cluster centers from sample clusters (exclude noise)
            clusters_found = sorted(set(labels_sample) - {-1})
            if clusters_found:
                centers = []
                for c in clusters_found:
                    centers.append(X_sample[labels_sample == c].mean(axis=0))
                centers = np.vstack(centers)
            else:
                centers = np.empty((0, self.geo_features_scaled.shape[1]))

            # Map sample cluster labels to full dataset
            full_labels = np.full(shape=(len(self.geo_features_scaled),), fill_value=-1, dtype=int)
            if centers.shape[0] > 0:
                nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(centers)
                dists, idxs = nbrs.kneighbors(self.geo_features_scaled)
                dists = dists.ravel()
                idxs = idxs.ravel()

                # threshold distance to decide if assign to nearest center or leave as noise.
                # We use eps (in scaled space) multiplied by a factor (tunable).
                assign_threshold = eps * 2.0
                assign_mask = dists <= assign_threshold
                # assign cluster labels (map nearest center index -> original DBSCAN cluster label)
                mapped_labels = np.array(clusters_found, dtype=int)[idxs]
                full_labels[assign_mask] = mapped_labels[assign_mask]
            else:
                print("   • No DBSCAN clusters found on sample -> marking all points as noise (-1)")

            # Write to dataframe
            self.df['DBSCAN_Cluster'] = full_labels

            n_clusters = len(clusters_found)
            n_noise = int((full_labels == -1).sum())
            noise_pct = n_noise / len(full_labels) * 100.0

            mlflow.log_param("algorithm", "DBSCAN_sampled_mapped")
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", min_samples)
            mlflow.log_metric("n_clusters_sampled", int(n_clusters))
            mlflow.log_metric("n_noise_points_mapped", int(n_noise))
            mlflow.log_metric("noise_percentage_mapped", float(noise_pct))
            if not math.isnan(silhouette_s):
                mlflow.log_metric("silhouette_sampled", float(silhouette_s))
                mlflow.log_metric("davies_bouldin_sampled", float(davies_s))

            self.models['dbscan'] = dbscan
            self.results['dbscan'] = {
                'silhouette': float(silhouette_s) if not math.isnan(silhouette_s) else -1.0,
                'davies_bouldin': float(davies_s) if math.isfinite(davies_s) else np.inf,
                'n_clusters': int(n_clusters),
                'n_noise': int(n_noise),
                'noise_pct': float(noise_pct)
            }

            print(f"   ✅ DBSCAN (sampled + mapped) completed in {time.time() - start:.1f}s")
            print(f"      • Sample clusters found: {n_clusters}")
            print(f"      • Mapped noise (approx): {n_noise:,} ({noise_pct:.1f}%)")
            if not math.isnan(silhouette_s):
                print(f"      • Silhouette (sample): {silhouette_s:.4f}")

        return self

    def hierarchical_clustering(self, n_clusters=8, run_name="Hierarchical_Geographic"):
        print("\n🌳 Running Hierarchical Clustering on a sample...")
        mlflow.end_run()


        total_rows = len(self.geo_features_scaled)
        sample_size = min(1000, total_rows)
        if sample_size < 100:
            print(f"❌ ERROR: Only {sample_size} rows available, not enough for hierarchical clustering.")
            return None

        sample_idx = self._get_sample_indices()
        X_sample = self.geo_features_scaled[sample_idx]

        try:
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            clusters = model.fit_predict(X_sample)
        except Exception as e:
            print(f"❌ ERROR during hierarchical clustering: {e}")
            return None

        # metrics on sample (safe)
        sil = self._safe_silhouette(X_sample, clusters, sample_for_metric=False)
        try:
            db = float(davies_bouldin_score(X_sample, clusters))
            ch = float(calinski_harabasz_score(X_sample, clusters))
        except Exception:
            db, ch = float('inf'), float('nan')

        self.df['Hierarchical_Cluster'] = -1  # sampled only
        self.models['hierarchical'] = model
        self.results['hierarchical'] = {
            'silhouette': float(sil) if not math.isnan(sil) else -1.0,
            'davies_bouldin': float(db) if math.isfinite(db) else np.inf,
            'calinski_harabasz': float(ch) if not math.isnan(ch) else -1.0,
            'n_clusters': n_clusters
        }

        mlflow.log_param("hierarchical_n_clusters", n_clusters)
        mlflow.log_metric("hierarchical_silhouette_sampled", float(sil) if not math.isnan(sil) else -1.0)

        print("   ✅ Hierarchical (sampled) complete")
        print(f"      • Silhouette (sample): {sil:.4f}" if not math.isnan(sil) else "      • Silhouette: N/A")
        return clusters

    def temporal_clustering(self, n_clusters=5, run_name="KMeans_Temporal"):
        print("\n⏰ Running Temporal Clustering (full temporal features)...")
        mlflow.end_run()

        temporal_cols = ['Hour', 'Day_of_Week_Num', 'Month', 'Is_Weekend']
        temporal_features = self.df[temporal_cols].copy()
        scaler = StandardScaler()
        temporal_scaled = scaler.fit_transform(temporal_features)

        with mlflow.start_run(run_name=run_name):
            kmeans_temporal = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans_temporal.fit_predict(temporal_scaled)
            silhouette = self._safe_silhouette(temporal_scaled, clusters, sample_for_metric=True)
            davies = float('nan')
            try:
                # compute db on sample if large
                idx = self._get_sample_indices()
                davies = float(davies_bouldin_score(temporal_scaled[idx], clusters[idx]))
            except Exception:
                davies = float('inf')

            self.df['Temporal_Cluster'] = clusters
            mlflow.log_param("algorithm", "KMeans_Temporal")
            mlflow.log_metric("temporal_silhouette_sampled", float(silhouette) if not math.isnan(silhouette) else -1.0)
            self.models['temporal'] = kmeans_temporal
            self.results['temporal'] = {
                'silhouette': float(silhouette) if not math.isnan(silhouette) else -1.0,
                'davies_bouldin': float(davies) if math.isfinite(davies) else np.inf,
                'n_clusters': n_clusters
            }
            print(f"   ✅ Temporal KMeans complete (k={n_clusters}) — Silhouette (sample): {silhouette:.4f}" if not math.isnan(silhouette) else "   ✅ Temporal KMeans complete")

        return self

    def compare_models(self):
        print("\n" + "="*70)
        print("📊 CLUSTERING MODEL COMPARISON (sample-based metrics shown)")
        print("="*70)

        comparison = []
        for name, result in self.results.items():
            if name != 'temporal':
                comparison.append({
                    'Algorithm': name.upper(),
                    'Silhouette': f"{result['silhouette']:.4f}" if result['silhouette'] != -1.0 else "N/A",
                    'Davies-Bouldin': f"{result.get('davies_bouldin', 'N/A')}",
                    'N_Clusters': result.get('n_clusters', 'N/A')
                })

        if comparison:
            print(pd.DataFrame(comparison).to_string(index=False))
        else:
            print("   No geographic model results to compare yet.")

        candidates = [(k, v) for k, v in self.results.items() if k != 'temporal']
        if not candidates:
            print("   ⚠ No candidate models available to recommend.")
            return None

        # choose best by silhouette (handle N/A)
        def _score(item):
            val = item[1].get('silhouette', -1.0)
            return val if val is not None else -1.0

        best_model = max(candidates, key=_score)
        print("\n" + "="*70)
        print(f"🏆 RECOMMENDED MODEL: {best_model[0].upper()}")
        print("="*70)
        print(f"   Reason: Highest silhouette score (sample) = {best_model[1]['silhouette']:.4f}" if best_model[1]['silhouette'] != -1.0 else "   Reason: Best available metric")
        return best_model[0]

    def save_results(self, output_path='data/processed/crime_data_clustered.csv'):
        self.df.to_csv(output_path, index=False)
        file_size_mb = pd.read_csv(output_path).memory_usage(deep=True).sum() / 1024**2
        print(f"\n💾 Saved clustered data to: {output_path} ({file_size_mb:.1f} MB)")
        return self

    def get_cluster_statistics(self, cluster_column='KMeans_Cluster', top_n_clusters=10):
      print(f"\n📊 Detailed Statistics for {cluster_column}")
      print("=" * 70)

      # Check column exists
      if cluster_column not in self.df.columns:
        print(f"❌ Column '{cluster_column}' not found.")
        return

      # Unique cluster IDs sorted
      unique_clusters = sorted(self.df[cluster_column].unique())

      # Safety limit
      if len(unique_clusters) > top_n_clusters:
        print(f"⚠ {len(unique_clusters)} total clusters. Showing first {top_n_clusters}.")
        unique_clusters = unique_clusters[:top_n_clusters]

      # Loop each cluster
      for cluster_id in unique_clusters:
        cluster_data = self.df[self.df[cluster_column] == cluster_id]

        label = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"
        print(f"\n{label}: {len(cluster_data):,} rows")

        if cluster_data.empty:
            continue

        # Statistics
        avg_sev = cluster_data['Crime_Severity'].mean()
        arrest_rate = cluster_data['Arrest'].mean() * 100

        print(f"   • Avg Severity: {avg_sev:.2f}")
        print(f"   • Arrest Rate: {arrest_rate:.1f}%")
        print(f"   • Top crimes (3):")

        # Top 3 crime types
        top_crimes = cluster_data['Primary Type'].value_counts().head(3)

        for crime, count in top_crimes.items():
            pct = (count / len(cluster_data)) * 100
            print(f"      - {crime}: {count:,} ({pct:.1f}%)")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("="*70)
    print("🚀 PATROLIQ CLUSTERING PIPELINE ")
    print("="*70)

    try:
        df = pd.read_csv('data/processed/crime_data_features.csv')
        print(f"✅ Loaded {len(df):,} records")
    except FileNotFoundError:
        print("❌ Error: data/processed/crime_data_features.csv not found. Run feature_engineering.py first.")
        raise SystemExit(1)

    clusterer = CrimeClusterer(df, sample_size=50000)
    clusterer.prepare_geographic_features()

    print("\n🔬 RUNNING CLUSTERING ALGORITHMS")
    # KMeans on full (silhouette sampled)
    #clusterer.kmeans_clustering(n_clusters=8)
    # DBSCAN sampled -> mapped to full
    clusterer.dbscan_clustering(eps=0.01, min_samples=50)
    # Hierarchical (sampled)
    clusterer.hierarchical_clustering(n_clusters=8)
    # Temporal on full
    clusterer.temporal_clustering(n_clusters=5)

    best_model = clusterer.compare_models()

    # map best model to column
    mapping = {
        'kmeans': 'KMeans_Cluster',
        'dbscan': 'DBSCAN_Cluster',
        'hierarchical': 'Hierarchical_Cluster',
        'temporal': 'Temporal_Cluster'
    }
    cluster_col = mapping.get(best_model, None)
    if cluster_col:
        clusterer.get_cluster_statistics(cluster_column=cluster_col, top_n_clusters=20)
    else:
        print("⚠ Best model has no column to inspect.")

    clusterer.save_results()
    print("\n✅ ALL DONE")
