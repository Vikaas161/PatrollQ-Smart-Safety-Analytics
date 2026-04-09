"""
Feature Engineering Module for PatrolIQ
========================================

PURPOSE: Transform raw crime data into ML-ready features
WHY: Machine learning models need numerical features, not raw text/dates
WHAT WE CREATE:
    - Temporal features (hour, day, season) → Tell us WHEN crimes happen
    - Geographic features (normalized coordinates) → Tell us WHERE crimes happen
    - Crime severity scores → Tell us HOW serious crimes are
    - Encoded categories → Convert text to numbers

REAL-WORLD ANALOGY:
    Raw data is like unorganized receipts
    Feature engineering is like organizing them into categories: date, amount, type
    This makes analysis possible!
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Creates ML-ready features from raw crime data
    
    Think of this as a factory that takes raw materials (crime records)
    and produces refined products (ML features)
    """
    
    def __init__(self, df):
        """
        Initialize with cleaned dataframe
        
        Parameters:
        -----------
        df : pandas DataFrame
            Cleaned crime data from preprocessing step
        """
        self.df = df.copy()
        print(f"📊 Loaded {len(self.df):,} crime records for feature engineering")
        
    def create_temporal_features(self):
        """
        Extract time-based patterns from datetime
        
        WHY THIS MATTERS:
            Crime patterns vary by time:
            - Late night → More violent crimes
            - Weekends → Different crime types
            - Summer → More outdoor crimes
            
        CREATES:
            - Hour (0-23): When during day
            - Day of week: Monday-Sunday
            - Month (1-12): Seasonal patterns
            - Season: Winter/Spring/Summer/Fall
            - Weekend flag: Yes/No
            - Time of day: Morning/Afternoon/Evening/Night
            
        BUSINESS VALUE:
            Police can deploy officers at right TIME
        """
        print("\n🕐 Creating temporal features...")
        
        # Ensure Date column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # Extract basic time components
        self.df['Hour'] = self.df['Date'].dt.hour  # 0-23
        self.df['Day_of_Week'] = self.df['Date'].dt.day_name()  # Monday, Tuesday, etc.
        self.df['Day_of_Week_Num'] = self.df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        self.df['Month'] = self.df['Date'].dt.month  # 1-12
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Day'] = self.df['Date'].dt.day
        
        # Create weekend flag (Saturday=5, Sunday=6)
        self.df['Is_Weekend'] = self.df['Day_of_Week_Num'].isin([5, 6]).astype(int)
        
        # Create season based on month
        def get_season(month):
            """Convert month to season"""
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:  # 9, 10, 11
                return 'Fall'
        
        self.df['Season'] = self.df['Month'].apply(get_season)
        
        # Create time of day categories
        def get_time_of_day(hour):
            """Categorize hour into time periods"""
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:  # 21-4
                return 'Night'
        
        self.df['Time_of_Day'] = self.df['Hour'].apply(get_time_of_day)
        
        print(f"   ✅ Created temporal features:")
        print(f"      • Hour range: {self.df['Hour'].min()}-{self.df['Hour'].max()}")
        print(f"      • Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        print(f"      • Weekend crimes: {self.df['Is_Weekend'].sum():,} ({self.df['Is_Weekend'].mean()*100:.1f}%)")
        
        return self
    
    def create_geographic_features(self):
        """
        Create location-based features
        
        WHY THIS MATTERS:
            Crime clusters in specific areas
            Need to identify high-risk zones for police deployment
            
        CREATES:
            - Normalized coordinates (0-1 scale): For fair distance calculations
            - Geographic bins: Divide city into grid cells
            - Grid cell IDs: Unique identifier for each area
            
        BUSINESS VALUE:
            Police know WHERE to focus resources
        """
        print("\n🗺️  Creating geographic features...")
        
        # Normalize coordinates to 0-1 scale
        # WHY? Clustering algorithms work better with normalized data
        lat_min, lat_max = self.df['Latitude'].min(), self.df['Latitude'].max()
        lon_min, lon_max = self.df['Longitude'].min(), self.df['Longitude'].max()
        
        self.df['Lat_Normalized'] = (self.df['Latitude'] - lat_min) / (lat_max - lat_min)
        self.df['Long_Normalized'] = (self.df['Longitude'] - lon_min) / (lon_max - lon_min)
        
        # Create geographic bins (divide city into 20x20 grid)
        # This creates 400 cells covering Chicago
        self.df['Lat_Bin'] = pd.cut(self.df['Latitude'], bins=20, labels=False)
        self.df['Long_Bin'] = pd.cut(self.df['Longitude'], bins=20, labels=False)
        
        # Create grid cell identifier (combination of lat and long bins)
        self.df['Grid_Cell'] = (
            self.df['Lat_Bin'].astype(str) + '_' + 
            self.df['Long_Bin'].astype(str)
        )
        
        # Calculate distance from city center (Chicago downtown)
        chicago_center_lat = 41.8781
        chicago_center_lon = -87.6298
        
        # Haversine distance approximation
        self.df['Distance_From_Center'] = np.sqrt(
            (self.df['Latitude'] - chicago_center_lat)**2 + 
            (self.df['Longitude'] - chicago_center_lon)**2
        )
        
        print(f"   ✅ Created geographic features:")
        print(f"      • Latitude range: {lat_min:.4f} to {lat_max:.4f}")
        print(f"      • Longitude range: {lon_min:.4f} to {lon_max:.4f}")
        print(f"      • Unique grid cells: {self.df['Grid_Cell'].nunique()}")
        
        return self
    
    def create_crime_severity_score(self):
        """
        Assign severity scores to different crime types
        
        WHY THIS MATTERS:
            Not all crimes are equal
            Homicide needs more attention than theft
            Helps prioritize police response
            
        SCORING SYSTEM:
            5 - Most Severe: Homicide, Sexual Assault, Kidnapping
            4 - Severe: Robbery, Arson, Weapons Violations
            3 - Moderate: Battery, Burglary, Narcotics
            2 - Minor: Theft, Vandalism, Trespassing
            1 - Least Severe: Liquor violations, Gambling
            
        BUSINESS VALUE:
            Prioritize high-severity crimes for faster response
        """
        print("\n⚖️  Creating crime severity scores...")
        
        # Define severity mapping based on crime seriousness
        severity_map = {
            # Most Severe (5) - Violent crimes against persons
            'HOMICIDE': 5,
            'CRIM SEXUAL ASSAULT': 5,
            'KIDNAPPING': 5,
            
            # Severe (4) - Serious violent crimes
            'ARSON': 4,
            'ROBBERY': 4,
            'ASSAULT': 4,
            'WEAPONS VIOLATION': 4,
            'OFFENSE INVOLVING CHILDREN': 4,
            'SEX OFFENSE': 4,
            
            # Moderate (3) - Property crimes and violence
            'BATTERY': 3,
            'BURGLARY': 3,
            'MOTOR VEHICLE THEFT': 3,
            'NARCOTICS': 3,
            'STALKING': 3,
            'INTIMIDATION': 3,
            'INTERFERENCE WITH PUBLIC OFFICER': 3,
            
            # Minor (2) - Property crimes
            'THEFT': 2,
            'CRIMINAL DAMAGE': 2,
            'DECEPTIVE PRACTICE': 2,
            'CRIMINAL TRESPASS': 2,
            'PUBLIC PEACE VIOLATION': 2,
            'OTHER OFFENSE': 2,
            
            # Least Severe (1) - Minor offenses
            'PROSTITUTION': 1,
            'GAMBLING': 1,
            'LIQUOR LAW VIOLATION': 1,
            'OBSCENITY': 1,
            'NON-CRIMINAL': 1,
            'PUBLIC INDECENCY': 1
        }
        
        # Apply severity score (default to 2 if not in map)
        self.df['Crime_Severity'] = self.df['Primary Type'].map(severity_map).fillna(2)
        
        # Show severity distribution
        severity_dist = self.df['Crime_Severity'].value_counts().sort_index()
        
        print(f"   ✅ Assigned severity scores:")
        print(f"      • Score range: {self.df['Crime_Severity'].min():.0f} to {self.df['Crime_Severity'].max():.0f}")
        print(f"      • Average severity: {self.df['Crime_Severity'].mean():.2f}")
        print(f"      • High severity crimes (4-5): {(self.df['Crime_Severity'] >= 4).sum():,} ({(self.df['Crime_Severity'] >= 4).mean()*100:.1f}%)")
        
        return self
    
    def encode_categorical_features(self):
        """
        Convert text categories to numbers
        
        WHY THIS MATTERS:
            ML algorithms need numbers, not text
            "THEFT" → 15, "BATTERY" → 3, etc.
            
        ENCODES:
            - Crime types: 33 categories → 0-32
            - Location descriptions: ~100 types → 0-99
            
        BUSINESS VALUE:
            Makes data ready for ML algorithms
        """
        print("\n🔢 Encoding categorical features...")
        
        # Label encode crime types
        le_crime = LabelEncoder()
        self.df['Primary_Type_Encoded'] = le_crime.fit_transform(self.df['Primary Type'])
        
        # Label encode location descriptions
        le_location = LabelEncoder()
        self.df['Location_Desc_Encoded'] = le_location.fit_transform(
            self.df['Location Description']
        )
        
        # Save encoders for future use (to decode predictions)
        self.crime_encoder = le_crime
        self.location_encoder = le_location
        
        print(f"   ✅ Encoded categorical variables:")
        print(f"      • Crime types: {len(le_crime.classes_)} unique values")
        print(f"      • Location types: {len(le_location.classes_)} unique values")
        
        return self
    
    def create_interaction_features(self):
        """
        Create combined features
        
        WHY THIS MATTERS:
            Some patterns only emerge when combining features
            Example: Crimes at night + weekends = party-related crimes
            
        CREATES:
            - Hour × Weekend: Identifies weekend night patterns
            - Severity × Arrest: High severity with arrests
            - Location × Time: Specific place-time combinations
            
        BUSINESS VALUE:
            Capture complex patterns that single features miss
        """
        print("\n🔗 Creating interaction features...")
        
        # Hour and weekend interaction
        # High values = weekend nights (high risk)
        self.df['Hour_Weekend_Interaction'] = (
            self.df['Hour'] * self.df['Is_Weekend']
        )
        
        # Severity and arrest interaction
        # Shows how effective policing is for serious crimes
        self.df['Severity_Arrest_Score'] = (
            self.df['Crime_Severity'] * self.df['Arrest'].astype(int)
        )
        
        # Time of day and severity
        # Some times have more serious crimes
        time_severity_map = {
            'Morning': 1,
            'Afternoon': 2, 
            'Evening': 3,
            'Night': 4
        }
        self.df['Time_Severity_Risk'] = (
            self.df['Time_of_Day'].map(time_severity_map) * 
            self.df['Crime_Severity']
        )
        
        print(f"   ✅ Created interaction features:")
        print(f"      • Hour-Weekend interaction range: {self.df['Hour_Weekend_Interaction'].min():.0f} to {self.df['Hour_Weekend_Interaction'].max():.0f}")
        print(f"      • Severity-Arrest score range: {self.df['Severity_Arrest_Score'].min():.0f} to {self.df['Severity_Arrest_Score'].max():.0f}")
        
        return self
    
    def create_aggregated_features(self):
        """
        Create area-level statistics
        
        WHY THIS MATTERS:
            Some areas are historically high-crime
            Past crime predicts future crime
            
        CREATES:
            - Crimes per grid cell: Historical density
            - Average severity per area: Risk level
            - Arrest rate per area: Police effectiveness
            
        BUSINESS VALUE:
            Identify historically dangerous areas
        """
        print("\n📊 Creating aggregated features...")
        
        # Count crimes per grid cell
        grid_crime_counts = self.df.groupby('Grid_Cell').size()
        self.df['Grid_Crime_Count'] = self.df['Grid_Cell'].map(grid_crime_counts)
        
        # Average severity per grid cell
        grid_severity = self.df.groupby('Grid_Cell')['Crime_Severity'].mean()
        self.df['Grid_Avg_Severity'] = self.df['Grid_Cell'].map(grid_severity)
        
        # Arrest rate per grid cell
        grid_arrest_rate = self.df.groupby('Grid_Cell')['Arrest'].mean()
        self.df['Grid_Arrest_Rate'] = self.df['Grid_Cell'].map(grid_arrest_rate)
        
        print(f"   ✅ Created aggregated features:")
        print(f"      • Avg crimes per cell: {self.df['Grid_Crime_Count'].mean():.0f}")
        print(f"      • Max crimes in one cell: {self.df['Grid_Crime_Count'].max():,}")
        
        return self
    
    def get_feature_summary(self):
        """Display comprehensive summary of all features"""
        print("\n" + "="*70)
        print("🎯 FEATURE ENGINEERING SUMMARY")
        print("="*70)
        
        print(f"\n📊 Dataset Info:")
        print(f"   • Total Records: {len(self.df):,}")
        print(f"   • Total Features: {self.df.shape[1]}")
        print(f"   • Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        print(f"\n⏰ Temporal Features Created:")
        temporal_features = ['Hour', 'Day_of_Week', 'Month', 'Season', 'Is_Weekend', 'Time_of_Day']
        for feat in temporal_features:
            if feat in self.df.columns:
                print(f"   ✓ {feat}")
        
        print(f"\n🗺️  Geographic Features Created:")
        geographic_features = ['Lat_Normalized', 'Long_Normalized', 'Grid_Cell', 'Distance_From_Center']
        for feat in geographic_features:
            if feat in self.df.columns:
                print(f"   ✓ {feat}")
        
        print(f"\n⚖️  Crime Features Created:")
        crime_features = ['Crime_Severity', 'Primary_Type_Encoded', 'Location_Desc_Encoded']
        for feat in crime_features:
            if feat in self.df.columns:
                print(f"   ✓ {feat}")
        
        print(f"\n🔗 Interaction Features Created:")
        interaction_features = ['Hour_Weekend_Interaction', 'Severity_Arrest_Score', 'Time_Severity_Risk']
        for feat in interaction_features:
            if feat in self.df.columns:
                print(f"   ✓ {feat}")
        
        print(f"\n📊 Aggregated Features Created:")
        agg_features = ['Grid_Crime_Count', 'Grid_Avg_Severity', 'Grid_Arrest_Rate']
        for feat in agg_features:
            if feat in self.df.columns:
                print(f"   ✓ {feat}")
        
        print(f"\n📈 Key Statistics:")
        print(f"   • Hour range: {self.df['Hour'].min()}-{self.df['Hour'].max()}")
        print(f"   • Severity range: {self.df['Crime_Severity'].min()}-{self.df['Crime_Severity'].max()}")
        print(f"   • Weekend crimes: {self.df['Is_Weekend'].sum():,} ({self.df['Is_Weekend'].mean()*100:.1f}%)")
        print(f"   • High severity (4-5): {(self.df['Crime_Severity'] >= 4).sum():,} ({(self.df['Crime_Severity'] >= 4).mean()*100:.1f}%)")
        print(f"   • Arrest rate: {self.df['Arrest'].mean()*100:.1f}%")
        
        return self
    
    def save_features(self, output_path='data/processed/crime_data_features.csv'):
        """Save engineered features to CSV"""
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Saved featured data to: {output_path}")
        print(f"   File size: {pd.read_csv(output_path).memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return self
    
    def get_dataframe(self):
        """Return featured dataframe"""
        return self.df
    
    def get_feature_columns(self):
        """
        Get list of feature columns ready for ML modeling
        
        RETURNS:
            List of numerical feature names that can be used directly
            in clustering and other ML algorithms
        """
        feature_cols = [
            # Temporal features
            'Hour', 'Day_of_Week_Num', 'Month', 'Is_Weekend',
            
            # Geographic features
            'Lat_Normalized', 'Long_Normalized',
            'Distance_From_Center',
            
            # Crime features
            'Crime_Severity', 'Primary_Type_Encoded',
            'Location_Desc_Encoded',
            
            # Boolean features
            'Arrest', 'Domestic',
            
            # Interaction features
            'Hour_Weekend_Interaction',
            'Severity_Arrest_Score',
            
            # Aggregated features
            'Grid_Crime_Count',
            'Grid_Avg_Severity',
            'Grid_Arrest_Rate'
        ]
        
        # Return only columns that exist in dataframe
        available_features = [col for col in feature_cols if col in self.df.columns]
        
        print(f"\n📋 Available ML Features: {len(available_features)}")
        for i, feat in enumerate(available_features, 1):
            print(f"   {i}. {feat}")
        
        return available_features


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("🚀 PATROLIQ FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Load cleaned data
    print("\n📥 Loading cleaned data...")
    try:
        df = pd.read_csv('data/processed/crime_data_cleaned.csv')
        print(f"✅ Loaded {len(df):,} records")
    except FileNotFoundError:
        print("❌ Error: crime_data_cleaned.csv not found!")
        print("   Please run data_preprocessing.py first")
        exit(1)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(df)
    
    # Execute feature engineering pipeline
    print("\n" + "="*70)
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    engineer.create_temporal_features()
    engineer.create_geographic_features()
    engineer.create_crime_severity_score()
    engineer.encode_categorical_features()
    engineer.create_interaction_features()
    engineer.create_aggregated_features()
    
    # Display summary
    engineer.get_feature_summary()
    
    # Get ML-ready features
    ml_features = engineer.get_feature_columns()
    
    # Save results
    engineer.save_features()
    
    # Get final dataframe
    df_featured = engineer.get_dataframe()
    
    print("\n" + "="*70)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Dataset Shape: {df_featured.shape}")
    print(f"📝 Total Features: {df_featured.shape[1]}")
    print(f"🎯 ML-Ready Features: {len(ml_features)}")
    print(f"\n💡 Next Step: Run clustering.py to identify crime hotspots")
    print("="*70)