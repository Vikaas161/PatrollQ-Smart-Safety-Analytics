# Data Preprocessing Module for Chicago Crime Dataset
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Handles all data loading, cleaning, and sampling operations"""
    
    def __init__(self, file_path, sample_size=500000, random_state=42):
        """
        Initialize preprocessor
        
        Parameters:
        - file_path: Path to Chicago crime CSV file
        - sample_size: Number of records to sample (default 500K)
        - random_state: For reproducibility
        """
        self.file_path = file_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.df = None
        
    def load_data(self):
        """
        Load and sample crime data
        WHY: 8.44M records are too large - we sample 500K for efficiency
        """
        print("📥 Loading Chicago Crime Dataset...")
        print("⏳ This may take a few minutes for large files...")
        
        # Load data in chunks for memory efficiency
        chunks = []
        chunk_size = 100000
        
        for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size >= self.sample_size * 2:
                break
        
        # Combine chunks
        df_full = pd.concat(chunks, ignore_index=True)
        
        # Sample recent crimes (sort by date first)
        df_full['Date'] = pd.to_datetime(df_full['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        df_full = df_full.sort_values('Date', ascending=False)
        
        # Take sample
        self.df = df_full.head(self.sample_size).copy()
        
        print(f"✅ Loaded {len(self.df):,} crime records")
        print(f"📅 Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        
        return self
    
    def clean_data(self):
        """
        Clean missing values and inconsistencies
        WHY: Missing data causes errors in ML models
        """
        print("\n🧹 Cleaning data...")
        
        initial_rows = len(self.df)
        
        # Remove rows with missing critical columns
        critical_cols = ['Date', 'Primary Type', 'Latitude', 'Longitude']
        self.df = self.df.dropna(subset=critical_cols)
        
        # Fill missing values for other columns
        self.df['Location Description'] = self.df['Location Description'].fillna('UNKNOWN')
        self.df['Arrest'] = self.df['Arrest'].fillna(False)
        self.df['Domestic'] = self.df['Domestic'].fillna(False)
        
        # Remove invalid coordinates (outside Chicago bounds)
        self.df = self.df[
            (self.df['Latitude'] >= 41.6) & (self.df['Latitude'] <= 42.1) &
            (self.df['Longitude'] >= -87.95) & (self.df['Longitude'] <= -87.5)
        ]
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['Case Number'], keep='first')
        
        cleaned_rows = len(self.df)
        print(f"✅ Removed {initial_rows - cleaned_rows:,} invalid records")
        print(f"✅ Final dataset: {cleaned_rows:,} clean records")
        
        return self
    
    def validate_data(self):
        """
        Perform data quality checks
        WHY: Ensure data meets requirements before modeling
        """
        print("\n🔍 Validating data quality...")
        
        # Check missing values
        missing = self.df.isnull().sum()
        if missing.any():
            print(f"⚠️  Missing values found:\n{missing[missing > 0]}")
        else:
            print("✅ No missing values in critical columns")
        
        # Check data types
        print(f"\n📊 Dataset shape: {self.df.shape}")
        print(f"📊 Crime types: {self.df['Primary Type'].nunique()}")
        print(f"📊 Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        
        # Check coordinate validity
        valid_coords = (
            (self.df['Latitude'].between(41.6, 42.1)) & 
            (self.df['Longitude'].between(-87.95, -87.5))
        ).all()
        
        if valid_coords:
            print("✅ All coordinates within valid Chicago bounds")
        else:
            print("⚠️  Some coordinates outside Chicago bounds")
        
        return self
    
    def get_data_summary(self):
        """Get comprehensive data summary"""
        print("\n" + "="*60)
        print("📈 DATA SUMMARY")
        print("="*60)
        
        print(f"\n📊 Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        
        print(f"\n🔝 Top 10 Crime Types:")
        print(self.df['Primary Type'].value_counts().head(10))
        
        print(f"\n📍 Top 10 Locations:")
        print(self.df['Location Description'].value_counts().head(10))
        
        print(f"\n🚔 Arrest Rate: {self.df['Arrest'].mean()*100:.2f}%")
        print(f"🏠 Domestic Rate: {self.df['Domestic'].mean()*100:.2f}%")
        
        return self
    
    def save_processed_data(self, output_path='data/processed/crime_data_cleaned.csv'):
        """Save cleaned data"""
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Saved processed data to: {output_path}")
        return self
    
    def get_dataframe(self):
        """Return processed dataframe"""
        return self.df


# Usage Example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        file_path='data/raw/chicago_crimes.csv',
        sample_size=500000
    )
    
    # Execute preprocessing pipeline
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.validate_data()
    preprocessor.get_data_summary()
    preprocessor.save_processed_data()
    
    # Get cleaned dataframe
    df_clean = preprocessor.get_dataframe()
    print("\n✅ Preprocessing complete!")