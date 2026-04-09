import pandas as pd

df = pd.read_csv("data/processed/crime_data_final.csv")
df.to_parquet("data/processed/crime_data_final.parquet")
print("Parquet file created!")
