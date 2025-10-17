import pandas as pd
import numpy as np

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('./data/paysim_fraud.csv')

print("\n" + "="*80)
print("DATASET LOADED SUCCESSFULLY")
print("="*80)

# Display basic information
print(f"\nDataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")

# Display first 5 rows
print("\n" + "="*80)
print("FIRST 5 ROWS")
print("="*80)
print(df.head())

# Display summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(df.describe())

# Display column names and data types
print("\n" + "="*80)
print("COLUMN INFORMATION")
print("="*80)
print(df.info())

# Display null values count
print("\n" + "="*80)
print("NULL VALUES COUNT")
print("="*80)
print(df.isnull().sum())
