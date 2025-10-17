import pandas as pd
import numpy as np

# Set display options for better viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('./data/paysim_fraud.csv')

print("\n" + "="*100)
print("DATASET OVERVIEW")
print("="*100)
print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]:,}")
print(f"Number of columns: {df.shape[1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print("\n" + "="*100)
print("FIRST 5 ROWS (FULL VIEW)")
print("="*100)
print(df.head())

print("\n" + "="*100)
print("SUMMARY STATISTICS (NUMERICAL COLUMNS)")
print("="*100)
print(df.describe())

print("\n" + "="*100)
print("COLUMN DETAILS")
print("="*100)
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    null_count = df[col].isnull().sum()
    unique_count = df[col].nunique()
    print(f"{i+1}. {col:20s} | Type: {str(dtype):10s} | Nulls: {null_count:8,} | Unique: {unique_count:10,}")

print("\n" + "="*100)
print("TRANSACTION TYPE DISTRIBUTION")
print("="*100)
print(df['type'].value_counts())

print("\n" + "="*100)
print("FRAUD STATISTICS")
print("="*100)
fraud_count = df['isFraud'].sum()
total_count = len(df)
fraud_percentage = (fraud_count / total_count) * 100
print(f"Total transactions: {total_count:,}")
print(f"Fraudulent transactions: {fraud_count:,}")
print(f"Non-fraudulent transactions: {total_count - fraud_count:,}")
print(f"Fraud rate: {fraud_percentage:.4f}%")

print("\n" + "="*100)
print("FLAGGED FRAUD STATISTICS")
print("="*100)
flagged_count = df['isFlaggedFraud'].sum()
flagged_percentage = (flagged_count / total_count) * 100
print(f"Flagged fraudulent transactions: {flagged_count:,}")
print(f"Flagged rate: {flagged_percentage:.6f}%")
