"""
Tableau Export Pipeline for PaySim Fraud Detection
Loads cleaned transaction data and exports aggregated summaries for Tableau dashboards
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/tableau_exports',
        'outputs/logs'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory verified: {directory}")


def log_message(message, log_file='outputs/logs/export_log.txt'):
    """Log messages to file with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry.strip())
    
    with open(log_file, 'a') as f:
        f.write(log_entry)


def extract_region_from_customer_id(customer_id):
    """
    Extract region code from customer ID
    Customer IDs are in format: C########
    We use the first digit after 'C' to determine region (0-9)
    """
    if pd.isna(customer_id):
        return 'UNKNOWN'
    
    customer_id = str(customer_id)
    
    if customer_id.startswith('C') and len(customer_id) > 1:
        try:
            # Use first digit to create 10 regions
            digit = int(customer_id[1])
            region_num = digit % 10
            return f'REGION_{region_num}'
        except (ValueError, IndexError):
            return 'UNKNOWN'
    elif customer_id.startswith('M'):
        return 'MERCHANT'
    else:
        return 'UNKNOWN'


def load_cleaned_data(file_path):
    """Load the cleaned transaction data"""
    log_message(f"Loading cleaned data from: {file_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            log_message(f"ERROR: File not found - {file_path}")
            return None
        
        # Load data
        df = pd.read_csv(file_path)
        log_message(f"✓ Successfully loaded {len(df):,} transactions")
        log_message(f"✓ Columns: {list(df.columns)}")
        log_message(f"✓ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
        
    except Exception as e:
        log_message(f"ERROR: Failed to load data - {str(e)}")
        return None


def prepare_data_for_aggregation(df):
    """Prepare data by adding/validating required columns"""
    log_message("Preparing data for aggregation...")
    
    # Add region column if not exists
    if 'region' not in df.columns:
        log_message("Creating 'region' column from customer IDs...")
        df['region'] = df['nameOrig'].apply(extract_region_from_customer_id)
        log_message(f"✓ Created {df['region'].nunique()} unique regions")
    
    # Ensure required columns exist
    required_cols = ['type', 'amount', 'isFraud']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        log_message(f"ERROR: Missing required columns: {missing_cols}")
        return None
    
    log_message("✓ Data preparation complete")
    return df


def create_fraud_summary_by_type(df):
    """
    Create fraud summary aggregated by transaction type
    
    Output columns:
    - transaction_type: Type of transaction
    - total_txns: Total number of transactions
    - fraud_txns: Number of fraudulent transactions
    - fraud_rate: Percentage of fraudulent transactions
    - avg_amount: Average transaction amount
    """
    log_message("Creating fraud summary by transaction type...")
    
    summary = df.groupby('type').agg({
        'isFraud': ['count', 'sum'],
        'amount': 'mean'
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['transaction_type', 'total_txns', 'fraud_txns', 'avg_amount']
    
    # Calculate fraud rate
    summary['fraud_rate'] = (summary['fraud_txns'] / summary['total_txns'] * 100).round(4)
    
    # Round average amount to 2 decimal places
    summary['avg_amount'] = summary['avg_amount'].round(2)
    
    # Sort by fraud rate descending
    summary = summary.sort_values('fraud_rate', ascending=False)
    
    # Log summary statistics
    log_message("✓ Fraud summary by type created:")
    for _, row in summary.iterrows():
        log_message(f"  {row['transaction_type']:12s} | "
                   f"Total: {int(row['total_txns']):10,} | "
                   f"Fraud: {int(row['fraud_txns']):8,} | "
                   f"Rate: {row['fraud_rate']:6.2f}% | "
                   f"Avg: ${row['avg_amount']:12,.2f}")
    
    return summary


def create_regional_fraud_summary(df):
    """
    Create fraud summary aggregated by customer region
    
    Output columns:
    - region: Customer region
    - total_txns: Total number of transactions
    - fraud_txns: Number of fraudulent transactions
    - fraud_rate: Percentage of fraudulent transactions
    - avg_amount: Average transaction amount
    """
    log_message("Creating regional fraud summary...")
    
    summary = df.groupby('region').agg({
        'isFraud': ['count', 'sum'],
        'amount': 'mean'
    }).reset_index()
    
    # Flatten column names
    summary.columns = ['region', 'total_txns', 'fraud_txns', 'avg_amount']
    
    # Calculate fraud rate
    summary['fraud_rate'] = (summary['fraud_txns'] / summary['total_txns'] * 100).round(4)
    
    # Round average amount to 2 decimal places
    summary['avg_amount'] = summary['avg_amount'].round(2)
    
    # Sort by fraud rate descending
    summary = summary.sort_values('fraud_rate', ascending=False)
    
    # Log summary statistics
    log_message("✓ Regional fraud summary created:")
    for _, row in summary.iterrows():
        log_message(f"  {row['region']:12s} | "
                   f"Total: {int(row['total_txns']):10,} | "
                   f"Fraud: {int(row['fraud_txns']):8,} | "
                   f"Rate: {row['fraud_rate']:6.2f}% | "
                   f"Avg: ${row['avg_amount']:12,.2f}")
    
    return summary


def save_export(df, filename, output_dir='data/tableau_exports'):
    """Save DataFrame to CSV in tableau exports directory"""
    output_path = os.path.join(output_dir, filename)
    
    try:
        df.to_csv(output_path, index=False)
        file_size = os.path.getsize(output_path) / 1024  # Size in KB
        log_message(f"✓ Saved: {filename} ({len(df):,} rows, {file_size:.2f} KB)")
        return True
    except Exception as e:
        log_message(f"ERROR: Failed to save {filename} - {str(e)}")
        return False


def validate_exports(export_files):
    """Validate that all export files were created successfully"""
    log_message("\nValidating exports...")
    
    all_valid = True
    for filename in export_files:
        filepath = os.path.join('data/tableau_exports', filename)
        
        if os.path.exists(filepath):
            # Check file size
            size_mb = os.path.getsize(filepath) / 1024**2
            
            # Load and check row count
            df = pd.read_csv(filepath)
            row_count = len(df)
            col_count = len(df.columns)
            
            log_message(f"✓ {filename}: {row_count} rows, {col_count} columns, {size_mb:.2f} MB")
            
            # Display preview
            log_message(f"  Preview of {filename}:")
            log_message(f"  Columns: {list(df.columns)}")
            log_message(f"  First row: {df.iloc[0].to_dict()}")
            
        else:
            log_message(f"✗ {filename}: FILE NOT FOUND")
            all_valid = False
    
    return all_valid


def generate_export_summary(df, export_files):
    """Generate a summary of the export process"""
    log_message("\n" + "="*80)
    log_message("EXPORT SUMMARY")
    log_message("="*80)
    
    # Overall statistics
    total_txns = len(df)
    total_fraud = df['isFraud'].sum()
    fraud_rate = (total_fraud / total_txns * 100)
    total_amount = df['amount'].sum()
    avg_amount = df['amount'].mean()
    
    log_message(f"Source Data Statistics:")
    log_message(f"  Total Transactions: {total_txns:,}")
    log_message(f"  Fraudulent Transactions: {total_fraud:,}")
    log_message(f"  Overall Fraud Rate: {fraud_rate:.4f}%")
    log_message(f"  Total Transaction Amount: ${total_amount:,.2f}")
    log_message(f"  Average Transaction Amount: ${avg_amount:,.2f}")
    
    log_message(f"\nTransaction Type Distribution:")
    type_dist = df['type'].value_counts()
    for txn_type, count in type_dist.items():
        pct = (count / total_txns * 100)
        log_message(f"  {txn_type:12s}: {count:10,} ({pct:5.2f}%)")
    
    log_message(f"\nRegion Distribution:")
    region_dist = df['region'].value_counts()
    for region, count in region_dist.head(10).items():
        pct = (count / total_txns * 100)
        log_message(f"  {region:12s}: {count:10,} ({pct:5.2f}%)")
    
    log_message(f"\nExported Files:")
    for filename in export_files:
        log_message(f"  ✓ {filename}")
    
    log_message("="*80)


def main():
    """Main execution function"""
    
    # Clear previous log
    log_file = 'outputs/logs/export_log.txt'
    if os.path.exists(log_file):
        os.remove(log_file)
    
    log_message("="*80)
    log_message("TABLEAU EXPORT PIPELINE - STARTED")
    log_message("="*80)
    
    start_time = datetime.now()
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Load cleaned data
    input_file = 'data/processed/cleaned_transactions.csv'
    df = load_cleaned_data(input_file)
    
    if df is None:
        log_message("\n✗ Export pipeline failed: Could not load data")
        return False
    
    # Step 3: Prepare data
    df = prepare_data_for_aggregation(df)
    
    if df is None:
        log_message("\n✗ Export pipeline failed: Data preparation error")
        return False
    
    # Step 4: Create aggregated summaries
    log_message("\n" + "-"*80)
    log_message("CREATING AGGREGATED SUMMARIES")
    log_message("-"*80)
    
    # Summary 1: Fraud summary by transaction type
    fraud_by_type = create_fraud_summary_by_type(df)
    
    # Summary 2: Regional fraud summary
    regional_summary = create_regional_fraud_summary(df)
    
    # Step 5: Save exports
    log_message("\n" + "-"*80)
    log_message("SAVING EXPORTS")
    log_message("-"*80)
    
    export_files = []
    
    success1 = save_export(fraud_by_type, 'fraud_summary_by_type.csv')
    if success1:
        export_files.append('fraud_summary_by_type.csv')
    
    success2 = save_export(regional_summary, 'regional_fraud_summary.csv')
    if success2:
        export_files.append('regional_fraud_summary.csv')
    
    # Step 6: Validate exports
    if export_files:
        validate_exports(export_files)
    
    # Step 7: Generate summary
    generate_export_summary(df, export_files)
    
    # Calculate total time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    log_message("\n" + "="*80)
    log_message("TABLEAU EXPORT PIPELINE - COMPLETED")
    log_message("="*80)
    log_message(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    log_message(f"Exports saved to: data/tableau_exports/")
    log_message(f"Log saved to: {log_file}")
    log_message("\n✓ All exports completed successfully!")
    log_message("✓ Ready for Tableau import")
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "="*80)
        print("SUCCESS: Tableau exports are ready!")
        print("="*80)
        print("\nNext steps:")
        print("1. Open Tableau Desktop")
        print("2. Connect to Text File data source")
        print("3. Navigate to data/tableau_exports/")
        print("4. Import the CSV files:")
        print("   - fraud_summary_by_type.csv")
        print("   - regional_fraud_summary.csv")
        print("5. Start creating your dashboards!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("ERROR: Export pipeline failed. Check logs for details.")
        print("="*80)
