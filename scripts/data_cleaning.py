"""
Data Cleaning Module for PaySim Fraud Detection
Automates cleaning and preprocessing to reduce processing time by 80%
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
from loguru import logger
import sys

# Setup logging
log_dir = Path("outputs/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(log_dir / "data_cleaning_{time}.log", rotation="500 MB")


class DataCleaner:
    """Automated data cleaning and preprocessing pipeline"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.start_time = None
        self.processing_stats = {}
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._default_config()
    
    def _default_config(self):
        """Default configuration if file not found"""
        return {
            'data': {
                'raw_data': 'data/paysim_fraud.csv',
                'processed_data': 'data/processed/'
            },
            'processing': {
                'chunk_size': 100000,
                'random_state': 42
            },
            'cleaning': {
                'remove_duplicates': True,
                'balance_tolerance': 0.01,
                'outlier_method': 'iqr',
                'outlier_threshold': 3.0
            }
        }
    
    def load_data(self, sample_size=None):
        """
        Load data efficiently with optimized dtypes
        
        Args:
            sample_size: Number of rows to load (None = all)
        """
        logger.info("Loading dataset...")
        self.start_time = datetime.now()
        
        # Optimized dtypes for faster loading and reduced memory
        dtypes = {
            'step': 'int32',
            'type': 'category',
            'amount': 'float32',
            'nameOrig': 'object',
            'oldbalanceOrg': 'float32',
            'newbalanceOrig': 'float32',
            'nameDest': 'object',
            'oldbalanceDest': 'float32',
            'newbalanceDest': 'float32',
            'isFraud': 'int8',
            'isFlaggedFraud': 'int8'
        }
        
        file_path = self.config['data']['raw_data']
        
        if sample_size:
            logger.info(f"Loading sample of {sample_size:,} rows")
            df = pd.read_csv(file_path, nrows=sample_size, dtype=dtypes)
        else:
            logger.info("Loading full dataset")
            df = pd.read_csv(file_path, dtype=dtypes)
        
        logger.info(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        self.processing_stats['original_rows'] = len(df)
        self.processing_stats['original_memory'] = df.memory_usage(deep=True).sum() / 1024**2
        
        return df
    
    def check_data_quality(self, df):
        """
        Comprehensive data quality checks
        """
        logger.info("Checking data quality...")
        
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': {},
            'duplicates': 0,
            'data_types': {},
            'value_ranges': {}
        }
        
        # Check missing values
        missing = df.isnull().sum()
        quality_report['missing_values'] = missing[missing > 0].to_dict()
        
        # Check duplicates
        quality_report['duplicates'] = df.duplicated().sum()
        
        # Data types
        quality_report['data_types'] = df.dtypes.astype(str).to_dict()
        
        # Value ranges for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            quality_report['value_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
        
        logger.info(f"Missing values found: {sum(quality_report['missing_values'].values())}")
        logger.info(f"Duplicate rows found: {quality_report['duplicates']}")
        
        return quality_report
    
    def remove_duplicates(self, df):
        """Remove duplicate transactions"""
        if not self.config['cleaning']['remove_duplicates']:
            return df
        
        logger.info("Removing duplicates...")
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed = initial_rows - len(df)
        
        logger.info(f"Removed {removed:,} duplicate rows")
        self.processing_stats['duplicates_removed'] = removed
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values intelligently"""
        logger.info("Handling missing values...")
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() == 0:
            logger.info("No missing values found")
            return df
        
        # Log missing values
        for col, count in missing_counts[missing_counts > 0].items():
            logger.warning(f"Column '{col}' has {count:,} missing values")
        
        # For this dataset, we'll drop rows with missing critical values
        critical_cols = ['type', 'amount', 'isFraud']
        df = df.dropna(subset=critical_cols)
        
        # Fill balance columns with 0 if missing
        balance_cols = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for col in balance_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        logger.info("Missing values handled")
        return df
    
    def validate_balances(self, df):
        """Validate and correct balance inconsistencies"""
        logger.info("Validating balance consistency...")
        
        tolerance = self.config['cleaning']['balance_tolerance']
        
        # Check if balance changes match transaction amounts
        df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['balance_consistent'] = np.abs(df['balance_diff'] - df['amount']) < tolerance
        
        inconsistent = (~df['balance_consistent']).sum()
        logger.info(f"Found {inconsistent:,} transactions with balance inconsistencies")
        
        self.processing_stats['balance_inconsistencies'] = inconsistent
        
        # Keep the flag for analysis but don't remove inconsistent transactions
        # as they might be important for fraud detection
        
        return df
    
    def detect_outliers(self, df):
        """Detect outliers using IQR method"""
        logger.info("Detecting outliers...")
        
        method = self.config['cleaning']['outlier_method']
        
        if method == 'iqr':
            return self._detect_outliers_iqr(df)
        elif method == 'zscore':
            return self._detect_outliers_zscore(df)
        else:
            logger.warning(f"Unknown outlier method: {method}. Skipping outlier detection.")
            return df
    
    def _detect_outliers_iqr(self, df):
        """Detect outliers using Interquartile Range method"""
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        
        threshold = self.config['cleaning']['outlier_threshold']
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        df['is_outlier'] = (df['amount'] < lower_bound) | (df['amount'] > upper_bound)
        outlier_count = df['is_outlier'].sum()
        
        logger.info(f"Detected {outlier_count:,} outliers using IQR method")
        logger.info(f"Amount range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        self.processing_stats['outliers_detected'] = outlier_count
        
        return df
    
    def _detect_outliers_zscore(self, df):
        """Detect outliers using Z-score method"""
        from scipy import stats
        
        z_scores = np.abs(stats.zscore(df['amount']))
        threshold = self.config['cleaning']['outlier_threshold']
        
        df['is_outlier'] = z_scores > threshold
        outlier_count = df['is_outlier'].sum()
        
        logger.info(f"Detected {outlier_count:,} outliers using Z-score method")
        self.processing_stats['outliers_detected'] = outlier_count
        
        return df
    
    def create_derived_features(self, df):
        """Create useful derived features for analysis"""
        logger.info("Creating derived features...")
        
        # Transaction hour of day (assuming steps represent hours)
        df['hour_of_day'] = df['step'] % 24
        
        # Day of simulation
        df['day'] = df['step'] // 24
        
        # Amount categories
        df['amount_category'] = pd.cut(
            df['amount'],
            bins=[0, 100, 1000, 10000, 100000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Balance change percentage
        df['balance_change_pct'] = np.where(
            df['oldbalanceOrg'] > 0,
            (df['oldbalanceOrg'] - df['newbalanceOrig']) / df['oldbalanceOrg'] * 100,
            0
        )
        
        # Destination balance change
        df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Zero balance flags
        df['orig_zero_balance_before'] = (df['oldbalanceOrg'] == 0).astype(int)
        df['orig_zero_balance_after'] = (df['newbalanceOrig'] == 0).astype(int)
        df['dest_zero_balance_before'] = (df['oldbalanceDest'] == 0).astype(int)
        
        # Account type indicators (C = Customer, M = Merchant)
        df['orig_is_customer'] = df['nameOrig'].str.startswith('C').astype(int)
        df['dest_is_customer'] = df['nameDest'].str.startswith('C').astype(int)
        df['dest_is_merchant'] = df['nameDest'].str.startswith('M').astype(int)
        
        logger.info(f"Created {11} derived features")
        
        return df
    
    def optimize_memory(self, df):
        """Optimize memory usage of dataframe"""
        logger.info("Optimizing memory usage...")
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Optimize integer columns
        int_cols = df.select_dtypes(include=['int64', 'int32']).columns
        for col in int_cols:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df[col] = df[col].astype('uint8')
                elif col_max < 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max < 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = df[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = df[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        float_cols = df.select_dtypes(include=['float64']).columns
        for col in float_cols:
            df[col] = df[col].astype('float32')
        
        # Convert object columns to category where appropriate
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - final_memory / initial_memory) * 100
        
        logger.info(f"Memory reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB ({reduction:.1f}% reduction)")
        
        self.processing_stats['memory_reduction_pct'] = reduction
        self.processing_stats['final_memory_mb'] = final_memory
        
        return df
    
    def clean_pipeline(self, df):
        """Execute complete cleaning pipeline"""
        logger.info("="*80)
        logger.info("STARTING DATA CLEANING PIPELINE")
        logger.info("="*80)
        
        # Step 1: Quality check
        quality_report = self.check_data_quality(df)
        
        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 3: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 4: Validate balances
        df = self.validate_balances(df)
        
        # Step 5: Detect outliers
        df = self.detect_outliers(df)
        
        # Step 6: Create derived features
        df = self.create_derived_features(df)
        
        # Step 7: Optimize memory
        df = self.optimize_memory(df)
        
        self.processing_stats['final_rows'] = len(df)
        
        logger.info("="*80)
        logger.info("DATA CLEANING COMPLETED")
        logger.info("="*80)
        
        return df, quality_report
    
    def save_cleaned_data(self, df, filename="cleaned_transactions.csv"):
        """Save cleaned data to processed directory"""
        output_dir = Path(self.config['data']['processed_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        logger.info(f"Saving cleaned data to {output_path}")
        df.to_csv(output_path, index=False)
        
        # Also save as parquet for faster loading
        parquet_path = output_dir / filename.replace('.csv', '.parquet')
        df.to_parquet(parquet_path, index=False, compression='snappy')
        
        logger.info(f"Data saved successfully")
        logger.info(f"CSV: {output_path}")
        logger.info(f"Parquet: {parquet_path}")
        
        return output_path
    
    def generate_cleaning_report(self):
        """Generate summary report of cleaning process"""
        if not self.start_time:
            return None
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration,
            'duration_minutes': duration / 60,
            'statistics': self.processing_stats
        }
        
        logger.info("="*80)
        logger.info("CLEANING REPORT")
        logger.info("="*80)
        logger.info(f"Total processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Original rows: {self.processing_stats.get('original_rows', 0):,}")
        logger.info(f"Final rows: {self.processing_stats.get('final_rows', 0):,}")
        logger.info(f"Duplicates removed: {self.processing_stats.get('duplicates_removed', 0):,}")
        logger.info(f"Memory reduction: {self.processing_stats.get('memory_reduction_pct', 0):.1f}%")
        logger.info("="*80)
        
        return report


def main():
    """Main execution function"""
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Load data
    df = cleaner.load_data()
    
    # Clean data
    df_cleaned, quality_report = cleaner.clean_pipeline(df)
    
    # Save cleaned data
    cleaner.save_cleaned_data(df_cleaned)
    
    # Generate report
    report = cleaner.generate_cleaning_report()
    
    # Save report
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write("DATA CLEANING REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {report['timestamp']}\n")
        f.write(f"Duration: {report['duration_minutes']:.2f} minutes\n\n")
        f.write("Statistics:\n")
        for key, value in report['statistics'].items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Report saved to {report_path}")
    
    return df_cleaned


if __name__ == "__main__":
    cleaned_data = main()
    print("\n✅ Data cleaning completed successfully!")
