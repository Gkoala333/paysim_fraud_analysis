"""
Feature Engineering Module for PaySim Fraud Detection
Advanced feature creation for enhanced fraud detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("outputs/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(log_dir / "feature_engineering_{time}.log", rotation="500 MB")


class FeatureEngineer:
    """Advanced feature engineering for fraud detection"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize FeatureEngineer"""
        self.config = self._load_config(config_path)
        self.df = None
        self.new_features = []
        
    def _load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found. Using defaults.")
            return {'data': {'processed_data': 'data/processed/'}}
    
    def load_data(self, filepath=None):
        """Load transaction data"""
        if filepath is None:
            filepath = Path(self.config['data']['processed_data']) / "cleaned_transactions.csv"
        
        logger.info(f"Loading data from {filepath}")
        self.df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.df):,} transactions with {len(self.df.columns)} features")
        
        return self.df
    
    def create_amount_features(self):
        """Create amount-based features"""
        logger.info("Creating amount-based features...")
        
        # Log transformation of amount
        self.df['amount_log'] = np.log1p(self.df['amount'])
        self.new_features.append('amount_log')
        
        # Amount squared (for non-linear patterns)
        self.df['amount_squared'] = self.df['amount'] ** 2
        self.new_features.append('amount_squared')
        
        # Amount to balance ratio
        self.df['amount_to_balance_ratio'] = np.where(
            self.df['oldbalanceOrg'] > 0,
            self.df['amount'] / self.df['oldbalanceOrg'],
            0
        )
        self.new_features.append('amount_to_balance_ratio')
        
        # Amount deviation from mean by transaction type
        type_mean_amount = self.df.groupby('type')['amount'].transform('mean')
        self.df['amount_deviation_from_type_mean'] = self.df['amount'] - type_mean_amount
        self.new_features.append('amount_deviation_from_type_mean')
        
        # Amount percentile within transaction type
        self.df['amount_percentile_in_type'] = self.df.groupby('type')['amount'].rank(pct=True)
        self.new_features.append('amount_percentile_in_type')
        
        logger.info(f"Created {5} amount-based features")
        
    def create_balance_features(self):
        """Create balance-based features"""
        logger.info("Creating balance-based features...")
        
        # Balance change
        self.df['balance_change'] = self.df['oldbalanceOrg'] - self.df['newbalanceOrig']
        self.new_features.append('balance_change')
        
        # Balance change ratio
        self.df['balance_change_ratio'] = np.where(
            self.df['oldbalanceOrg'] > 0,
            self.df['balance_change'] / self.df['oldbalanceOrg'],
            0
        )
        self.new_features.append('balance_change_ratio')
        
        # Destination balance change
        self.df['dest_balance_change'] = self.df['newbalanceDest'] - self.df['oldbalanceDest']
        self.new_features.append('dest_balance_change')
        
        # Total balance (origin + destination)
        self.df['total_balance_before'] = self.df['oldbalanceOrg'] + self.df['oldbalanceDest']
        self.df['total_balance_after'] = self.df['newbalanceOrig'] + self.df['newbalanceDest']
        self.new_features.extend(['total_balance_before', 'total_balance_after'])
        
        # Balance velocity (change per hour)
        self.df['balance_velocity'] = self.df['balance_change'] / (self.df['step'] + 1)
        self.new_features.append('balance_velocity')
        
        # Error in balance calculation
        expected_new_balance = self.df['oldbalanceOrg'] - self.df['amount']
        self.df['balance_error'] = np.abs(expected_new_balance - self.df['newbalanceOrig'])
        self.new_features.append('balance_error')
        
        logger.info(f"Created {8} balance-based features")
        
    def create_temporal_features(self):
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        # Time of day categories
        self.df['time_category'] = pd.cut(
            self.df['hour_of_day'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        # Is weekend (assuming 7-day weeks)
        self.df['is_weekend'] = (self.df['day'] % 7 >= 5).astype(int)
        self.new_features.append('is_weekend')
        
        # Is business hours (9am-5pm)
        self.df['is_business_hours'] = ((self.df['hour_of_day'] >= 9) & (self.df['hour_of_day'] <= 17)).astype(int)
        self.new_features.append('is_business_hours')
        
        # Is late night (11pm-5am)
        self.df['is_late_night'] = ((self.df['hour_of_day'] >= 23) | (self.df['hour_of_day'] <= 5)).astype(int)
        self.new_features.append('is_late_night')
        
        # Week of simulation
        self.df['week'] = self.df['day'] // 7
        self.new_features.append('week')
        
        logger.info(f"Created {5} temporal features")
        
    def create_transaction_type_features(self):
        """Create transaction type features"""
        logger.info("Creating transaction type features...")
        
        # Binary encoding for each transaction type
        type_dummies = pd.get_dummies(self.df['type'], prefix='type')
        self.df = pd.concat([self.df, type_dummies], axis=1)
        self.new_features.extend(type_dummies.columns.tolist())
        
        # Is high-risk transaction type
        high_risk_types = ['TRANSFER', 'CASH_OUT']
        self.df['is_high_risk_type'] = self.df['type'].isin(high_risk_types).astype(int)
        self.new_features.append('is_high_risk_type')
        
        # Is merchant transaction
        merchant_types = ['PAYMENT', 'DEBIT']
        self.df['is_merchant_transaction'] = self.df['type'].isin(merchant_types).astype(int)
        self.new_features.append('is_merchant_transaction')
        
        logger.info(f"Created {len(type_dummies.columns) + 2} transaction type features")
        
    def create_account_features(self):
        """Create account-based features"""
        logger.info("Creating account-based features...")
        
        # Account type flags
        if 'orig_is_customer' not in self.df.columns:
            self.df['orig_is_customer'] = self.df['nameOrig'].str.startswith('C').astype(int)
        if 'dest_is_customer' not in self.df.columns:
            self.df['dest_is_customer'] = self.df['nameDest'].str.startswith('C').astype(int)
        if 'dest_is_merchant' not in self.df.columns:
            self.df['dest_is_merchant'] = self.df['nameDest'].str.startswith('M').astype(int)
        
        self.new_features.extend(['orig_is_customer', 'dest_is_customer', 'dest_is_merchant'])
        
        # Customer-to-customer transaction
        self.df['is_c2c_transaction'] = (self.df['orig_is_customer'] & self.df['dest_is_customer']).astype(int)
        self.new_features.append('is_c2c_transaction')
        
        # Customer-to-merchant transaction
        self.df['is_c2m_transaction'] = (self.df['orig_is_customer'] & self.df['dest_is_merchant']).astype(int)
        self.new_features.append('is_c2m_transaction')
        
        logger.info(f"Created {5} account-based features")
        
    def create_statistical_features(self):
        """Create statistical aggregation features"""
        logger.info("Creating statistical features...")
        
        # Transaction count by step (transactions at same time)
        step_counts = self.df.groupby('step').size()
        self.df['transactions_at_same_time'] = self.df['step'].map(step_counts)
        self.new_features.append('transactions_at_same_time')
        
        # Average amount at this time step
        step_avg_amount = self.df.groupby('step')['amount'].transform('mean')
        self.df['step_avg_amount'] = step_avg_amount
        self.new_features.append('step_avg_amount')
        
        # Amount deviation from step average
        self.df['amount_vs_step_avg'] = self.df['amount'] - self.df['step_avg_amount']
        self.new_features.append('amount_vs_step_avg')
        
        logger.info(f"Created {3} statistical features")
        
    def create_interaction_features(self):
        """Create interaction features between variables"""
        logger.info("Creating interaction features...")
        
        # Amount * Is high risk type
        self.df['amount_x_risk_type'] = self.df['amount'] * self.df['is_high_risk_type']
        self.new_features.append('amount_x_risk_type')
        
        # Amount * Balance ratio
        self.df['amount_x_balance_ratio'] = self.df['amount'] * self.df['amount_to_balance_ratio']
        self.new_features.append('amount_x_balance_ratio')
        
        # Is late night * High risk type
        self.df['late_night_x_risk_type'] = self.df['is_late_night'] * self.df['is_high_risk_type']
        self.new_features.append('late_night_x_risk_type')
        
        # Balance change * Type
        self.df['balance_change_x_c2c'] = self.df['balance_change'] * self.df['is_c2c_transaction']
        self.new_features.append('balance_change_x_c2c')
        
        logger.info(f"Created {4} interaction features")
        
    def create_anomaly_features(self):
        """Create anomaly detection features"""
        logger.info("Creating anomaly detection features...")
        
        # Z-score of amount
        amount_mean = self.df['amount'].mean()
        amount_std = self.df['amount'].std()
        self.df['amount_zscore'] = (self.df['amount'] - amount_mean) / amount_std
        self.new_features.append('amount_zscore')
        
        # Is amount outlier (>3 std dev)
        self.df['is_amount_outlier'] = (np.abs(self.df['amount_zscore']) > 3).astype(int)
        self.new_features.append('is_amount_outlier')
        
        # Balance inconsistency flag
        if 'balance_error' in self.df.columns:
            self.df['has_balance_error'] = (self.df['balance_error'] > 0.01).astype(int)
            self.new_features.append('has_balance_error')
        
        # Complete account drain
        self.df['complete_drain'] = (self.df['newbalanceOrig'] == 0).astype(int)
        self.new_features.append('complete_drain')
        
        # Suspicious pattern: large amount + complete drain + high risk type
        self.df['suspicious_pattern'] = (
            (self.df['amount_percentile_in_type'] > 0.9) &
            (self.df['complete_drain'] == 1) &
            (self.df['is_high_risk_type'] == 1)
        ).astype(int)
        self.new_features.append('suspicious_pattern')
        
        logger.info(f"Created {5} anomaly detection features")
        
    def create_risk_score(self):
        """Create composite risk score"""
        logger.info("Creating composite risk score...")
        
        risk_components = []
        
        # Component 1: Transaction type risk (30%)
        if 'is_high_risk_type' in self.df.columns:
            risk_components.append(self.df['is_high_risk_type'] * 0.30)
        
        # Component 2: Amount risk (25%)
        if 'amount_percentile_in_type' in self.df.columns:
            risk_components.append(self.df['amount_percentile_in_type'] * 0.25)
        
        # Component 3: Balance pattern risk (20%)
        if 'complete_drain' in self.df.columns:
            risk_components.append(self.df['complete_drain'] * 0.20)
        
        # Component 4: Temporal risk (15%)
        if 'is_late_night' in self.df.columns:
            risk_components.append(self.df['is_late_night'] * 0.15)
        
        # Component 5: Anomaly risk (10%)
        if 'is_amount_outlier' in self.df.columns:
            risk_components.append(self.df['is_amount_outlier'] * 0.10)
        
        # Calculate composite risk score
        if risk_components:
            self.df['composite_risk_score'] = sum(risk_components)
            self.new_features.append('composite_risk_score')
            
            # Risk level categories
            self.df['risk_category'] = pd.cut(
                self.df['composite_risk_score'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['Low', 'Medium', 'High', 'Critical'],
                include_lowest=True
            )
        
        logger.info("Composite risk score created")
        
    def normalize_features(self):
        """Normalize numerical features"""
        logger.info("Normalizing numerical features...")
        
        # Select features to normalize
        features_to_normalize = [
            'amount', 'amount_log', 'oldbalanceOrg', 'newbalanceOrig',
            'balance_change', 'amount_to_balance_ratio'
        ]
        
        features_to_normalize = [f for f in features_to_normalize if f in self.df.columns]
        
        if features_to_normalize:
            scaler = StandardScaler()
            normalized_cols = [f + '_normalized' for f in features_to_normalize]
            
            self.df[normalized_cols] = scaler.fit_transform(self.df[features_to_normalize])
            self.new_features.extend(normalized_cols)
            
            logger.info(f"Normalized {len(features_to_normalize)} features")
        
    def get_feature_importance_proxy(self):
        """Calculate correlation-based feature importance"""
        logger.info("Calculating feature importance proxy...")
        
        if 'isFraud' not in self.df.columns:
            logger.warning("isFraud column not found. Skipping feature importance.")
            return None
        
        # Select only numeric features
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f != 'isFraud']
        
        # Calculate correlation with fraud
        correlations = self.df[numeric_features].corrwith(self.df['isFraud']).abs().sort_values(ascending=False)
        
        top_features = correlations.head(20)
        
        logger.info("\nTop 20 features by correlation with fraud:")
        for feature, corr in top_features.items():
            logger.info(f"  {feature}: {corr:.4f}")
        
        return correlations
        
    def save_engineered_data(self):
        """Save data with engineered features"""
        output_dir = Path(self.config['data']['processed_data'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "transactions_engineered.csv"
        self.df.to_csv(output_path, index=False)
        
        logger.info(f"Engineered dataset saved to {output_path}")
        logger.info(f"Total features: {len(self.df.columns)}")
        logger.info(f"New features created: {len(self.new_features)}")
        
        # Save feature list
        feature_list_path = output_dir / "feature_list.txt"
        with open(feature_list_path, 'w') as f:
            f.write("NEW FEATURES CREATED\n")
            f.write("="*80 + "\n\n")
            for i, feature in enumerate(self.new_features, 1):
                f.write(f"{i}. {feature}\n")
        
        logger.info(f"Feature list saved to {feature_list_path}")
        
        return output_path
    
    def run_feature_engineering(self):
        """Execute complete feature engineering pipeline"""
        logger.info("="*80)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("="*80)
        
        # Load data
        self.load_data()
        
        initial_features = len(self.df.columns)
        
        # Create features
        self.create_amount_features()
        self.create_balance_features()
        self.create_temporal_features()
        self.create_transaction_type_features()
        self.create_account_features()
        self.create_statistical_features()
        self.create_interaction_features()
        self.create_anomaly_features()
        self.create_risk_score()
        self.normalize_features()
        
        # Get feature importance
        self.get_feature_importance_proxy()
        
        # Save results
        self.save_engineered_data()
        
        final_features = len(self.df.columns)
        
        logger.info("="*80)
        logger.info("FEATURE ENGINEERING COMPLETED")
        logger.info(f"Initial features: {initial_features}")
        logger.info(f"Final features: {final_features}")
        logger.info(f"New features: {len(self.new_features)}")
        logger.info("="*80)
        
        return self.df


def main():
    """Main execution function"""
    engineer = FeatureEngineer()
    df_engineered = engineer.run_feature_engineering()
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*80)
    print(f"Total Features: {len(df_engineered.columns)}")
    print(f"New Features Created: {len(engineer.new_features)}")
    print(f"Dataset Shape: {df_engineered.shape}")
    print("\n✅ Feature engineering completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
