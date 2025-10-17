"""
Fraud Analysis Module for PaySim Fraud Detection
Statistical and correlation analysis for fraud pattern detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from datetime import datetime
from loguru import logger
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Setup logging
log_dir = Path("outputs/logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(log_dir / "fraud_analysis_{time}.log", rotation="500 MB")


class FraudAnalyzer:
    """Comprehensive fraud pattern analysis"""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize FraudAnalyzer"""
        self.config = self._load_config(config_path)
        self.df = None
        self.analysis_results = {}
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def _load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found. Using defaults.")
            return {'data': {'processed_data': 'data/processed/'}}
    
    def load_cleaned_data(self, filepath=None):
        """Load cleaned transaction data"""
        if filepath is None:
            filepath = Path(self.config['data']['processed_data']) / "cleaned_transactions.parquet"
            if not filepath.exists():
                filepath = Path(self.config['data']['processed_data']) / "cleaned_transactions.csv"
        
        logger.info(f"Loading data from {filepath}")
        
        if str(filepath).endswith('.parquet'):
            self.df = pd.read_csv(str(filepath).replace('.parquet', '.csv'))
        else:
            self.df = pd.read_csv(filepath)
        
        logger.info(f"Loaded {len(self.df):,} transactions")
        return self.df
    
    def basic_fraud_statistics(self):
        """Calculate basic fraud statistics"""
        logger.info("Calculating basic fraud statistics...")
        
        total_transactions = len(self.df)
        fraud_transactions = self.df['isFraud'].sum()
        fraud_rate = (fraud_transactions / total_transactions) * 100
        
        total_amount = self.df['amount'].sum()
        fraud_amount = self.df[self.df['isFraud'] == 1]['amount'].sum()
        fraud_amount_pct = (fraud_amount / total_amount) * 100
        
        avg_transaction = self.df['amount'].mean()
        avg_fraud_transaction = self.df[self.df['isFraud'] == 1]['amount'].mean()
        avg_legit_transaction = self.df[self.df['isFraud'] == 0]['amount'].mean()
        
        stats_dict = {
            'total_transactions': int(total_transactions),
            'fraud_transactions': int(fraud_transactions),
            'legitimate_transactions': int(total_transactions - fraud_transactions),
            'fraud_rate': float(fraud_rate),
            'total_amount': float(total_amount),
            'fraud_amount': float(fraud_amount),
            'fraud_amount_percentage': float(fraud_amount_pct),
            'avg_transaction_amount': float(avg_transaction),
            'avg_fraud_amount': float(avg_fraud_transaction),
            'avg_legitimate_amount': float(avg_legit_transaction)
        }
        
        self.analysis_results['basic_statistics'] = stats_dict
        
        logger.info(f"Fraud Rate: {fraud_rate:.4f}%")
        logger.info(f"Fraud Amount: ${fraud_amount:,.2f} ({fraud_amount_pct:.2f}%)")
        
        return stats_dict
    
    def fraud_by_transaction_type(self):
        """Analyze fraud patterns by transaction type"""
        logger.info("Analyzing fraud by transaction type...")
        
        type_analysis = self.df.groupby('type').agg({
            'isFraud': ['count', 'sum', 'mean'],
            'amount': ['sum', 'mean', 'median']
        }).round(4)
        
        type_analysis.columns = ['_'.join(col).strip() for col in type_analysis.columns.values]
        type_analysis.rename(columns={
            'isFraud_count': 'total_transactions',
            'isFraud_sum': 'fraud_count',
            'isFraud_mean': 'fraud_rate',
            'amount_sum': 'total_amount',
            'amount_mean': 'avg_amount',
            'amount_median': 'median_amount'
        }, inplace=True)
        
        type_analysis['fraud_rate'] = type_analysis['fraud_rate'] * 100
        
        self.analysis_results['fraud_by_type'] = type_analysis.to_dict('index')
        
        logger.info("\nFraud Rate by Transaction Type:")
        for ttype, data in type_analysis.iterrows():
            logger.info(f"  {ttype}: {data['fraud_rate']:.4f}% ({int(data['fraud_count']):,} frauds)")
        
        return type_analysis
    
    def temporal_fraud_analysis(self):
        """Analyze fraud patterns over time"""
        logger.info("Analyzing temporal fraud patterns...")
        
        # Hourly analysis
        hourly_fraud = self.df.groupby('hour_of_day').agg({
            'isFraud': ['count', 'sum', 'mean'],
            'amount': 'sum'
        }).round(4)
        
        hourly_fraud.columns = ['total_trans', 'fraud_count', 'fraud_rate', 'total_amount']
        hourly_fraud['fraud_rate'] = hourly_fraud['fraud_rate'] * 100
        
        # Daily analysis
        daily_fraud = self.df.groupby('day').agg({
            'isFraud': ['count', 'sum', 'mean'],
            'amount': 'sum'
        }).round(4)
        
        daily_fraud.columns = ['total_trans', 'fraud_count', 'fraud_rate', 'total_amount']
        daily_fraud['fraud_rate'] = daily_fraud['fraud_rate'] * 100
        
        self.analysis_results['temporal_patterns'] = {
            'hourly': hourly_fraud.to_dict('index'),
            'daily': daily_fraud.to_dict('index')
        }
        
        # Find peak fraud hours
        peak_hour = hourly_fraud['fraud_rate'].idxmax()
        peak_fraud_rate = hourly_fraud['fraud_rate'].max()
        
        logger.info(f"Peak fraud hour: {peak_hour}:00 with {peak_fraud_rate:.4f}% fraud rate")
        
        return hourly_fraud, daily_fraud
    
    def amount_distribution_analysis(self):
        """Analyze amount distributions for fraud vs legitimate"""
        logger.info("Analyzing amount distributions...")
        
        fraud_amounts = self.df[self.df['isFraud'] == 1]['amount']
        legit_amounts = self.df[self.df['isFraud'] == 0]['amount']
        
        # Statistical comparison
        t_stat, p_value = stats.ttest_ind(fraud_amounts, legit_amounts)
        
        distribution_stats = {
            'fraud': {
                'count': int(len(fraud_amounts)),
                'mean': float(fraud_amounts.mean()),
                'median': float(fraud_amounts.median()),
                'std': float(fraud_amounts.std()),
                'min': float(fraud_amounts.min()),
                'max': float(fraud_amounts.max()),
                'percentile_25': float(fraud_amounts.quantile(0.25)),
                'percentile_75': float(fraud_amounts.quantile(0.75)),
                'percentile_95': float(fraud_amounts.quantile(0.95))
            },
            'legitimate': {
                'count': int(len(legit_amounts)),
                'mean': float(legit_amounts.mean()),
                'median': float(legit_amounts.median()),
                'std': float(legit_amounts.std()),
                'min': float(legit_amounts.min()),
                'max': float(legit_amounts.max()),
                'percentile_25': float(legit_amounts.quantile(0.25)),
                'percentile_75': float(legit_amounts.quantile(0.75)),
                'percentile_95': float(legit_amounts.quantile(0.95))
            },
            'statistical_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        }
        
        self.analysis_results['amount_distribution'] = distribution_stats
        
        logger.info(f"Fraud amounts - Mean: ${distribution_stats['fraud']['mean']:,.2f}, "
                   f"Median: ${distribution_stats['fraud']['median']:,.2f}")
        logger.info(f"Legitimate amounts - Mean: ${distribution_stats['legitimate']['mean']:,.2f}, "
                   f"Median: ${distribution_stats['legitimate']['median']:,.2f}")
        
        return distribution_stats
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        logger.info("Performing correlation analysis...")
        
        # Select numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns
        exclude_cols = ['step', 'day', 'hour_of_day']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Extract fraud correlations
        fraud_correlations = corr_matrix['isFraud'].sort_values(ascending=False)
        
        self.analysis_results['fraud_correlations'] = fraud_correlations.to_dict()
        
        logger.info("\nTop correlations with fraud:")
        for col, corr in fraud_correlations.head(10).items():
            if col != 'isFraud':
                logger.info(f"  {col}: {corr:.4f}")
        
        return corr_matrix, fraud_correlations
    
    def balance_pattern_analysis(self):
        """Analyze balance patterns in fraudulent transactions"""
        logger.info("Analyzing balance patterns...")
        
        fraud_df = self.df[self.df['isFraud'] == 1]
        legit_df = self.df[self.df['isFraud'] == 0]
        
        balance_patterns = {
            'fraud': {
                'zero_balance_before': int((fraud_df['oldbalanceOrg'] == 0).sum()),
                'zero_balance_after': int((fraud_df['newbalanceOrig'] == 0).sum()),
                'complete_drain': int((fraud_df['newbalanceOrig'] == 0).sum()),
                'avg_balance_before': float(fraud_df['oldbalanceOrg'].mean()),
                'avg_balance_after': float(fraud_df['newbalanceOrig'].mean())
            },
            'legitimate': {
                'zero_balance_before': int((legit_df['oldbalanceOrg'] == 0).sum()),
                'zero_balance_after': int((legit_df['newbalanceOrig'] == 0).sum()),
                'complete_drain': int((legit_df['newbalanceOrig'] == 0).sum()),
                'avg_balance_before': float(legit_df['oldbalanceOrg'].mean()),
                'avg_balance_after': float(legit_df['newbalanceOrig'].mean())
            }
        }
        
        self.analysis_results['balance_patterns'] = balance_patterns
        
        logger.info("Fraud transactions with complete balance drain: "
                   f"{balance_patterns['fraud']['complete_drain']:,}")
        
        return balance_patterns
    
    def high_risk_transaction_identification(self):
        """Identify high-risk transaction categories"""
        logger.info("Identifying high-risk transaction categories...")
        
        # Create risk score
        risk_factors = []
        
        # Factor 1: Transaction type (TRANSFER and CASH_OUT are riskier)
        risky_types = ['TRANSFER', 'CASH_OUT']
        self.df['risk_type'] = self.df['type'].isin(risky_types).astype(int)
        risk_factors.append('risk_type')
        
        # Factor 2: Complete balance drain
        self.df['risk_drain'] = (self.df['newbalanceOrig'] == 0).astype(int)
        risk_factors.append('risk_drain')
        
        # Factor 3: Large amount (top 10%)
        large_threshold = self.df['amount'].quantile(0.90)
        self.df['risk_large_amount'] = (self.df['amount'] > large_threshold).astype(int)
        risk_factors.append('risk_large_amount')
        
        # Factor 4: Destination is customer (not merchant)
        if 'dest_is_customer' in self.df.columns:
            self.df['risk_customer_dest'] = self.df['dest_is_customer']
            risk_factors.append('risk_customer_dest')
        
        # Calculate composite risk score
        self.df['risk_score'] = self.df[risk_factors].sum(axis=1) / len(risk_factors)
        
        # Categorize risk levels
        self.df['risk_level'] = pd.cut(
            self.df['risk_score'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical'],
            include_lowest=True
        )
        
        # Analyze fraud rate by risk level
        risk_analysis = self.df.groupby('risk_level').agg({
            'isFraud': ['count', 'sum', 'mean'],
            'amount': ['sum', 'mean']
        }).round(4)
        
        risk_analysis.columns = ['total_trans', 'fraud_count', 'fraud_rate', 'total_amount', 'avg_amount']
        risk_analysis['fraud_rate'] = risk_analysis['fraud_rate'] * 100
        
        self.analysis_results['risk_analysis'] = risk_analysis.to_dict('index')
        
        logger.info("\nFraud Rate by Risk Level:")
        for level, data in risk_analysis.iterrows():
            logger.info(f"  {level}: {data['fraud_rate']:.2f}% ({int(data['fraud_count']):,} frauds)")
        
        return risk_analysis
    
    def merchant_vs_customer_analysis(self):
        """Analyze fraud patterns: merchant vs customer destinations"""
        logger.info("Analyzing merchant vs customer patterns...")
        
        if 'dest_is_merchant' not in self.df.columns:
            logger.warning("Destination type columns not found. Skipping this analysis.")
            return None
        
        merchant_fraud = self.df[self.df['dest_is_merchant'] == 1]['isFraud'].mean() * 100
        customer_fraud = self.df[self.df['dest_is_customer'] == 1]['isFraud'].mean() * 100
        
        dest_analysis = {
            'merchant_destination': {
                'count': int((self.df['dest_is_merchant'] == 1).sum()),
                'fraud_rate': float(merchant_fraud)
            },
            'customer_destination': {
                'count': int((self.df['dest_is_customer'] == 1).sum()),
                'fraud_rate': float(customer_fraud)
            }
        }
        
        self.analysis_results['destination_analysis'] = dest_analysis
        
        logger.info(f"Fraud rate - Merchant destinations: {merchant_fraud:.4f}%")
        logger.info(f"Fraud rate - Customer destinations: {customer_fraud:.4f}%")
        
        return dest_analysis
    
    def flagged_fraud_effectiveness(self):
        """Analyze effectiveness of fraud flagging system"""
        logger.info("Analyzing fraud detection effectiveness...")
        
        # Confusion matrix components
        true_positives = ((self.df['isFraud'] == 1) & (self.df['isFlaggedFraud'] == 1)).sum()
        false_positives = ((self.df['isFraud'] == 0) & (self.df['isFlaggedFraud'] == 1)).sum()
        true_negatives = ((self.df['isFraud'] == 0) & (self.df['isFlaggedFraud'] == 0)).sum()
        false_negatives = ((self.df['isFraud'] == 1) & (self.df['isFlaggedFraud'] == 0)).sum()
        
        # Calculate metrics
        total_fraud = true_positives + false_negatives
        total_flagged = true_positives + false_positives
        
        detection_rate = (true_positives / total_fraud * 100) if total_fraud > 0 else 0
        precision = (true_positives / total_flagged * 100) if total_flagged > 0 else 0
        false_positive_rate = (false_positives / (false_positives + true_negatives) * 100)
        
        effectiveness = {
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives),
            'detection_rate': float(detection_rate),
            'precision': float(precision),
            'false_positive_rate': float(false_positive_rate)
        }
        
        self.analysis_results['fraud_detection_effectiveness'] = effectiveness
        
        logger.info(f"Detection Rate: {detection_rate:.2f}%")
        logger.info(f"Precision: {precision:.2f}%")
        logger.info(f"False Positive Rate: {false_positive_rate:.4f}%")
        
        return effectiveness
    
    def save_analysis_results(self):
        """Save analysis results to file"""
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as YAML
        yaml_path = output_dir / f"fraud_analysis_{timestamp}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(self.analysis_results, f, default_flow_style=False)
        
        logger.info(f"Analysis results saved to {yaml_path}")
        
        # Save enhanced dataset with risk scores
        processed_dir = Path(self.config['data']['processed_data'])
        enhanced_path = processed_dir / "transactions_with_risk_scores.csv"
        self.df.to_csv(enhanced_path, index=False)
        logger.info(f"Enhanced dataset saved to {enhanced_path}")
        
        return yaml_path
    
    def run_complete_analysis(self):
        """Execute complete fraud analysis pipeline"""
        logger.info("="*80)
        logger.info("STARTING FRAUD ANALYSIS")
        logger.info("="*80)
        
        # Load data
        self.load_cleaned_data()
        
        # Run all analyses
        self.basic_fraud_statistics()
        self.fraud_by_transaction_type()
        self.temporal_fraud_analysis()
        self.amount_distribution_analysis()
        self.correlation_analysis()
        self.balance_pattern_analysis()
        self.high_risk_transaction_identification()
        self.merchant_vs_customer_analysis()
        self.flagged_fraud_effectiveness()
        
        # Save results
        self.save_analysis_results()
        
        logger.info("="*80)
        logger.info("FRAUD ANALYSIS COMPLETED")
        logger.info("="*80)
        
        return self.analysis_results


def main():
    """Main execution function"""
    analyzer = FraudAnalyzer()
    results = analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("FRAUD ANALYSIS SUMMARY")
    print("="*80)
    
    if 'basic_statistics' in results:
        stats = results['basic_statistics']
        print(f"\nTotal Transactions: {stats['total_transactions']:,}")
        print(f"Fraud Transactions: {stats['fraud_transactions']:,}")
        print(f"Fraud Rate: {stats['fraud_rate']:.4f}%")
        print(f"Total Fraud Amount: ${stats['fraud_amount']:,.2f}")
    
    if 'fraud_detection_effectiveness' in results:
        eff = results['fraud_detection_effectiveness']
        print(f"\nDetection System Performance:")
        print(f"  Detection Rate: {eff['detection_rate']:.2f}%")
        print(f"  Precision: {eff['precision']:.2f}%")
        print(f"  False Positive Rate: {eff['false_positive_rate']:.4f}%")
    
    print("\n✅ Fraud analysis completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
