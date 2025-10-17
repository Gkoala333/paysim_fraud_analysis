"""
PaySim Fraud Detection - Main Pipeline Orchestrator
Executes complete data processing and Tableau export pipeline
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess


def print_banner(title):
    """Print a formatted banner"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_script(script_path, script_name):
    """Run a Python script and handle errors"""
    print(f"→ Running {script_name}...")
    print("-"*80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=True
        )
        
        print("-"*80)
        print(f"✓ {script_name} completed successfully\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print("-"*80)
        print(f"✗ {script_name} failed with error code {e.returncode}\n")
        return False
    except Exception as e:
        print("-"*80)
        print(f"✗ {script_name} failed: {str(e)}\n")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    print_banner("CHECKING DEPENDENCIES")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name:15s} - installed")
        except ImportError:
            print(f"✗ {package_name:15s} - NOT INSTALLED")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ All required dependencies are installed")
    return True


def check_data_file():
    """Check if the raw data file exists"""
    print_banner("CHECKING DATA FILE")
    
    data_file = 'data/paysim_fraud.csv'
    
    if os.path.exists(data_file):
        file_size = os.path.getsize(data_file) / 1024**2  # MB
        print(f"✓ Data file found: {data_file}")
        print(f"✓ File size: {file_size:.2f} MB")
        
        # Count lines
        try:
            with open(data_file, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # Exclude header
            print(f"✓ Approximate transactions: {line_count:,}")
        except:
            pass
        
        return True
    else:
        print(f"✗ Data file not found: {data_file}")
        print("\nPlease ensure the data file is located at:")
        print(f"  {os.path.abspath(data_file)}")
        return False


def setup_environment():
    """Setup project directories and environment"""
    print_banner("SETTING UP ENVIRONMENT")
    
    directories = [
        'data/processed',
        'data/tableau_exports',
        'outputs/logs',
        'outputs/figures',
        'outputs/reports',
        'scripts',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory ready: {directory}")
    
    print("\n✓ Environment setup complete")
    return True


def run_data_cleaning():
    """Run the data cleaning script"""
    print_banner("STEP 1: DATA CLEANING")
    
    script_path = 'scripts/data_cleaning.py'
    
    if not os.path.exists(script_path):
        print(f"✗ Script not found: {script_path}")
        return False
    
    success = run_script(script_path, "Data Cleaning")
    
    if success:
        # Check if output was created
        output_file = 'data/processed/cleaned_transactions.csv'
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file) / 1024**2
            print(f"✓ Cleaned data saved: {output_file}")
            print(f"✓ File size: {file_size:.2f} MB\n")
        else:
            print(f"⚠️  Warning: Expected output file not found: {output_file}\n")
    
    return success


def run_tableau_export():
    """Run the Tableau export script"""
    print_banner("STEP 2: TABLEAU EXPORT")
    
    script_path = 'scripts/export_for_tableau.py'
    
    if not os.path.exists(script_path):
        print(f"✗ Script not found: {script_path}")
        return False
    
    success = run_script(script_path, "Tableau Export")
    
    if success:
        # Check exports
        export_dir = 'data/tableau_exports'
        expected_files = [
            'fraud_summary_by_type.csv',
            'regional_fraud_summary.csv'
        ]
        
        print("✓ Checking exported files:")
        for filename in expected_files:
            filepath = os.path.join(export_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / 1024
                print(f"  ✓ {filename:35s} ({file_size:.2f} KB)")
            else:
                print(f"  ✗ {filename:35s} (NOT FOUND)")
        print()
    
    return success


def display_final_summary(start_time, end_time):
    """Display final execution summary"""
    print_banner("EXECUTION SUMMARY")
    
    duration = (end_time - start_time).total_seconds()
    
    print(f"Start time:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:       {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    print("\n" + "-"*80)
    print("OUTPUT LOCATIONS")
    print("-"*80)
    
    # List all generated files
    outputs = {
        "Cleaned Data": ["data/processed/cleaned_transactions.csv"],
        "Tableau Exports": [
            "data/tableau_exports/fraud_summary_by_type.csv",
            "data/tableau_exports/regional_fraud_summary.csv"
        ],
        "Logs": [
            "outputs/logs/data_cleaning_log.txt",
            "outputs/logs/export_log.txt"
        ]
    }
    
    for category, files in outputs.items():
        print(f"\n{category}:")
        for filepath in files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024
                if size > 1024:
                    size_str = f"{size/1024:.2f} MB"
                else:
                    size_str = f"{size:.2f} KB"
                print(f"  ✓ {filepath:50s} ({size_str})")
            else:
                print(f"  ✗ {filepath:50s} (not found)")
    
    print("\n" + "-"*80)
    print("NEXT STEPS")
    print("-"*80)
    print("""
1. Open Tableau Desktop or Tableau Public
2. Connect to Text File data source
3. Navigate to: data/tableau_exports/
4. Import the CSV files:
   • fraud_summary_by_type.csv
   • regional_fraud_summary.csv
5. Create dashboards with:
   • Transaction volume by type
   • Fraud rates across categories
   • Regional fraud patterns
   • High-risk transaction analysis

6. For detailed analysis, you can also explore:
   • outputs/logs/ - Processing logs
   • data/processed/ - Cleaned full dataset
    """)


def main():
    """Main pipeline orchestrator"""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "     PaySim Fraud Detection - Complete Pipeline Execution".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    start_time = datetime.now()
    
    # Pre-flight checks
    if not check_dependencies():
        print("\n✗ Pipeline aborted: Missing dependencies")
        sys.exit(1)
    
    if not check_data_file():
        print("\n✗ Pipeline aborted: Data file not found")
        sys.exit(1)
    
    if not setup_environment():
        print("\n✗ Pipeline aborted: Environment setup failed")
        sys.exit(1)
    
    # Execute pipeline steps
    steps_success = {
        'data_cleaning': False,
        'tableau_export': False
    }
    
    # Step 1: Data Cleaning
    steps_success['data_cleaning'] = run_data_cleaning()
    
    if not steps_success['data_cleaning']:
        print("\n⚠️  Warning: Data cleaning failed. Attempting to continue with existing cleaned data...")
        
        # Check if cleaned data already exists
        if not os.path.exists('data/processed/cleaned_transactions.csv'):
            print("✗ No existing cleaned data found. Pipeline cannot continue.")
            sys.exit(1)
        else:
            print("✓ Found existing cleaned data. Continuing...")
    
    # Step 2: Tableau Export
    steps_success['tableau_export'] = run_tableau_export()
    
    end_time = datetime.now()
    
    # Display final summary
    display_final_summary(start_time, end_time)
    
    # Final status
    if all(steps_success.values()):
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("⚠️  PIPELINE COMPLETED WITH WARNINGS")
        print("="*80)
        print("\nSteps status:")
        for step, success in steps_success.items():
            status = "✓" if success else "✗"
            print(f"  {status} {step.replace('_', ' ').title()}")
        print("\nCheck logs for details:")
        print("  • outputs/logs/data_cleaning_log.txt")
        print("  • outputs/logs/export_log.txt\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
