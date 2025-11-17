#!/usr/bin/env python3
"""
Analyze column types using ydata-profiling for the S5E7 dataset.
"""

import pandas as pd
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import json
from pathlib import Path


def analyze_dataset_types(csv_path: str):
    """Analyze column types in dataset using ydata-profiling."""
    
    # Load dataset
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Create profile report with custom config
    print("Creating profile report...")
    
    # Create custom settings
    settings = Settings()
    settings.vars.num.low_categorical_threshold = 25
    
    profile = ProfileReport(
        df, 
        title="S5E7 Dataset Type Analysis",
        minimal=True,  # Faster analysis
        explorative=False,  # Focus on types, not full exploration
        config=settings
    )
    
    # Get variable descriptions - API changed in newer versions
    description = profile.description_set
    if hasattr(description, 'variables'):
        variables = description.variables
    else:
        # Try alternative approach
        variables = profile.to_json()
        import json
        data = json.loads(variables)
        variables = data.get('variables', {})
    
    # Extract column types
    column_types = {}
    for col_name, col_info in variables.items():
        column_types[col_name] = {
            'ydata_type': col_info['type'],
            'pandas_dtype': str(df[col_name].dtype),
            'unique_count': df[col_name].nunique(),
            'null_count': df[col_name].isnull().sum(),
            'sample_values': [str(v) for v in df[col_name].dropna().head(5).tolist()]
        }
    
    return column_types, df


def print_analysis(column_types, df):
    """Print analysis results in a formatted way."""
    
    print("\n" + "="*80)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("="*80)
    
    print("\nColumn Type Analysis:")
    print("-"*80)
    
    for col_name, info in column_types.items():
        print(f"\nColumn: {col_name}")
        print(f"  YData Type: {info['ydata_type']}")
        print(f"  Pandas Dtype: {info['pandas_dtype']}")
        print(f"  Unique Values: {info['unique_count']}")
        print(f"  Null Count: {info['null_count']}")
        print(f"  Sample Values: {info['sample_values'][:5]}")
        
        # Additional analysis for numeric columns
        if info['pandas_dtype'] in ['int64', 'float64']:
            unique_ratio = info['unique_count'] / len(df)
            print(f"  Unique Ratio: {unique_ratio:.2%}")
            
            # Check if it might be categorical
            if info['unique_count'] < 20:
                print(f"  ⚠️  Low unique count for numeric column - might be categorical!")
                unique_vals = sorted(df[col_name].dropna().unique())[:10]
                print(f"  Unique values: {unique_vals}")


def save_results(column_types, output_path):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(column_types, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    # Paths
    dataset_path = "../datasets/playground-series-s5e7/train.csv"
    output_path = "column_types_analysis.json"
    
    # Check if file exists
    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Analyze
    try:
        column_types, df = analyze_dataset_types(dataset_path)
        print_analysis(column_types, df)
        save_results(column_types, output_path)
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        
        type_counts = {}
        for col_info in column_types.values():
            ydata_type = col_info['ydata_type']
            type_counts[ydata_type] = type_counts.get(ydata_type, 0) + 1
        
        print("\nColumn types distribution:")
        for type_name, count in sorted(type_counts.items()):
            print(f"  {type_name}: {count} columns")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()