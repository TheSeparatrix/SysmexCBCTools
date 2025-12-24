#!/usr/bin/env python3
import sys

import pandas as pd


def detect_numeric_strings(series):
    """Check if a string column could be numeric"""
    if series.dtype == 'object':
        # Try to convert to numeric, coerce errors to NaN
        numeric_attempt = pd.to_numeric(series, errors='coerce')
        # If most non-null values converted successfully, it's likely numeric
        non_null_count = series.notna().sum()
        if non_null_count > 0:
            successful_conversions = numeric_attempt.notna().sum()
            if successful_conversions / non_null_count > 0.9:  # 90% threshold
                return True, 'numeric (stored as string)'
    return False, None


def detect_datetime_strings(series):
    """Check if a string column could be datetime"""
    if series.dtype == 'object':
        # Sample a few values to test
        sample = series.dropna().head(min(100, len(series)))
        if len(sample) > 0:
            try:
                # Try to parse as datetime
                pd.to_datetime(sample, errors='raise')
                return True, 'datetime (stored as string)'
            except:
                pass
    return False, None


def analyse_csv(filepath, output_file=None):
    """Analyse a CSV file and print comprehensive information"""

    # Create output string that will be both printed and optionally saved
    output = []

    output.append(f"\n{'='*80}")
    output.append(f"CSV ANALYSIS: {filepath}")
    output.append(f"{'='*80}\n")

    # Load CSV (max 100k rows)
    try:
        df = pd.read_csv(filepath, nrows=100000)
    except Exception as e:
        error_msg = f"Error reading CSV file: {e}"
        print(error_msg)
        return

    # 1. BASIC SHAPE INFORMATION
    output.append(f"{'='*80}")
    output.append("BASIC INFORMATION")
    output.append(f"{'='*80}")
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    output.append(f"Rows: {total_rows:,}")
    output.append(f"Columns: {total_cols}")
    if total_rows == 100000:
        output.append("⚠️  Note: Only first 100,000 rows loaded")

    # 3. MEMORY USAGE
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    output.append(f"Memory usage: {memory_mb:.2f} MB")

    # 4. DUPLICATE ROWS
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / total_rows * 100) if total_rows > 0 else 0
    output.append(f"Duplicate rows: {duplicate_count:,} ({duplicate_pct:.2f}%)")

    # 2. MISSING VALUES ANALYSIS
    output.append(f"\n{'='*80}")
    output.append("MISSING VALUES")
    output.append(f"{'='*80}")
    missing_data = pd.DataFrame({
        'Missing Count': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / total_rows * 100).round(2)
    })
    missing_data = missing_data[missing_data['Missing Count'] > 0]

    if len(missing_data) > 0:
        output.append(missing_data.to_string())
    else:
        output.append("No missing values found")

    # COLUMN INFORMATION WITH TYPE DETECTION
    output.append(f"\n{'='*80}")
    output.append("COLUMN INFORMATION")
    output.append(f"{'='*80}\n")

    for col in df.columns:
        output.append(f"Column: '{col}'")
        output.append(f"  Datatype: {df[col].dtype}")

        # Check if numeric string
        is_numeric_str, numeric_type = detect_numeric_strings(df[col])
        if is_numeric_str:
            output.append(f"  ⚠️  Could be converted to: {numeric_type}")

        # Check if datetime string
        is_datetime_str, datetime_type = detect_datetime_strings(df[col])
        if is_datetime_str:
            output.append(f"  ⚠️  Could be converted to: {datetime_type}")

        # Unique values
        unique_count = df[col].nunique()
        output.append(f"  Unique values: {unique_count:,}")

        # Show unique values if ≤ 10
        if unique_count <= 10:
            unique_vals = df[col].unique()
            output.append(f"  All unique values: {list(unique_vals)}")

        # Show percentage of unique values for high cardinality columns
        if unique_count > 10:
            unique_pct = (unique_count / total_rows * 100)
            output.append(f"  Unique percentage: {unique_pct:.2f}%")

        output.append("")

    output.append(f"{'='*80}")
    output.append("ANALYSIS COMPLETE")
    output.append(f"{'='*80}\n")

    # Join all output lines and print
    final_output = '\n'.join(output)
    print(final_output)

    # Optionally save to file
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(final_output)
            print(f"Analysis saved to: {output_file}")
        except Exception as e:
            print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python analyse_csv.py <file.csv> [output.txt]")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) == 3 else None

    analyse_csv(csv_file, output_file)
