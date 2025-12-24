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


def get_basic_info(df, filepath, file_label):
    """Get basic information about a CSV file"""
    output = []

    output.append(f"\n{'='*80}")
    output.append(f"{file_label}: {filepath}")
    output.append(f"{'='*80}\n")

    # BASIC SHAPE INFORMATION
    output.append(f"{'='*80}")
    output.append("BASIC INFORMATION")
    output.append(f"{'='*80}")
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    output.append(f"Rows: {total_rows:,}")
    output.append(f"Columns: {total_cols}")
    if total_rows == 100000:
        output.append("⚠️  Note: Only first 100,000 rows loaded")

    # MEMORY USAGE
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    output.append(f"Memory usage: {memory_mb:.2f} MB")

    # DUPLICATE ROWS
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / total_rows * 100) if total_rows > 0 else 0
    output.append(f"Duplicate rows: {duplicate_count:,} ({duplicate_pct:.2f}%)")

    # MISSING VALUES ANALYSIS
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

    return output


def get_column_info(df, col):
    """Get detailed information about a single column"""
    info = []
    total_rows = df.shape[0]

    info.append(f"  Datatype: {df[col].dtype}")

    # Check if numeric string
    is_numeric_str, numeric_type = detect_numeric_strings(df[col])
    if is_numeric_str:
        info.append(f"  ⚠️  Could be converted to: {numeric_type}")

    # Check if datetime string
    is_datetime_str, datetime_type = detect_datetime_strings(df[col])
    if is_datetime_str:
        info.append(f"  ⚠️  Could be converted to: {datetime_type}")

    # Unique values
    unique_count = df[col].nunique()
    info.append(f"  Unique values: {unique_count:,}")

    # Show unique values if ≤ 10
    if unique_count <= 10:
        unique_vals = df[col].unique()
        info.append(f"  All unique values: {list(unique_vals)}")

    # Show percentage of unique values for high cardinality columns
    if unique_count > 10:
        unique_pct = (unique_count / total_rows * 100)
        info.append(f"  Unique percentage: {unique_pct:.2f}%")

    return info


def compare_csv_files(filepath1, filepath2, output_file=None):
    """Compare two CSV files and print comprehensive comparison"""

    output = []

    output.append(f"\n{'='*80}")
    output.append("CSV COMPARISON")
    output.append(f"{'='*80}\n")

    # Load CSV files (max 100k rows each)
    try:
        df1 = pd.read_csv(filepath1, nrows=100000)
    except Exception as e:
        print(f"Error reading first CSV file: {e}")
        return

    try:
        df2 = pd.read_csv(filepath2, nrows=100000)
    except Exception as e:
        print(f"Error reading second CSV file: {e}")
        return

    # Get basic information for both files
    output.extend(get_basic_info(df1, filepath1, "CSV FILE 1"))
    output.extend(get_basic_info(df2, filepath2, "CSV FILE 2"))

    # COLUMN COMPARISON
    output.append(f"\n{'='*80}")
    output.append("COLUMN COMPARISON")
    output.append(f"{'='*80}\n")

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    common_cols = cols1.intersection(cols2)
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1

    output.append(f"Columns in both files: {len(common_cols)}")
    output.append(f"Columns only in file 1: {len(only_in_1)}")
    output.append(f"Columns only in file 2: {len(only_in_2)}")

    if only_in_1:
        output.append("\nColumns only in file 1:")
        for col in sorted(only_in_1):
            output.append(f"  - '{col}'")

    if only_in_2:
        output.append("\nColumns only in file 2:")
        for col in sorted(only_in_2):
            output.append(f"  - '{col}'")

    # DETAILED COMPARISON OF COMMON COLUMNS
    if common_cols:
        output.append(f"\n{'='*80}")
        output.append("DETAILED COMPARISON OF COMMON COLUMNS")
        output.append(f"{'='*80}\n")

        for col in sorted(common_cols):
            output.append(f"Column: '{col}'")
            output.append(f"{'-'*80}")

            # File 1 info
            output.append("FILE 1:")
            col_info_1 = get_column_info(df1, col)
            output.extend(col_info_1)

            output.append("")

            # File 2 info
            output.append("FILE 2:")
            col_info_2 = get_column_info(df2, col)
            output.extend(col_info_2)

            output.append("")

    output.append(f"{'='*80}")
    output.append("COMPARISON COMPLETE")
    output.append(f"{'='*80}\n")

    # Join all output lines and print
    final_output = '\n'.join(output)
    print(final_output)

    # Optionally save to file
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(final_output)
            print(f"Comparison saved to: {output_file}")
        except Exception as e:
            print(f"Error writing to output file: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python compare_csv.py <file1.csv> <file2.csv> [output.txt]")
        sys.exit(1)

    csv_file1 = sys.argv[1]
    csv_file2 = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) == 4 else None

    compare_csv_files(csv_file1, csv_file2, output_file)
