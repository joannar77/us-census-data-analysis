#!/usr/bin/env python3
"""
Census Cleaning & Viz Pipeline
Author: Joanna Ronchi
Repo: <your-repo-url>

Usage:
  python census_pipeline.py --data_path ./data --pattern "states*.csv" --out_dir ./figures
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, StrMethodFormatter

# Race percentage columns
RACE_COLS = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']


# ---------------------------
# Loading & Cleaning Helpers
# ---------------------------

def load_data(data_path: str, pattern: str) -> pd.DataFrame:
    """Load all CSVs matching pattern from data_path and vertically concatenate."""
    files = sorted(glob.glob(os.path.join(data_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern!r} in {data_path!r}")
    print(f"[load] Found {len(files)} files")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] Combined shape: {df.shape}")
    return df


def clean_income(df: pd.DataFrame) -> pd.DataFrame:
    """Remove $ and commas from Income and convert to numeric."""
    if 'Income' in df.columns:
        df['Income'] = (
            df['Income']
            .astype(str)
            .str.replace('$', '', regex=False)
            .str.replace(',', '', regex=False)
        )
        df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    else:
        print("[warn] 'Income' column not found; skipping income cleaning.")
    return df


def split_genderpop(df: pd.DataFrame) -> pd.DataFrame:
    """Split GenderPop into Men and Women numeric columns."""
    if 'GenderPop' in df.columns:
        # Expect values like "2341093M_2341091F"
        split = df['GenderPop'].astype(str).str.split('_', expand=True)
        # Guard in case split fails for some rows
        if split.shape[1] >= 1:
            df['Men'] = pd.to_numeric(
                split[0].astype(str).str.replace('M', '', regex=False),
                errors='coerce'
            )
        if split.shape[1] >= 2:
            df['Women'] = pd.to_numeric(
                split[1].astype(str).str.replace('F', '', regex=False),
                errors='coerce'
            )
    else:
        print("[warn] 'GenderPop' column not found; skipping split to Men/Women.")
    return df


def clean_race_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip '%' and convert race columns to numeric."""
    for col in RACE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace('%', '', regex=False),
                errors='coerce'
            )
        else:
            print(f"[warn] Race column '{col}' not found; skipping.")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs logically: Women = TotalPop - Men; races with column mean."""
    # Women = TotalPop - Men when Women is NaN and the others are present
    if all(c in df.columns for c in ['TotalPop', 'Men', 'Women']):
        mask = df['Women'].isna() & df['TotalPop'].notna() & df['Men'].notna()
        filled = (df.loc[mask, 'TotalPop'] - df.loc[mask, 'Men']).astype('float')
        df.loc[mask, 'Women'] = filled
        print(f"[fill] Filled {mask.sum()} Women values using TotalPop - Men")
    else:
        print("[warn] Missing one of ['TotalPop','Men','Women']; Women fill skipped.")

    # Fill race columns with column means
    for col in RACE_COLS:
        if col in df.columns:
            n_before = df[col].isna().sum()
            if n_before > 0:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                print(f"[fill] '{col}': filled {n_before} NaNs with mean={mean_val:.2f}")
    return df


def drop_dupes(df: pd.DataFrame) -> pd.DataFrame:
    """Drop complete duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"[dedupe] Dropped {before - after} duplicate rows; new shape: {df.shape}")
    return df


def quick_data_quality_report(df: pd.DataFrame):
    """Print a simple data quality report."""
    print("\n=== Data Quality Report ===")
    print("Columns:", list(df.columns))
    print("\nNulls per column:\n", df.isna().sum())
    print("\ndtypes:\n", df.dtypes)
    print("===========================\n")


# ---------------------------
# Plotting
# ---------------------------

def plot_scatter_income_women(df: pd.DataFrame, out_dir: str):
    """Scatter: Income vs Women with axis formatting and save to file."""
    if not {'Income', 'Women'}.issubset(df.columns):
        print("[plot] Skipping Income vs Women (columns missing).")
        return

    plt.scatter(df['Income'], df['Women'], alpha=0.5)
    plt.xlabel("Median Income ($)")
    plt.ylabel("Female Population (millions)")
    plt.title("Median Income vs Female Population by State (US Census)")

    ax = plt.gca()
    # y in millions (e.g., 2M)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y/1_000_000)}M"))
    # x as dollars, no decimals
    ax.xaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "scatter_income_vs_women.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[plot] Saved {out}")


def plot_race_histograms(df: pd.DataFrame, out_dir: str):
    """Create and save a histogram for each race column."""
    os.makedirs(out_dir, exist_ok=True)
    for col in RACE_COLS:
        if col not in df.columns:
            continue
        plt.hist(df[col], bins=20, alpha=0.7)
        plt.title(f"Distribution of {col} Population (%)")
        plt.xlabel(f"{col} (%)")
        plt.ylabel("Number of States")
        plt.tight_layout()
        out = os.path.join(out_dir, f"hist_{col.lower()}.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[plot] Saved {out}")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean US Census data and create plots.")
    parser.add_argument("--data_path", default=".", help="Folder containing states*.csv")
    parser.add_argument("--pattern", default="states*.csv", help="Glob pattern for CSVs")
    parser.add_argument("--out_dir", default="./figures", help="Where to save figures & cleaned CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.data_path, args.pattern)
    df = clean_income(df)
    df = split_genderpop(df)
    df = clean_race_columns(df)

    # Pre-fill NaN audit (nice for logs)
    pre_nan = df.isna().sum().sum()
    print(f"[audit] Total NaNs before fill: {pre_nan}")

    df = fill_missing(df)
    df = drop_dupes(df)

    quick_data_quality_report(df)

    plot_scatter_income_women(df, args.out_dir)
    plot_race_histograms(df, args.out_dir)

    # Save cleaned dataset for reproducibility
    cleaned_path = os.path.join(args.out_dir, "us_census_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"[data] Saved cleaned data -> {cleaned_path}")


if __name__ == "__main__":
    main()
