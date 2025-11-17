"""
Initial Exploratory Data Analysis
Competition: Playground Series S5E11
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


def load_data():
    """Load training and test datasets"""
    console.print("[bold blue]Loading data...[/bold blue]")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    console.print(f"Train shape: {train.shape}")
    console.print(f"Test shape: {test.shape}")

    return train, test


def basic_info(df, name="Dataset"):
    """Display basic dataset information"""
    console.print(f"\n[bold green]{name} Info[/bold green]")

    # Create info table
    table = Table(title=f"{name} Overview")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Rows", str(df.shape[0]))
    table.add_row("Columns", str(df.shape[1]))
    table.add_row("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    table.add_row("Duplicates", str(df.duplicated().sum()))

    console.print(table)

    # Data types
    console.print("\n[bold]Column Types:[/bold]")
    console.print(df.dtypes.value_counts())

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        console.print("\n[bold yellow]Missing Values:[/bold yellow]")
        console.print(missing[missing > 0])
    else:
        console.print("\n[bold green]No missing values![/bold green]")


def numeric_summary(df):
    """Summary statistics for numeric columns"""
    console.print("\n[bold green]Numeric Features Summary[/bold green]")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    console.print(df[numeric_cols].describe())


def categorical_summary(df):
    """Summary for categorical columns"""
    console.print("\n[bold green]Categorical Features Summary[/bold green]")
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        unique_count = df[col].nunique()
        console.print(f"\n[cyan]{col}[/cyan]: {unique_count} unique values")
        if unique_count <= 20:
            console.print(df[col].value_counts())


def main():
    """Main EDA workflow"""
    console.print("[bold magenta]Playground Series S5E11 - Initial EDA[/bold magenta]\n")

    # Load data
    train, test = load_data()

    # Basic info
    basic_info(train, "Training Set")
    basic_info(test, "Test Set")

    # Numeric summary
    numeric_summary(train)

    # Categorical summary
    categorical_summary(train)

    console.print("\n[bold green]✓ EDA Complete[/bold green]")


if __name__ == "__main__":
    main()
