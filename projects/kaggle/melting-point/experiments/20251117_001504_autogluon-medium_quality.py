"""
Baseline AutoGluon Model
Competition: Thermophysical Property - Melting Point
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH,
    TARGET_COLUMN, RANDOM_SEED, AUTOGLUON_TIME_LIMIT,
    AUTOGLUON_PRESET, AUTOGLUON_PROBLEM_TYPE, AUTOGLUON_EVAL_METRIC,
    PROJECT_ROOT
)
from utils.submission import create_submission

from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def load_data():
    """Load train and test data"""
    console.print("[bold blue]Loading data...[/bold blue]")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    console.print(f"Train shape: {train.shape}")
    console.print(f"Test shape: {test.shape}")
    console.print(f"Target distribution:\n{train[TARGET_COLUMN].value_counts(normalize=True)}")

    return train, test


def train_autogluon(train_df, preset="medium_quality", time_limit=10):
    """
    Train AutoGluon model

    Args:
        train_df: Training dataframe with target column
        preset: AutoGluon preset quality
        time_limit: Training time limit in seconds

    Returns:
        Trained TabularPredictor
    """
    console.print(f"\n[bold green]Training AutoGluon ({preset}, {time_limit}s)[/bold green]")

    # Create predictor
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        problem_type=AUTOGLUON_PROBLEM_TYPE,
        eval_metric=AUTOGLUON_EVAL_METRIC,
        path=str(PROJECT_ROOT / "AutogluonModels"),
        verbosity=2
    )

    # Drop ID column
    train_data = train_df.drop('id', axis=1)

    # Train
    start_time = datetime.now()
    predictor.fit(
        train_data,
        presets=preset,
        time_limit=time_limit,
        num_cpus='auto',
        num_gpus=1
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    console.print(f"[green]✓ Training completed in {elapsed:.0f}s[/green]")

    return predictor


def evaluate_model(predictor, train_df):
    """Evaluate model on training data with cross-validation"""
    console.print("\n[bold blue]Model Evaluation[/bold blue]")

    # Get leaderboard
    leaderboard = predictor.leaderboard(train_df.drop('id', axis=1), silent=True)

    # Display top models
    console.print("\n[bold]Top Models:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Time", justify="right")

    for idx, row in leaderboard.head(10).iterrows():
        table.add_row(
            row['model'],
            f"{row['score_val']:.5f}",
            f"{row['fit_time']:.1f}s"
        )

    console.print(table)

    # Best model info
    best_model = leaderboard.iloc[0]
    console.print(f"\n[bold green]Best Model: {best_model['model']}[/bold green]")
    console.print(f"Score: {best_model['score_val']:.5f}")

    return best_model['score_val']


def make_predictions(predictor, test_df):
    """Make predictions on test set"""
    console.print("\n[bold blue]Making predictions...[/bold blue]")

    # Drop ID for prediction
    test_data = test_df.drop('id', axis=1)

    # Get predictions (regression uses predict, not predict_proba)
    if predictor.can_predict_proba:
        # Classification
        predictions = predictor.predict_proba(test_data, as_multiclass=False)
    else:
        # Regression
        predictions = predictor.predict(test_data)

    console.print(f"Predictions shape: {predictions.shape}")
    console.print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    console.print(f"Predictions mean: {predictions.mean():.4f}")

    return predictions


def main():
    """Main training workflow"""
    console.print(Panel.fit(
        "[bold magenta]AutoGluon Baseline Model[/bold magenta]\n"
        f"Preset: {AUTOGLUON_PRESET}\n"
        f"Time Limit: {AUTOGLUON_TIME_LIMIT}s",
        title="Playground Series S5E11"
    ))

    # Load data
    train, test = load_data()

    # Train model
    predictor = train_autogluon(
        train,
        preset=AUTOGLUON_PRESET,
        time_limit=AUTOGLUON_TIME_LIMIT
    )

    # Evaluate
    cv_score = evaluate_model(predictor, train)

    # Make predictions
    predictions = make_predictions(predictor, test)

    # Create submission
    console.print("\n[bold blue]Creating submission...[/bold blue]")

    submission_path = create_submission(
        predictions=predictions,
        test_ids=test['id'],
        model_name=f"autogluon-{AUTOGLUON_PRESET}",
        local_cv_score=cv_score,
        notes=f"Baseline AutoGluon {AUTOGLUON_PRESET}, {AUTOGLUON_TIME_LIMIT}s",
        config={
            'preset': AUTOGLUON_PRESET,
            'time_limit': AUTOGLUON_TIME_LIMIT,
            'eval_metric': AUTOGLUON_EVAL_METRIC
        }
    )

    console.print(f"\n[bold green]✓ Complete![/bold green]")
    console.print(f"Submission: {submission_path}")
    console.print(f"Local CV Score: {cv_score:.5f}")

    # Display next steps
    console.print(Panel.fit(
        f"[bold]Next Steps:[/bold]\n\n"
        f"1. Submit to Kaggle:\n"
        f"   cd submissions/\n"
        f"   kaggle competitions submit -c playground-series-s5e11 \\\n"
        f"       -f {submission_path.name} \\\n"
        f"       -m 'AutoGluon {AUTOGLUON_PRESET} baseline'\n\n"
        f"2. Update public score:\n"
        f"   python ../tools/submissions_tracker.py --project playground-series-s5e11 \\\n"
        f"       update 1 --public <SCORE>\n\n"
        f"3. View submissions:\n"
        f"   python ../tools/submissions_tracker.py --project playground-series-s5e11 list",
        title="📊 Next Steps"
    ))


if __name__ == "__main__":
    main()
