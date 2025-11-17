"""
Baseline AutoGluon Model
Competition: Thermophysical Property - Melting Point
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (  # noqa: E402
    AUTOGLUON_EVAL_METRIC,
    AUTOGLUON_PRESET,
    AUTOGLUON_PROBLEM_TYPE,
    AUTOGLUON_TIME_LIMIT,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)
from utils.submission import create_submission  # noqa: E402

TOOLS_DIR = PROJECT_ROOT.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from submission_workflow import SubmissionRunner  # noqa: E402

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


def parse_args():
    parser = argparse.ArgumentParser(description="AutoGluon baseline for melting-point")
    parser.add_argument("--skip-submit", action="store_true", help="Skip Kaggle submission workflow")
    parser.add_argument("--auto-submit", action="store_true", help="Submit without asking for confirmation")
    parser.add_argument("--kaggle-message", help="Custom Kaggle submission message")
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=30,
        help="Time (seconds) to wait before scraping the Kaggle submissions page",
    )
    parser.add_argument(
        "--cdp-url",
        default="http://localhost:9222",
        help="Playwright CDP URL used for scraping scores",
    )
    parser.add_argument(
        "--skip-score-fetch",
        action="store_true",
        help="Do not run Playwright scraping for the latest score",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Do not stage/commit git changes automatically",
    )
    return parser.parse_args()


def main(args):
    """Main training workflow"""
    console.print(Panel.fit(
        "[bold magenta]AutoGluon Baseline Model[/bold magenta]\n"
        f"Preset: {AUTOGLUON_PRESET}\n"
        f"Time Limit: {AUTOGLUON_TIME_LIMIT}s",
        title="Melting Point"
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

    submission_artifact = create_submission(
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
    console.print(f"Submission: {submission_artifact.path}")
    console.print(f"Local CV Score: {cv_score:.5f}")

    if args.skip_submit:
        console.print("[yellow]Skipping Kaggle submission workflow (--skip-submit).[/yellow]")
    else:
        runner = SubmissionRunner(
            artifact=submission_artifact,
            kaggle_message=args.kaggle_message or f"autogluon-{AUTOGLUON_PRESET} | local {cv_score:.5f}",
            wait_seconds=args.wait_seconds,
            cdp_url=args.cdp_url,
            auto_submit=args.auto_submit,
            prompt=not args.auto_submit,
            skip_browser=args.skip_score_fetch,
            skip_git=args.skip_git,
        )
        runner.execute()

    console.print(Panel.fit(
        "[bold]Tip:[/bold] Use --auto-submit to automate the pipeline or "
        "--skip-submit to only regenerate predictions.",
        title="Workflow",
        border_style="cyan"
    ))


if __name__ == "__main__":
    main(parse_args())
