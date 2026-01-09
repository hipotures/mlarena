#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_contour,
    plot_parallel_coordinate
)

def main():
    parser = argparse.ArgumentParser(description="Visualize Optuna study from SQLite")
    parser.add_argument("--db", required=True, help="Path to SQLite file")
    parser.add_argument("--out-dir", default="./optuna_plots", help="Output directory for PNGs")
    parser.add_argument("--study", help="Study name (optional if only one exists)")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.exists():
        print(f"Error: DB file not found at {db_path}")
        sys.exit(1)

    storage = f"sqlite:///{db_path}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load study
        study = optuna.load_study(study_name=args.study, storage=storage)
        print(f"Loaded study: {study.study_name}")
        print(f"Trials: {len(study.trials)}")

        # 1. Optimization History
        print("Generating optimization history...")
        plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(out_dir / "optimization_history.png")
        plt.close()

        # 2. Param Importances
        print("Generating parameter importances...")
        try:
            from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
            import pandas as pd

            # Calculate importance numerically first
            importance = optuna.importance.get_param_importances(
                study, evaluator=MeanDecreaseImpurityImportanceEvaluator()
            )
            
            print("\nTop Parameter Importances (Numerical):")
            for name, val in list(importance.items())[:10]:
                print(f"  {name:30}: {val:.4f}")
            print("")

            plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
            plt.tight_layout()
            plt.savefig(out_dir / "param_importances.png")
            plt.close()
        except Exception as e:
            print(f"  Skipping importance plot: {e}")

        # 3. Slice Plot
        print("Generating slice plot...")
        plot_slice(study)
        plt.tight_layout()
        plt.savefig(out_dir / "slice.png")
        plt.close()

        # 4. Parallel Coordinate
        print("Generating parallel coordinate plot...")
        plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(out_dir / "parallel_coordinate.png")
        plt.close()

        print(f"\nDone! Plots saved in: {out_dir.absolute()}")
        print("Files:")
        for f in out_dir.glob("*.png"):
            print(f"  - {f.name}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
