"""
Optuna study lifecycle management with SQLite storage.

This module manages creation, loading, and resumption of Optuna studies
with persistent storage in SQLite databases.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner

from kaggle_tools.config_models import OptunaConfig


class StudyManager:
    """
    Manages Optuna study creation, resumption, and storage.

    Features:
    - SQLite persistent storage (enables resume after interruption)
    - Optuna-dashboard compatible (web UI visualization)
    - Automatic sampler/pruner configuration
    - Trial history export

    Example:
        >>> config = OptunaConfig(
        ...     storage="sqlite:///experiments/exp-001/artifacts/optuna/study.db",
        ...     study_name="exp-001_xgboost",
        ...     direction="maximize",
        ...     n_trials=100,
        ...     sampler="TPESampler",
        ...     pruner="MedianPruner",
        ... )
        >>> manager = StudyManager(config)
        >>> study = manager.create_or_load_study()
        >>>
        >>> # Optimize
        >>> study.optimize(objective, n_trials=100)
        >>>
        >>> # Get best parameters
        >>> best_params = manager.get_best_params()
        >>>
        >>> # Export trials for analysis
        >>> manager.export_trials_dataframe(Path("trials.csv"))
    """

    def __init__(self, config: OptunaConfig):
        """
        Initialize study manager.

        Args:
            config: Optuna configuration
        """
        self.config = config
        self.study: Optional[optuna.Study] = None

    def create_or_load_study(self) -> optuna.Study:
        """
        Create new study or load existing from storage.

        If a study with the same name already exists in the storage,
        it will be loaded and can be resumed.

        Returns:
            Optuna Study object

        Raises:
            ValueError: If sampler or pruner name is invalid
        """
        storage = self.config.storage
        study_name = self.config.study_name

        # Create sampler and pruner
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        # Create or load study
        self.study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,  # Resume interrupted studies
        )

        return self.study

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """
        Instantiate sampler from configuration.

        Returns:
            Optuna sampler instance

        Raises:
            ValueError: If sampler name is not recognized
        """
        sampler_name = self.config.sampler

        if sampler_name == "TPESampler":
            return TPESampler()
        elif sampler_name == "RandomSampler":
            return RandomSampler()
        elif sampler_name == "CmaEsSampler":
            return CmaEsSampler()
        else:
            raise ValueError(
                f"Unknown sampler: {sampler_name}. "
                f"Supported: TPESampler, RandomSampler, CmaEsSampler"
            )

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """
        Instantiate pruner from configuration.

        Returns:
            Optuna pruner instance

        Raises:
            ValueError: If pruner name is not recognized
        """
        pruner_name = self.config.pruner

        if pruner_name == "MedianPruner":
            return MedianPruner()
        elif pruner_name == "SuccessiveHalvingPruner":
            return SuccessiveHalvingPruner()
        elif pruner_name == "NopPruner":
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(
                f"Unknown pruner: {pruner_name}. "
                f"Supported: MedianPruner, SuccessiveHalvingPruner, NopPruner"
            )

    def get_best_params(self) -> Dict[str, Any]:
        """
        Retrieve best parameters from completed study.

        Returns:
            Dictionary of best hyperparameters

        Raises:
            RuntimeError: If study not loaded or no trials completed
        """
        if self.study is None:
            raise RuntimeError("Study not loaded. Call create_or_load_study() first.")

        if len(self.study.trials) == 0:
            raise RuntimeError("No trials completed yet.")

        return self.study.best_params

    def get_best_value(self) -> float:
        """
        Retrieve best objective value from completed study.

        Returns:
            Best objective value achieved

        Raises:
            RuntimeError: If study not loaded or no trials completed
        """
        if self.study is None:
            raise RuntimeError("Study not loaded. Call create_or_load_study() first.")

        if len(self.study.trials) == 0:
            raise RuntimeError("No trials completed yet.")

        return self.study.best_value

    def export_trials_dataframe(self, path: Path):
        """
        Export trials to CSV for analysis.

        Args:
            path: Output CSV path

        Raises:
            RuntimeError: If study not loaded
        """
        if self.study is None:
            raise RuntimeError("Study not loaded. Call create_or_load_study() first.")

        import pandas as pd

        df = self.study.trials_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    def print_study_statistics(self):
        """
        Print study statistics to console.

        Raises:
            RuntimeError: If study not loaded
        """
        if self.study is None:
            raise RuntimeError("Study not loaded. Call create_or_load_study() first.")

        print(f"Number of finished trials: {len(self.study.trials)}")

        if len(self.study.trials) > 0:
            print(f"Best trial:")
            trial = self.study.best_trial

            print(f"  Value: {trial.value}")
            print(f"  Params:")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
