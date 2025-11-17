"""
Experiment Logger - Track all experiments with code snapshots and git history
"""

import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib


class ExperimentLogger:
    """Log experiments with code snapshots, git history, and configuration"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.repo_root = self.project_root.parent
        self.experiments_dir = self.project_root / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)

    def _relative_code_path(self, code_path: Path) -> str:
        """Return path relative to project or repo root, falling back to absolute."""
        bases = [self.project_root, self.repo_root]
        for base in bases:
            try:
                return str(code_path.relative_to(base))
            except ValueError:
                continue
        return str(code_path)

    def _get_git_info(self) -> Dict[str, Any]:
        """Get git information"""
        try:
            # Get current commit hash
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get short hash
            git_hash_short = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=self.project_root.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Check for uncommitted changes
            status = subprocess.check_output(
                ['git', 'status', '--porcelain', str(self.project_root)],
                cwd=self.project_root.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            has_uncommitted = len(status) > 0

            # Get commit message
            commit_msg = subprocess.check_output(
                ['git', 'log', '-1', '--format=%s'],
                cwd=self.project_root.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get branch name
            branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.project_root.parent,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            return {
                'hash': git_hash,
                'hash_short': git_hash_short,
                'branch': branch,
                'commit_message': commit_msg,
                'has_uncommitted_changes': has_uncommitted,
                'uncommitted_files': status.split('\n') if has_uncommitted else []
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                'hash': None,
                'hash_short': None,
                'branch': None,
                'commit_message': None,
                'has_uncommitted_changes': True,
                'uncommitted_files': ['Git not available']
            }

    def _create_experiment_id(self, model_name: str) -> str:
        """Create unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{model_name}"

    def _create_code_hash(self, code_path: Path) -> str:
        """Create hash of code file for change detection"""
        if not code_path.exists():
            return "NO_FILE"

        with open(code_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]

    def log_experiment(
        self,
        model_name: str,
        code_path: Optional[Path] = None,
        config: Optional[Dict] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Log experiment with full tracking

        Args:
            model_name: Name/identifier of the model
            code_path: Path to the code file used
            config: Configuration dictionary
            notes: Additional notes

        Returns:
            Experiment metadata dictionary
        """
        # Create experiment ID
        experiment_id = self._create_experiment_id(model_name)

        # Get git info
        git_info = self._get_git_info()

        # Warn about uncommitted changes
        if git_info['has_uncommitted_changes']:
            print(f"⚠️  WARNING: Uncommitted changes detected!")
            print(f"   Files: {', '.join(git_info['uncommitted_files'][:5])}")
            if len(git_info['uncommitted_files']) > 5:
                print(f"   ... and {len(git_info['uncommitted_files']) - 5} more")
            print(f"   Consider committing before running experiment!")

        # Create experiment metadata
        experiment = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': model_name,
            'code_path': self._relative_code_path(code_path) if code_path else None,
            'code_hash': self._create_code_hash(code_path) if code_path else None,
            'git': git_info,
            'config': config or {},
            'notes': notes
        }

        # Save experiment metadata
        metadata_path = self.experiments_dir / f"{experiment_id}.json"
        with open(metadata_path, 'w') as f:
            json.dump(experiment, f, indent=2)

        # Save code snapshot if available
        if code_path and code_path.exists():
            snapshot_path = self.experiments_dir / f"{experiment_id}.py"
            shutil.copy2(code_path, snapshot_path)
            experiment['code_snapshot'] = str(snapshot_path.relative_to(self.project_root))

        print(f"✓ Experiment logged: {experiment_id}")
        print(f"  Git: {git_info['hash_short']} ({git_info['branch']})")
        if code_path:
            print(f"  Code: {code_path.name} (hash: {experiment['code_hash']})")

        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Retrieve experiment metadata"""
        metadata_path = self.experiments_dir / f"{experiment_id}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def list_experiments(self, limit: Optional[int] = None) -> list:
        """List all experiments, sorted by timestamp (newest first)"""
        experiments = []

        for metadata_file in sorted(self.experiments_dir.glob("*.json"), reverse=True):
            with open(metadata_file, 'r') as f:
                experiments.append(json.load(f))

            if limit and len(experiments) >= limit:
                break

        return experiments

    def restore_code(self, experiment_id: str, output_path: Optional[Path] = None) -> Path:
        """
        Restore code from experiment snapshot

        Args:
            experiment_id: ID of experiment to restore
            output_path: Where to save restored code (default: original location)

        Returns:
            Path to restored code file
        """
        experiment = self.get_experiment(experiment_id)

        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        snapshot_path = self.project_root / experiment['code_snapshot']

        if not snapshot_path.exists():
            raise FileNotFoundError(f"Code snapshot not found: {snapshot_path}")

        if output_path is None:
            output_path = self.project_root / experiment['code_path']

        shutil.copy2(snapshot_path, output_path)
        print(f"✓ Code restored from {experiment_id}")
        print(f"  Saved to: {output_path}")

        return output_path

    def checkout_git_version(self, experiment_id: str):
        """
        Print instructions to checkout git version from experiment

        Args:
            experiment_id: ID of experiment
        """
        experiment = self.get_experiment(experiment_id)

        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        git_hash = experiment['git']['hash']

        if not git_hash:
            print("⚠️  No git hash available for this experiment")
            return

        print(f"To checkout git version from experiment {experiment_id}:")
        print(f"\n  git checkout {git_hash}\n")
        print(f"Branch: {experiment['git']['branch']}")
        print(f"Commit: {experiment['git']['commit_message']}")

        if experiment['git']['has_uncommitted_changes']:
            print(f"\n⚠️  WARNING: Experiment was run with uncommitted changes!")
            print(f"   Checking out commit may not fully reproduce the experiment")


# CLI interface
if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from rich.table import Table

    console = Console()

    parser = argparse.ArgumentParser(description="Manage experiment logs")
    parser.add_argument('--project', required=True, help='Project directory name')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List experiments
    list_parser = subparsers.add_parser('list', help='List experiments')
    list_parser.add_argument('--limit', type=int, help='Limit number of results')

    # Show experiment details
    show_parser = subparsers.add_parser('show', help='Show experiment details')
    show_parser.add_argument('experiment_id', help='Experiment ID')

    # Restore code
    restore_parser = subparsers.add_parser('restore', help='Restore code from experiment')
    restore_parser.add_argument('experiment_id', help='Experiment ID')
    restore_parser.add_argument('--output', help='Output path for restored code')

    # Checkout git version
    checkout_parser = subparsers.add_parser('checkout', help='Show git checkout command')
    checkout_parser.add_argument('experiment_id', help='Experiment ID')

    args = parser.parse_args()

    # Determine project root
    script_path = Path(__file__).resolve()
    competitions_root = script_path.parent.parent
    project_root = competitions_root / args.project

    if not project_root.exists():
        console.print(f"[red]Error: Project directory '{args.project}' not found[/red]")
        exit(1)

    logger = ExperimentLogger(project_root)

    if args.command == 'list':
        experiments = logger.list_experiments(limit=args.limit)

        if not experiments:
            console.print("[yellow]No experiments found[/yellow]")
        else:
            table = Table(title="Experiments", show_header=True, header_style="bold magenta")
            table.add_column("ID", style="cyan", width=25)
            table.add_column("Model", style="green", width=20)
            table.add_column("Git", width=15)
            table.add_column("Uncommitted", width=12)
            table.add_column("Timestamp", width=16)

            for exp in experiments:
                git_info = exp['git']
                table.add_row(
                    exp['experiment_id'],
                    exp['model_name'],
                    git_info.get('hash_short', 'N/A'),
                    '⚠️  Yes' if git_info.get('has_uncommitted_changes') else '✓ No',
                    exp['timestamp']
                )

            console.print(table)

    elif args.command == 'show':
        experiment = logger.get_experiment(args.experiment_id)

        if not experiment:
            console.print(f"[red]Experiment {args.experiment_id} not found[/red]")
            exit(1)

        console.print(f"\n[bold]Experiment: {experiment['experiment_id']}[/bold]")
        console.print(f"Model: {experiment['model_name']}")
        console.print(f"Timestamp: {experiment['timestamp']}")
        console.print(f"Code: {experiment.get('code_path', 'N/A')}")
        console.print(f"Code Hash: {experiment.get('code_hash', 'N/A')}")

        git = experiment['git']
        console.print(f"\n[bold]Git Info:[/bold]")
        console.print(f"Hash: {git.get('hash', 'N/A')}")
        console.print(f"Branch: {git.get('branch', 'N/A')}")
        console.print(f"Commit: {git.get('commit_message', 'N/A')}")
        console.print(f"Uncommitted: {'⚠️  Yes' if git.get('has_uncommitted_changes') else '✓ No'}")

        if experiment.get('config'):
            console.print(f"\n[bold]Config:[/bold]")
            console.print(json.dumps(experiment['config'], indent=2))

        if experiment.get('notes'):
            console.print(f"\n[bold]Notes:[/bold]")
            console.print(experiment['notes'])

    elif args.command == 'restore':
        output = Path(args.output) if args.output else None
        logger.restore_code(args.experiment_id, output)

    elif args.command == 'checkout':
        logger.checkout_git_version(args.experiment_id)

    else:
        parser.print_help()
