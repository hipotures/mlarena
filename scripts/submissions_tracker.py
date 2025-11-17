"""
Submission tracking system for managing model submissions and scores
Tracks: local CV score, public leaderboard score, private leaderboard score
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class SubmissionsTracker:
    """Track and manage competition submissions"""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.tracker_file = self.project_root / "submissions" / "submissions.json"
        self.submissions = self._load_submissions()

    def _load_submissions(self) -> List[Dict]:
        """Load submissions from JSON file"""
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                return json.load(f)
        return []

    def _save_submissions(self):
        """Save submissions to JSON file"""
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracker_file, 'w') as f:
            json.dump(self.submissions, f, indent=2)

    def add_submission(
        self,
        filename: str,
        model_name: str,
        local_cv_score: Optional[float] = None,
        cv_std: Optional[float] = None,
        public_score: Optional[float] = None,
        private_score: Optional[float] = None,
        notes: str = "",
        config: Optional[Dict] = None,
        experiment_id: Optional[str] = None,
        git_hash: Optional[str] = None,
        code_path: Optional[str] = None
    ) -> Dict:
        """
        Add a new submission to tracking

        Args:
            filename: Name of the submission file
            model_name: Model identifier (e.g., 'autogluon-medium', 'lgbm-v1')
            local_cv_score: Local cross-validation score
            cv_std: Standard deviation of CV score
            public_score: Public leaderboard score
            private_score: Private leaderboard score (available after competition ends)
            notes: Additional notes about the submission
            config: Model configuration dictionary
            experiment_id: Experiment ID from experiment logger
            git_hash: Git commit hash
            code_path: Path to code file used

        Returns:
            Dictionary with submission details
        """
        submission = {
            "id": len(self.submissions) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename,
            "model_name": model_name,
            "local_cv_score": local_cv_score,
            "cv_std": cv_std,
            "public_score": public_score,
            "private_score": private_score,
            "notes": notes,
            "config": config or {},
            "experiment_id": experiment_id,
            "git_hash": git_hash,
            "code_path": code_path
        }

        self.submissions.append(submission)
        self._save_submissions()

        console.print(f"[green]✓[/green] Submission #{submission['id']} added: {filename}")
        return submission

    def update_scores(
        self,
        submission_id: int,
        public_score: Optional[float] = None,
        private_score: Optional[float] = None
    ):
        """
        Update leaderboard scores for a submission

        Args:
            submission_id: ID of the submission to update
            public_score: New public leaderboard score
            private_score: New private leaderboard score
        """
        for sub in self.submissions:
            if sub['id'] == submission_id:
                if public_score is not None:
                    sub['public_score'] = public_score
                if private_score is not None:
                    sub['private_score'] = private_score
                self._save_submissions()
                console.print(f"[green]✓[/green] Updated scores for submission #{submission_id}")
                return

        console.print(f"[red]✗[/red] Submission #{submission_id} not found")

    def get_best_submissions(self, metric: str = "public_score", top_n: int = 5) -> List[Dict]:
        """
        Get top N submissions by specified metric

        Args:
            metric: Metric to sort by ('local_cv_score', 'public_score', 'private_score')
            top_n: Number of top submissions to return

        Returns:
            List of top submissions
        """
        scored_submissions = [s for s in self.submissions if s.get(metric) is not None]
        sorted_submissions = sorted(scored_submissions, key=lambda x: x[metric], reverse=True)
        return sorted_submissions[:top_n]

    def display_submissions(self, limit: Optional[int] = None, sort_by: str = "id"):
        """
        Display submissions in a formatted table

        Args:
            limit: Limit number of submissions to display
            sort_by: Field to sort by ('id', 'local_cv_score', 'public_score', 'private_score', 'timestamp')
        """
        if not self.submissions:
            console.print("[yellow]No submissions tracked yet[/yellow]")
            return

        # Sort submissions
        reverse = sort_by != "id"  # Most metrics should be sorted descending
        submissions_to_display = sorted(
            self.submissions,
            key=lambda x: x.get(sort_by, -float('inf')) if sort_by != "id" else x["id"],
            reverse=reverse
        )

        if limit:
            submissions_to_display = submissions_to_display[:limit]

        # Create table
        table = Table(title="Submissions Tracking", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Timestamp", style="dim", width=16)
        table.add_column("Model", style="green", width=20)
        table.add_column("Local CV", justify="right", width=10)
        table.add_column("Public", justify="right", width=10)
        table.add_column("Private", justify="right", width=10)
        table.add_column("Git", width=8)
        table.add_column("Exp ID", style="dim", width=15)
        table.add_column("Notes", width=25)

        for sub in submissions_to_display:
            local_cv = f"{sub['local_cv_score']:.5f}" if sub.get('local_cv_score') else "-"
            public = f"{sub['public_score']:.5f}" if sub.get('public_score') else "-"
            private = f"{sub['private_score']:.5f}" if sub.get('private_score') else "-"
            git_short = sub.get('git_hash', '')[:7] if sub.get('git_hash') else "-"
            exp_id_short = sub.get('experiment_id', '')[-15:] if sub.get('experiment_id') else "-"

            # Highlight best scores
            style = ""
            if sub.get('public_score') and self.is_best_submission(sub['id'], 'public_score'):
                style = "bold yellow"

            table.add_row(
                str(sub['id']),
                sub['timestamp'],
                sub['model_name'],
                local_cv,
                public,
                private,
                git_short,
                exp_id_short,
                sub.get('notes', '')[:25],
                style=style
            )

        console.print(table)

        # Display summary
        self._display_summary()

    def _display_summary(self):
        """Display summary statistics"""
        if not self.submissions:
            return

        best_local = max((s for s in self.submissions if s.get('local_cv_score')),
                        key=lambda x: x['local_cv_score'], default=None)
        best_public = max((s for s in self.submissions if s.get('public_score')),
                         key=lambda x: x['public_score'], default=None)
        best_private = max((s for s in self.submissions if s.get('private_score')),
                          key=lambda x: x['private_score'], default=None)

        summary_lines = [
            f"Total Submissions: {len(self.submissions)}",
        ]

        if best_local:
            summary_lines.append(
                f"Best Local CV: {best_local['local_cv_score']:.5f} "
                f"(#{best_local['id']} - {best_local['model_name']})"
            )

        if best_public:
            summary_lines.append(
                f"Best Public: {best_public['public_score']:.5f} "
                f"(#{best_public['id']} - {best_public['model_name']})"
            )

        if best_private:
            summary_lines.append(
                f"Best Private: {best_private['private_score']:.5f} "
                f"(#{best_private['id']} - {best_private['model_name']})"
            )

        console.print(Panel("\n".join(summary_lines), title="Summary", border_style="blue"))

    def is_best_submission(self, submission_id: int, metric: str) -> bool:
        """Check if submission has the best score for given metric"""
        scored_submissions = [s for s in self.submissions if s.get(metric) is not None]
        if not scored_submissions:
            return False
        best = max(scored_submissions, key=lambda x: x[metric])
        return best['id'] == submission_id

    def export_to_csv(self, output_path: Optional[Path] = None):
        """Export submissions tracking to CSV"""
        import pandas as pd

        if not self.submissions:
            console.print("[yellow]No submissions to export[/yellow]")
            return

        df = pd.DataFrame(self.submissions)

        if output_path is None:
            output_path = self.project_root / "submissions" / "submissions_tracking.csv"

        df.to_csv(output_path, index=False)
        console.print(f"[green]✓[/green] Exported to {output_path}")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage competition submissions")
    parser.add_argument('--project', required=True, help='Project directory name (e.g., playground-series-s5e11)')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add submission
    add_parser = subparsers.add_parser('add', help='Add new submission')
    add_parser.add_argument('filename', help='Submission filename')
    add_parser.add_argument('model_name', help='Model name/identifier')
    add_parser.add_argument('--local-cv', type=float, help='Local CV score')
    add_parser.add_argument('--cv-std', type=float, help='CV standard deviation')
    add_parser.add_argument('--public', type=float, help='Public leaderboard score')
    add_parser.add_argument('--private', type=float, help='Private leaderboard score')
    add_parser.add_argument('--notes', default='', help='Additional notes')

    # Update scores
    update_parser = subparsers.add_parser('update', help='Update submission scores')
    update_parser.add_argument('id', type=int, help='Submission ID')
    update_parser.add_argument('--public', type=float, help='Public leaderboard score')
    update_parser.add_argument('--private', type=float, help='Private leaderboard score')

    # List submissions
    list_parser = subparsers.add_parser('list', help='List submissions')
    list_parser.add_argument('--limit', type=int, help='Limit number of submissions')
    list_parser.add_argument('--sort-by', default='id',
                           choices=['id', 'local_cv_score', 'public_score', 'private_score', 'timestamp'],
                           help='Sort by field')

    # Export
    export_parser = subparsers.add_parser('export', help='Export to CSV')

    args = parser.parse_args()

    # Determine project root
    script_path = Path(__file__).resolve()
    competitions_root = script_path.parent.parent
    project_root = competitions_root / args.project

    if not project_root.exists():
        console.print(f"[red]Error: Project directory '{args.project}' not found[/red]")
        console.print(f"Available projects: {[d.name for d in competitions_root.iterdir() if d.is_dir() and not d.name.startswith('.')]}")
        exit(1)

    tracker = SubmissionsTracker(project_root)

    if args.command == 'add':
        tracker.add_submission(
            filename=args.filename,
            model_name=args.model_name,
            local_cv_score=args.local_cv,
            cv_std=args.cv_std,
            public_score=args.public,
            private_score=args.private,
            notes=args.notes
        )
        tracker.display_submissions(limit=10, sort_by='id')

    elif args.command == 'update':
        tracker.update_scores(
            submission_id=args.id,
            public_score=args.public,
            private_score=args.private
        )
        tracker.display_submissions(limit=10, sort_by='public_score')

    elif args.command == 'list':
        tracker.display_submissions(limit=args.limit, sort_by=args.sort_by)

    elif args.command == 'export':
        tracker.export_to_csv()

    else:
        parser.print_help()
