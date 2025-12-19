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
        """
        Load submissions from JSON file and merge with scanned experiments
        
        Merge strategy:
        1. Load legacy submissions from submissions.json (preserves manual entries + old tracked data)
        2. Scan experiments/ folder for state.json files
        3. Merge entries:
           - Matches by experiment_id are merged (manual data wins)
           - New experiments are added with new IDs
           - Manual entries (no experiment_id) are preserved with original IDs
        """
        # 1. Load legacy submissions
        legacy_subs = []
        if self.tracker_file.exists():
            with open(self.tracker_file, 'r') as f:
                legacy_subs = json.load(f)
        
        # Index legacy submissions
        legacy_by_id = {}      # For preserving IDs
        legacy_by_exp_id = {}  # For experiment_id lookup
        
        for sub in legacy_subs:
            legacy_by_id[sub['id']] = sub
            exp_id = sub.get('experiment_id')
            if exp_id:
                legacy_by_exp_id[exp_id] = sub

        # 2. Scan state.json files
        state_entries = {sub['experiment_id']: sub for sub in self._scan_experiments()}

        # 3. Merge
        result_dict = {}  # Keyed by ID

        # Add all legacy entries first (preserves IDs)
        for legacy_sub in legacy_subs:
            exp_id = legacy_sub.get('experiment_id')
            if exp_id and exp_id in state_entries:
                # Merge: combine data from both sources
                state_sub = state_entries[exp_id]
                merged = {
                    'id': legacy_sub['id'],  # KEEP original ID
                    'experiment_id': exp_id,
                    'timestamp': state_sub.get('timestamp') or legacy_sub.get('timestamp'),
                    'filename': state_sub.get('filename') or legacy_sub.get('filename'),
                    'model_name': state_sub.get('model_name') or legacy_sub.get('model_name'),
                    'preprocess_name': state_sub.get('preprocess_name') or legacy_sub.get('preprocess_name'),
                    'local_cv_score': state_sub.get('local_cv_score') or legacy_sub.get('local_cv_score'),
                    'cv_std': legacy_sub.get('cv_std'),
                    # Manual public_score wins if present in submissions.json
                    'public_score': legacy_sub.get('public_score') or state_sub.get('public_score'),
                    'private_score': legacy_sub.get('private_score'),
                    # Concatenate notes if both have them
                    'notes': ' | '.join(filter(None, [legacy_sub.get('notes'), state_sub.get('notes')])),
                    'config': legacy_sub.get('config', {}),
                    'git_hash': state_sub.get('git_hash') or legacy_sub.get('git_hash'),
                    'code_path': legacy_sub.get('code_path'),
                    'feature_count': state_sub.get('feature_count') or legacy_sub.get('feature_count'),
                    '_source': 'merged'
                }
                result_dict[legacy_sub['id']] = merged
                # Mark as processed
                state_entries.pop(exp_id, None)
            else:
                # Keep legacy entry as-is (no state.json match)
                result_dict[legacy_sub['id']] = legacy_sub

        # 4. Add new state.json entries (not in submissions.json)
        max_existing_id = max(legacy_by_id.keys(), default=0)
        next_id = max_existing_id + 1
        
        # Sort remaining new entries by timestamp to assign stable-ish IDs
        sorted_new_entries = sorted(state_entries.values(), key=lambda x: x.get('timestamp', ''))
        
        for state_sub in sorted_new_entries:
            state_sub['id'] = next_id
            result_dict[next_id] = state_sub
            next_id += 1

        # 5. Convert to list and sort by timestamp for display
        result = list(result_dict.values())
        # Sort desc by timestamp (newest first) or ID
        result.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return result

    def _scan_experiments(self) -> List[Dict]:
        """Scan experiments directory for state.json files"""
        submissions = []
        experiments_dir = self.project_root / "experiments"
        
        if not experiments_dir.exists():
            return []

        for state_file in experiments_dir.glob("*/state.json"):
            try:
                # Skip preprocessing and setup experiments
                if state_file.parent.name.startswith(('pre-', 'av.', 'solo.', 'init', 'eda', 'data')):
                    continue

                state_data = json.loads(state_file.read_text())
                modules = state_data.get("modules", {})

                # Extract submission file (REQUIRED - skip if missing)
                submission_file = None
                predict_module = modules.get("predict", {})
                model_module = modules.get("model", {})
                submit_module = modules.get("submit", {})

                # Try new schema first
                if predict_module.get("payload", {}).get("submission_file"):
                    submission_file = predict_module["payload"]["submission_file"]
                # Try legacy schema
                elif model_module.get("submission_file"):
                    submission_file = model_module["submission_file"]
                elif submit_module.get("submission_file"):
                    submission_file = submit_module["submission_file"]

                if not submission_file:
                    continue  # SKIP if no submission file

                # Extract model name (try template name first for readability)
                model_payload = model_module.get("payload", {})
                model_name = (
                    model_module.get("invocation", {}).get("model_template") or
                    model_payload.get("template") or
                    model_payload.get("model_implementation") or
                    model_module.get("template") or
                    model_module.get("compute", {}).get("template") or
                    "unknown"
                )

                # Extract preprocessing name
                preprocess_name = (
                    model_payload.get("preprocess_template") or
                    model_module.get("invocation", {}).get("preprocess_template") or
                    predict_module.get("invocation", {}).get("preprocess_template") or
                    "unknown"
                )

                # Extract local CV (try new then legacy)
                local_cv = (
                    model_payload.get("local_cv") or
                    model_payload.get("local_cv_score") or
                    model_module.get("local_cv")
                )

                # Extract git hash (try new then legacy)
                git_info = state_data.get("git", {})
                git_hash = git_info.get("commit") or git_info.get("hash") or ""

                # Extract public score
                fetch_score_module = modules.get("fetch-score", {})
                public_score = None
                if fetch_score_module.get("status") == "completed":
                    public_score = fetch_score_module.get("payload", {}).get("score")

                # Parse timestamp
                timestamp = self._parse_timestamp(state_data.get("created_at", ""))

                # Build submission entry
                submission = {
                    "experiment_id": state_data.get("experiment_id"),
                    "timestamp": timestamp,
                    "filename": Path(submission_file).name,
                    "model_name": model_name,
                    "preprocess_name": preprocess_name,
                    "local_cv_score": local_cv,
                    "cv_std": None,
                    "public_score": public_score,
                    "private_score": None,
                    "notes": "",
                    "config": {},
                    "git_hash": git_hash,
                    "code_path": None,
                    "feature_count": predict_module.get("payload", {}).get("feature_count"),
                    "_source": "state.json"
                }

                submissions.append(submission)

            except Exception as e:
                # console.print(f"[dim]Warning: Could not parse {state_file}: {e}[/dim]")
                continue

        return submissions

    def _parse_timestamp(self, timestamp_str: str) -> str:
        """Parse ISO timestamp to display format, handling both ...Z and +00:00"""
        try:
            # Replace Z with +00:00 for compatibility
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime("%Y%m%d %H%M%S")
        except:
            # Fallback: return as-is or empty
            return timestamp_str if timestamp_str else ""

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
        code_path: Optional[str] = None,
        feature_count: Optional[int] = None
    ) -> Dict:
        """
        Add a new submission to tracking
        """
        submission = {
            "id": len(self.submissions) + 1,
            "timestamp": datetime.now().strftime("%Y%m%d %H%M%S"),
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
            "code_path": code_path,
            "feature_count": feature_count,
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
        """
        scored_submissions = [s for s in self.submissions if s.get(metric) is not None]
        sorted_submissions = sorted(scored_submissions, key=lambda x: x[metric], reverse=True)
        return sorted_submissions[:top_n]

    def display_submissions(self, limit: Optional[int] = None, sort_by: str = "id"):
        """
        Display submissions in a formatted table
        """
        if not self.submissions:
            console.print("[yellow]No submissions tracked yet[/yellow]")
            return

        # Sort submissions
        reverse = sort_by != "id"  # Most metrics should be sorted descending
        
        def get_sort_key(item):
            if sort_by == "id":
                return item["id"]
            val = item.get(sort_by)
            return val if val is not None else -float('inf')

        submissions_to_display = sorted(
            self.submissions,
            key=get_sort_key,
            reverse=reverse
        )

        if limit:
            submissions_to_display = submissions_to_display[:limit]

        # Create table
        table = Table(title="Submissions Tracking", show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=4)
        table.add_column("Timestamp", style="dim", width=16)
        table.add_column("Model", style="green", width=15)
        table.add_column("Preproc", style="blue", width=12)
        table.add_column("Local CV", justify="right", width=10)
        table.add_column("Public", justify="right", width=8)
        table.add_column("Filename", style="dim", width=25)
        table.add_column("Git", width=8)
        table.add_column("Exp ID", style="dim", width=12)

        for sub in submissions_to_display:
            local_cv_score = f"{sub['local_cv_score']:.5f}" if sub.get('local_cv_score') else "-"
            public = f"{sub['public_score']:.5f}" if sub.get('public_score') else "-"
            private = f"{sub['private_score']:.5f}" if sub.get('private_score') else "-"
            git_short = sub.get('git_hash', '')[:7] if sub.get('git_hash') else "-"
            exp_id_short = sub.get('experiment_id', '')[-12:] if sub.get('experiment_id') else "-"
            preproc = sub.get('preprocess_name', '-')
            filename = sub.get('filename', '-')

            # Highlight best scores
            style = ""
            if sub.get('public_score') and self.is_best_submission(sub['id'], 'public_score'):
                style = "bold yellow"

            table.add_row(
                str(sub['id']),
                sub['timestamp'],
                sub['model_name'],
                preproc,
                local_cv_score,
                public,
                filename,
                git_short,
                exp_id_short,
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
    repo_root = script_path.parent.parent
    competitions_root = repo_root / "projects" / "kaggle"
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