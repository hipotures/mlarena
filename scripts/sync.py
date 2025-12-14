#!/usr/bin/env python3
"""
Skrypt do synchronizacji projektów Kaggle między lokalizacjami.

Kopiuje kod, dane i state.json pomijając artefakty AutoGluon i cache.
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def check_rsync_available() -> bool:
    """Sprawdza czy rsync jest zainstalowany."""
    return shutil.which("rsync") is not None


def validate_paths(source: Path, dest: Path, dry_run: bool = False):
    """
    Waliduje ścieżki źródłową i docelową.

    Args:
        source: Ścieżka źródłowa
        dest: Ścieżka docelowa
        dry_run: Czy to tryb dry-run

    Raises:
        SystemExit: Jeśli walidacja się nie powiedzie
    """
    # Sprawdź czy source istnieje
    if not source.exists():
        console.print(f"[red]Błąd:[/red] Ścieżka źródłowa nie istnieje: {source}")
        sys.exit(1)

    if not source.is_dir():
        console.print(f"[red]Błąd:[/red] Ścieżka źródłowa nie jest katalogiem: {source}")
        sys.exit(1)

    # Sprawdź czy dest istnieje
    if not dest.exists():
        if not dry_run:
            console.print(f"[yellow]Ostrzeżenie:[/yellow] Katalog docelowy nie istnieje: {dest}")
            console.print(f"Czy chcesz go utworzyć? [dim](y/N)[/dim]: ", end="")
            response = input().strip().lower()
            if response == 'y':
                try:
                    dest.mkdir(parents=True, exist_ok=True)
                    console.print(f"[green]✓[/green] Utworzono: {dest}")
                except OSError as e:
                    console.print(f"[red]Błąd:[/red] Nie można utworzyć katalogu: {e}")
                    sys.exit(1)
            else:
                console.print("[yellow]Anulowano.[/yellow]")
                sys.exit(0)
    elif not dest.is_dir():
        console.print(f"[red]Błąd:[/red] Ścieżka docelowa nie jest katalogiem: {dest}")
        sys.exit(1)


def build_rsync_command(source: Path, dest: Path, dry_run: bool = False, projects: list = None) -> list:
    """
    Buduje komendę rsync z odpowiednimi flagami.

    Args:
        source: Ścieżka źródłowa
        dest: Ścieżka docelowa
        dry_run: Czy dodać flagę --dry-run
        projects: Lista nazw projektów do synchronizacji (None = wszystkie)

    Returns:
        Lista argumentów do subprocess
    """
    cmd = [
        "rsync",
        "-avh",  # archive, verbose, human-readable
        "--progress",
        "--stats",
        "--prune-empty-dirs",  # Nie kopiuj pustych katalogów
    ]

    # Dry run
    if dry_run:
        cmd.append("--dry-run")

    # Jeśli podano listę projektów, ustaw include/exclude dla projektów NAJPIERW
    # (rsync przetwarza reguły w kolejności, więc include musi być przed exclude)
    if projects:
        # Include katalogów nadrzędnych
        cmd.extend(["--include", "projects/"])
        cmd.extend(["--include", "projects/kaggle/"])
        # Include wybranych projektów
        for project in projects:
            cmd.extend(["--include", f"projects/kaggle/{project}/"])
            cmd.extend(["--include", f"projects/kaggle/{project}/**"])
        # Wyklucz wszystkie pozostałe projekty
        cmd.extend(["--exclude", "projects/kaggle/*"])

    # WYKLUCZENIA ARTEFAKTÓW
    # Strategia: wyklucz katalogi AutoGluon tylko wewnątrz artifacts/

    # Wyklucz katalogi AutoGluon wewnątrz artifacts/ (używamy pełnej ścieżki z projects/kaggle/)
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/artifacts/model/model/"])
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/artifacts/model/models/"])
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/artifacts/model/utils/"])
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/artifacts/model/av_model/"])
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/artifacts/model/AutogluonModels/"])
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/artifacts/model/ds_sub_fit/"])

    # To samo dla preprocessing
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/artifacts/preprocess/av_model/"])
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/artifacts/preprocess/model/"])

    # Dla chainowanych preprocessingów (0-*, 1-*, etc.)
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/[0-9]*-*/artifacts/model/model/"])
    cmd.extend(["--exclude", "projects/kaggle/*/experiments/*/[0-9]*-*/artifacts/preprocess/av_model/"])

    # Pozostałe wykluczenia
    excludes = [
        ".git/",
        ".venv/",
        "venv/",
        "__pycache__/",
        "*.pyc",
        "**/*.pkl",  # Wszystkie pliki .pkl
        "**/*.lock",
        "**/state.lock",
        "*.egg-info/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "**/logs/",
    ]

    for exclude in excludes:
        cmd.extend(["--exclude", exclude])

    # Ścieżki (z trailing slash dla source)
    cmd.append(f"{source}/")
    cmd.append(f"{dest}/")

    return cmd


def parse_rsync_stats(output: str) -> dict:
    """
    Parsuje statystyki z outputu rsync.

    Args:
        output: Output z rsync --stats

    Returns:
        Dict ze statystykami
    """
    stats = {
        "files": 0,
        "size": "0 bytes",
        "transferred": "0 bytes",
    }

    # Parsuj liczby z outputu
    # Number of files: 1,234 (reg: 1,100, dir: 134)
    files_match = re.search(r"Number of files: ([\d,]+)", output)
    if files_match:
        stats["files"] = files_match.group(1)

    # Total file size: 1.23G bytes
    size_match = re.search(r"Total file size: ([\d.]+ \w+) bytes", output)
    if size_match:
        stats["size"] = size_match.group(1)

    # Total transferred file size: 1.23G bytes
    transferred_match = re.search(r"Total transferred file size: ([\d.]+ \w+) bytes", output)
    if transferred_match:
        stats["transferred"] = transferred_match.group(1)

    return stats


def display_sync_summary(source: Path, dest: Path, dry_run: bool = False, projects: list = None):
    """Wyświetla podsumowanie przed synchronizacją."""
    console.print()
    title = "[bold cyan]DRY RUN[/bold cyan] - Synchronizacja" if dry_run else "[bold green]Synchronizacja[/bold green]"
    console.print(Panel.fit(title, border_style="cyan" if dry_run else "green"))
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(style="cyan")

    table.add_row("Źródło:", str(source))
    table.add_row("Cel:", str(dest))

    if projects:
        projects_str = ", ".join(projects)
        table.add_row("Projekty:", f"[yellow]{projects_str}[/yellow] (tylko te)")
    else:
        table.add_row("Projekty:", "[green]wszystkie[/green]")

    console.print(table)
    console.print()

    # Wykluczone
    console.print("[bold]Wykluczone (nie będą kopiowane):[/bold]")
    excludes_display = [
        ".git/, .venv/",
        "AutogluonModels/",
        "__pycache__/, *.lock",
        "*.egg-info/, cache/",
    ]
    for excl in excludes_display:
        console.print(f"  [dim]- {excl}[/dim]")

    if projects:
        console.print(f"  [dim]- projects/kaggle/* (oprócz wybranych: {', '.join(projects)})[/dim]")

    console.print()


def run_rsync(source: Path, dest: Path, dry_run: bool = False, projects: list = None):
    """
    Uruchamia rsync i wyświetla wyniki.

    Args:
        source: Ścieżka źródłowa
        dest: Ścieżka docelowa
        dry_run: Czy uruchomić w trybie dry-run
        projects: Lista nazw projektów do synchronizacji (None = wszystkie)
    """
    # Wyświetl podsumowanie
    display_sync_summary(source, dest, dry_run, projects)

    # Zbuduj komendę
    cmd = build_rsync_command(source, dest, dry_run, projects)

    # Info o komendzie (dla debugowania)
    if dry_run:
        console.print("[dim]Komenda rsync:[/dim]")
        console.print(f"[dim]{' '.join(cmd)}[/dim]")
        console.print()

    # Uruchom rsync
    try:
        console.print("[cyan]Uruchamianie rsync...[/cyan]")
        console.print()

        result = subprocess.run(
            cmd,
            capture_output=False,  # Pokaż output na bieżąco
            text=True,
        )

        console.print()

        if result.returncode == 0:
            if dry_run:
                console.print("[bold green]✓ DRY RUN zakończony![/bold green]")
                console.print("[dim]Użyj bez --dry-run aby faktycznie skopiować pliki[/dim]")
            else:
                console.print("[bold green]✓ Synchronizacja zakończona pomyślnie![/bold green]")
        else:
            console.print(f"[red]Błąd:[/red] rsync zakończył się z kodem: {result.returncode}")
            sys.exit(result.returncode)

    except KeyboardInterrupt:
        console.print()
        console.print("[yellow]Przerwano przez użytkownika.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Błąd:[/red] {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Synchronizuje projekty Kaggle między lokalizacjami używając rsync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Zobacz co zostanie skopiowane (dry run)
  python scripts/sync.py \\
    projects/kaggle/playground-s5e12 \\
    /mnt/backup/kaggle/playground-s5e12 \\
    --dry-run

  # Synchronizuj pojedynczy projekt
  python scripts/sync.py \\
    projects/kaggle/playground-s5e12 \\
    /mnt/backup/kaggle/playground-s5e12

  # Synchronizuj wszystkie projekty
  python scripts/sync.py \\
    projects/kaggle \\
    /mnt/backup/kaggle

  # Synchronizuj cały kod + tylko wybrane projekty
  python scripts/sync.py \\
    /home/xai/ml/kaggle \\
    /mnt/mlarena \\
    --projects Titanic,playground-series-s5e12

Co jest kopiowane:
  - code/ (cały kod projektu)
  - templates/ (templaty)
  - data/*.csv (surowe dane)
  - experiments/*/state.json (metadane)
  - experiments/*/code_snapshot/ (snapshoty kodu)
  - experiments/*/artifacts/model/leaderboard.csv
  - experiments/*/artifacts/predict/submission.csv
  - submissions/submissions.json
  - docs/, *.md (dokumentacja)

Co jest POMIJANE:
  - .git/, .venv/ (repozytorium i virtual env)
  - **/AutogluonModels/ (artefakty modelowania)
  - **/__pycache__/, *.lock (cache i temp)
  - *.egg-info/ (package metadata)
        """
    )

    parser.add_argument(
        "source",
        type=Path,
        help="Ścieżka źródłowa (projekt lub katalog z projektami)"
    )

    parser.add_argument(
        "dest",
        type=Path,
        help="Ścieżka docelowa"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pokaż co zostanie zsynchronizowane bez wykonywania kopiowania"
    )

    parser.add_argument(
        "--projects",
        type=str,
        help="Lista projektów do synchronizacji (oddzielone przecinkami), np: Titanic,playground-s5e12. Pomija wszystkie pozostałe projekty."
    )

    args = parser.parse_args()

    # Parse projects list
    projects_list = None
    if args.projects:
        projects_list = [p.strip() for p in args.projects.split(",")]

    # Sprawdź czy rsync jest dostępny
    if not check_rsync_available():
        console.print("[red]Błąd:[/red] rsync nie jest zainstalowany!")
        console.print()
        console.print("Zainstaluj rsync:")
        console.print("  Ubuntu/Debian: [cyan]sudo apt install rsync[/cyan]")
        console.print("  Arch/Manjaro:  [cyan]sudo pacman -S rsync[/cyan]")
        console.print("  macOS:         [cyan]brew install rsync[/cyan]")
        sys.exit(1)

    # Konwertuj do absolutnych ścieżek
    source = args.source.resolve()
    dest = args.dest.resolve()

    # Walidacja
    validate_paths(source, dest, args.dry_run)

    # Uruchom synchronizację
    run_rsync(source, dest, args.dry_run, projects_list)


if __name__ == "__main__":
    main()
