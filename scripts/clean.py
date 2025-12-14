#!/usr/bin/env python3
"""
Skrypt do czyszczenia artefaktów AutoGluon z projektów Kaggle.

Usuwa duże pliki modelowania zachowując dane potrzebne do reprodukcji eksperymentów.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

console = Console()


def get_project_root() -> Path:
    """Zwraca główny katalog repozytorium (gdzie znajduje się scripts/)."""
    return Path(__file__).parent.parent.resolve()


def validate_project(project_name: str) -> Path:
    """
    Waliduje czy projekt istnieje i zwraca ścieżkę do katalogu projektu.

    Args:
        project_name: Nazwa projektu

    Returns:
        Path do katalogu projektu

    Raises:
        SystemExit: Jeśli projekt nie istnieje
    """
    repo_root = get_project_root()
    project_path = repo_root / "projects" / "kaggle" / project_name

    if not project_path.exists():
        console.print(f"[red]Błąd:[/red] Projekt '{project_name}' nie istnieje w {project_path}")
        available = list((repo_root / "projects" / "kaggle").glob("*"))
        if available:
            console.print("\nDostępne projekty:")
            for proj in available:
                if proj.is_dir():
                    console.print(f"  - {proj.name}")
        sys.exit(1)

    experiments_path = project_path / "experiments"
    if not experiments_path.exists():
        console.print(f"[yellow]Ostrzeżenie:[/yellow] Katalog experiments/ nie istnieje w {project_path}")
        console.print("Projekt nie ma żadnych eksperymentów do wyczyszczenia.")
        sys.exit(0)

    return project_path


def get_directory_size(path: Path) -> int:
    """Oblicza rozmiar katalogu (rekursywnie)."""
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Formatuje rozmiar w bajtach do czytelnej formy."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def find_autogluon_model_dirs(experiments_path: Path) -> List[Path]:
    """
    Znajduje wszystkie katalogi zawierające modele AutoGluon.

    AutoGluon może zapisywać modele pod różnymi nazwami (model/, AutogluonModels/, av_model/, etc.).
    Identyfikujemy je po obecności plików predictor.pkl lub learner.pkl.

    Args:
        experiments_path: Ścieżka do katalogu experiments

    Returns:
        Lista ścieżek do katalogów AutoGluon
    """
    autogluon_dirs = set()

    # Znajdź wszystkie pliki predictor.pkl i learner.pkl
    for pkl_file in experiments_path.rglob("predictor.pkl"):
        if pkl_file.is_file():
            # Katalog zawierający predictor.pkl to katalog AutoGluon
            autogluon_dirs.add(pkl_file.parent)

    for pkl_file in experiments_path.rglob("learner.pkl"):
        if pkl_file.is_file():
            # Katalog zawierający learner.pkl to katalog AutoGluon
            autogluon_dirs.add(pkl_file.parent)

    return list(autogluon_dirs)


def find_items_to_clean(
    project_path: Path,
    remove_processed_csv: bool = False
) -> Tuple[List[Tuple[Path, str, int]], int]:
    """
    Znajduje wszystkie pliki i katalogi do usunięcia.

    Args:
        project_path: Ścieżka do projektu
        remove_processed_csv: Czy usuwać także *_processed.csv

    Returns:
        Tuple z listą (ścieżka, typ, rozmiar) i całkowitym rozmiarem
    """
    items_to_remove = []
    total_size = 0
    experiments_path = project_path / "experiments"

    # Znajdź wszystkie katalogi zawierające modele AutoGluon
    # (identyfikowane po predictor.pkl lub learner.pkl)
    autogluon_dirs = find_autogluon_model_dirs(experiments_path)

    for autogluon_dir in autogluon_dirs:
        size = get_directory_size(autogluon_dir)
        # Typ: nazwa katalogu + "/"
        dir_name = autogluon_dir.name + "/"
        items_to_remove.append((autogluon_dir, dir_name, size))
        total_size += size

    # Znajdź wszystkie pliki .lock i katalogi __pycache__
    seen_paths = set()  # Unikaj duplikatów

    for exp_dir in experiments_path.iterdir():
        if not exp_dir.is_dir():
            continue

        # Pliki .lock (wszystkie pliki .lock łącznie z state.lock i state.json.lock)
        for lock_file in exp_dir.rglob("*.lock"):
            if lock_file.is_file() and lock_file not in seen_paths:
                size = lock_file.stat().st_size
                items_to_remove.append((lock_file, "*.lock", size))
                total_size += size
                seen_paths.add(lock_file)

        # state.lock (bez rozszerzenia - może nie mieć .lock jeśli jest bez rozszerzenia)
        state_lock = exp_dir / "state.lock"
        if state_lock.exists() and state_lock.is_file() and state_lock not in seen_paths:
            size = state_lock.stat().st_size
            items_to_remove.append((state_lock, "state.lock", size))
            total_size += size
            seen_paths.add(state_lock)

        # __pycache__
        for pycache_dir in exp_dir.rglob("__pycache__"):
            if pycache_dir.is_dir() and pycache_dir not in seen_paths:
                size = get_directory_size(pycache_dir)
                items_to_remove.append((pycache_dir, "__pycache__/", size))
                total_size += size
                seen_paths.add(pycache_dir)

    # Opcjonalnie: *_processed.csv
    if remove_processed_csv:
        for pre_dir in experiments_path.glob("pre-*"):
            if not pre_dir.is_dir():
                continue

            artifacts_preprocess = pre_dir / "artifacts" / "preprocess"
            if artifacts_preprocess.exists():
                for csv_file in artifacts_preprocess.glob("*_processed.csv"):
                    if csv_file.is_file():
                        size = csv_file.stat().st_size
                        items_to_remove.append((csv_file, "*_processed.csv", size))
                        total_size += size

    return items_to_remove, total_size


def display_dry_run(items: List[Tuple[Path, str, int]], total_size: int, project_path: Path):
    """Wyświetla tabelę plików do usunięcia w trybie dry-run."""
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]DRY RUN[/bold cyan] - Pliki do usunięcia w projekcie: [yellow]{project_path.name}[/yellow]",
        border_style="cyan"
    ))
    console.print()

    if not items:
        console.print("[green]✓[/green] Brak plików do usunięcia - projekt już czysty!")
        return

    # Tabelka z plikami (sortowanie: od największych do najmniejszych)
    table = Table(show_header=True, header_style="bold magenta", title="Top 20 największych elementów")
    table.add_column("Ścieżka", style="dim", no_wrap=False)
    table.add_column("Typ", style="cyan", width=20)
    table.add_column("Rozmiar", justify="right", style="green")

    # Sortuj według rozmiaru (od największych)
    sorted_items = sorted(items, key=lambda x: x[2], reverse=True)

    # Pokaż top 20 (lub wszystkie jeśli mniej)
    display_count = min(20, len(sorted_items))
    for path, item_type, size in sorted_items[:display_count]:
        # Ścieżka relatywna do projektu
        try:
            rel_path = path.relative_to(project_path)
        except ValueError:
            rel_path = path

        table.add_row(str(rel_path), item_type, format_size(size))

    if len(sorted_items) > display_count:
        # Policz rozmiar pozostałych
        remaining_size = sum(size for _, _, size in sorted_items[display_count:])
        table.add_row(
            f"... i {len(sorted_items) - display_count} mniejszych elementów ({format_size(remaining_size)})",
            "",
            "",
            style="dim italic"
        )

    console.print(table)
    console.print()

    # Podsumowanie
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column(style="bold")
    summary_table.add_column(style="cyan")

    # Policz katalogi vs pliki
    dirs_count = sum(1 for _, t, _ in items if t.endswith("/"))
    files_count = len(items) - dirs_count

    summary_table.add_row("Katalogów:", str(dirs_count))
    summary_table.add_row("Plików:", str(files_count))
    summary_table.add_row("Całkowity rozmiar:", f"[bold green]{format_size(total_size)}[/bold green]")

    console.print(Panel(summary_table, title="[bold]Podsumowanie[/bold]", border_style="green"))
    console.print()


def clean_items(items: List[Tuple[Path, str, int]], project_path: Path):
    """Usuwa pliki i katalogi z progress barem."""
    console.print()
    console.print(f"[bold green]Czyszczenie projektu:[/bold green] {project_path.name}")
    console.print()

    if not items:
        console.print("[green]✓[/green] Brak plików do usunięcia - projekt już czysty!")
        return

    removed_count = 0
    removed_size = 0
    errors = []

    # Bezpieczeństwo: wszystkie ścieżki MUSZĄ być w experiments/
    experiments_dir = (project_path / "experiments").resolve()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Usuwanie plików...", total=len(items))

        for path, item_type, size in items:
            try:
                # KRYTYCZNE ZABEZPIECZENIE: sprawdź czy ścieżka jest wewnątrz experiments/
                path_resolved = path.resolve()
                try:
                    path_resolved.relative_to(experiments_dir)
                except ValueError:
                    # Ścieżka jest poza experiments/ - NIGDY NIE USUWAJ
                    errors.append(f"BEZPIECZEŃSTWO: Pominięto {path} (poza katalogiem experiments)")
                    progress.advance(task)
                    continue

                # Dodatkowe zabezpieczenie: nie usuwaj samego katalogu experiments
                if path_resolved == experiments_dir:
                    errors.append(f"BEZPIECZEŃSTWO: Pominięto {path} (katalog główny experiments)")
                    progress.advance(task)
                    continue

                # Usuń
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink()
                else:
                    # Już nie istnieje
                    pass

                removed_count += 1
                removed_size += size

            except (PermissionError, OSError) as e:
                errors.append(f"Błąd przy usuwaniu {path}: {e}")

            progress.advance(task)

    console.print()

    # Podsumowanie
    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column(style="bold")
    summary_table.add_column(style="cyan")

    summary_table.add_row("Usuniętych elementów:", str(removed_count))
    summary_table.add_row("Zwolniono miejsca:", f"[bold green]{format_size(removed_size)}[/bold green]")

    if errors:
        summary_table.add_row("Błędów:", f"[red]{len(errors)}[/red]")

    console.print(Panel(summary_table, title="[bold]Wynik czyszczenia[/bold]", border_style="green"))
    console.print()

    # Pokaż błędy jeśli były
    if errors:
        console.print("[yellow]Ostrzeżenia/Błędy:[/yellow]")
        for error in errors[:10]:  # Pokaż max 10
            console.print(f"  [dim]{error}[/dim]")
        if len(errors) > 10:
            console.print(f"  [dim]... i {len(errors) - 10} więcej[/dim]")
        console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Czyści artefakty AutoGluon z projektu Kaggle zachowując dane do reprodukcji",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  # Zobacz co zostanie usunięte
  python scripts/clean.py --project playground-series-s5e12 --dry-run

  # Usuń artefakty AutoGluon (zachowaj processed CSV)
  python scripts/clean.py --project playground-series-s5e12

  # Usuń artefakty + processed CSV
  python scripts/clean.py --project playground-series-s5e12 --remove-processed-csv

Co jest usuwane:
  - AutogluonModels/ (modele, pickle'y, cache)
  - *.lock, state.lock (pliki tymczasowe)
  - __pycache__/ (bytecode Python)
  - *_processed.csv (opcjonalnie, z --remove-processed-csv)

Co jest ZACHOWANE:
  - state.json (metadane eksperymentów)
  - code_snapshot/ (kod)
  - leaderboard.csv, train_used.csv (wyniki)
  - submission.csv (predykcje)
  - data/ (surowe dane)
        """
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Nazwa projektu do wyczyszczenia (np. playground-series-s5e12)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pokaż co zostanie usunięte bez wykonywania usuwania"
    )

    parser.add_argument(
        "--remove-processed-csv",
        action="store_true",
        help="Usuń także *_processed.csv z katalogów pre-* (domyślnie: zachowaj)"
    )

    args = parser.parse_args()

    # Walidacja projektu
    project_path = validate_project(args.project)

    # Znajdź pliki do usunięcia
    console.print(f"[cyan]Skanowanie projektu:[/cyan] {args.project}")
    items, total_size = find_items_to_clean(project_path, args.remove_processed_csv)

    # Dry run lub wykonanie
    if args.dry_run:
        display_dry_run(items, total_size, project_path)
        console.print("[dim]Użyj bez --dry-run aby faktycznie usunąć pliki[/dim]")
    else:
        # Pokaż co będzie usunięte i poproś o potwierdzenie
        display_dry_run(items, total_size, project_path)

        if items:
            console.print("[bold yellow]Czy na pewno chcesz usunąć te pliki?[/bold yellow] [dim](y/N)[/dim]: ", end="")
            response = input().strip().lower()

            if response == 'y':
                clean_items(items, project_path)
                console.print("[bold green]✓ Czyszczenie zakończone![/bold green]")
            else:
                console.print("[yellow]Anulowano.[/yellow]")


if __name__ == "__main__":
    main()
