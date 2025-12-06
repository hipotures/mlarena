"""File operations for project initialization."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict

from rich.console import Console

# Repository paths
REPO_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE_ROOT = REPO_ROOT / "config" / "templates"
GITIGNORE_TEMPLATE = REPO_ROOT / "config" / ".gitignore"


def create_directory_structure(project_root: Path, console: Console) -> None:
    """Create standard Kaggle project directory structure."""
    console.print(f"[cyan]Creating project structure...[/cyan]")

    dirs = [
        "data",
        "code/exploration",
        "code/models",
        "code/utils",
        "templates",
        "docs",
        "experiments",
        "submissions",
        "logs",
    ]

    for dir_path in dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]✓[/green] {dir_path}/")

    # Create .gitkeep files
    for keep_dir in ["data", "docs", "experiments", "submissions", "logs"]:
        (project_root / keep_dir / ".gitkeep").touch()


def copy_templates(project_root: Path, console: Console) -> None:
    """Copy template files (.gitignore, README, configs, etc.)."""
    console.print("\n[cyan]Copying template files...[/cyan]")

    template_project = TEMPLATE_ROOT / "kaggle_competition"

    # .gitignore
    gitignore_src = template_project / ".gitignore"
    gitignore_dst = project_root / ".gitignore"
    if gitignore_src.exists():
        shutil.copy(gitignore_src, gitignore_dst)
    else:
        # Minimal fallback
        gitignore_dst.write_text(
            "# Data and outputs\n"
            "data/\n"
            "experiments/\n"
            "submissions/\n"
            "logs/\n"
            "__pycache__/\n"
        )
    console.print("  [green]✓[/green] .gitignore")

    # README.md
    readme_src = template_project / "README.md"
    readme_dst = project_root / "README.md"
    if readme_src.exists():
        shutil.copy(readme_src, readme_dst)
    else:
        readme_dst.write_text(f"# {project_root.name}\n\n")
    console.print("  [green]✓[/green] README.md")

    # configs/
    configs_src = template_project / "configs"
    if configs_src.exists():
        shutil.copytree(configs_src, project_root / "configs", dirs_exist_ok=True)
    else:
        (project_root / "configs").mkdir(exist_ok=True)
        (project_root / "configs" / ".gitkeep").touch()
    console.print("  [green]✓[/green] configs/")

    # Initialize empty template files
    templates_dir = project_root / "templates"
    for tpl_name in ["model.yaml", "preprocess.yaml"]:
        tpl_path = templates_dir / tpl_name
        if not tpl_path.exists():
            tpl_path.write_text("templates: {}\n")


def download_kaggle_data(competition_slug: str, project_root: Path, console: Console) -> bool:
    """Download and extract Kaggle competition data.

    Returns:
        True on success, False on failure
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    data_dir = project_root / "data"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Download
        task = progress.add_task(f"Downloading data from Kaggle for '{competition_slug}'...", total=None)

        try:
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", competition_slug, "-p", str(data_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            progress.remove_task(task)
            console.print(f"  [green]✓[/green] Data downloaded")

            # Find and unzip
            zip_files = list(data_dir.glob("*.zip"))
            if zip_files:
                zip_file = zip_files[0]
                task2 = progress.add_task(f"Extracting {zip_file.name}...", total=None)
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(data_dir)
                progress.remove_task(task2)
                console.print(f"  [green]✓[/green] Data extracted")
                zip_file.unlink()  # Remove zip after extraction

            return True

        except subprocess.CalledProcessError as e:
            progress.remove_task(task)
            console.print(f"[red]Error downloading data: {e.stderr}[/red]")
            console.print("[yellow]You can download data manually later with:[/yellow]")
            console.print(f"  cd {competition_slug}/data && kaggle competitions download -c {competition_slug}")
            return False


def customize_readme(project_root: Path, replacements: Dict[str, str], console: Console) -> None:
    """Replace placeholders in README.md with actual values."""
    readme_path = project_root / "README.md"
    if not readme_path.exists():
        console.print(f"[yellow]! README.md missing; skipped customization[/yellow]")
        return

    try:
        readme_content = readme_path.read_text()
        for needle, repl in replacements.items():
            readme_content = readme_content.replace(needle, str(repl))
        readme_path.write_text(readme_content)
        console.print(f"[green]✓[/green] Customized README.md")
    except Exception as exc:
        console.print(f"[yellow]! README customization failed: {exc}[/yellow]")
