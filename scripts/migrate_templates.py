#!/usr/bin/env python3
"""
Migrate monolithic template YAMLs to individual files.

Usage:
    python scripts/migrate_templates.py --dry-run  # Preview changes
    python scripts/migrate_templates.py            # Execute migration
    python scripts/migrate_templates.py --project playground-series-s5e12  # Migrate specific project
    python scripts/migrate_templates.py --all-projects  # Migrate all projects
    python scripts/migrate_templates.py --validate  # Validate migration
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from rich.console import Console
from rich.table import Table

console = Console()
REPO_ROOT = Path(__file__).resolve().parent.parent


class TemplateValidationError(Exception):
    """Raised when template validation fails."""
    pass


def migrate_template_file(
    source_file: Path,
    target_dir: Path,
    template_type: str,
    dry_run: bool = True
) -> Tuple[int, List[str]]:
    """
    Migrate single YAML file to directory structure.

    Args:
        source_file: Path to source YAML file (e.g., model.yaml)
        target_dir: Target directory (e.g., templates/model/)
        template_type: Type of template ("model" or "preprocess")
        dry_run: If True, only preview changes

    Returns:
        Tuple of (count of migrated templates, list of template names)
    """
    if not source_file.exists():
        console.print(f"[yellow]Skip[/yellow] {source_file} (not found)")
        return 0, []

    # Read old format
    try:
        data = yaml.safe_load(source_file.read_text())
    except Exception as e:
        console.print(f"[red]Error[/red] parsing {source_file}: {e}")
        return 0, []

    templates = data.get("templates", {})

    if not templates:
        console.print(f"[yellow]Skip[/yellow] {source_file} (no templates)")
        return 0, []

    # Create target directory
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Migrating {source_file.name}[/bold cyan]")
    console.print(f"  Source: {source_file}")
    console.print(f"  Target: {target_dir}/")

    # Write individual files
    count = 0
    template_names = []
    for name, content in templates.items():
        target_file = target_dir / f"{name}.yaml"
        template_names.append(name)

        if dry_run:
            console.print(f"  [dim]→[/dim] {target_file.name}")
        else:
            # Write direct content (no "templates:" wrapper)
            try:
                yaml_content = yaml.dump(
                    content,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True
                )
                target_file.write_text(yaml_content)
                console.print(f"  [green]✓[/green] {target_file.name}")
            except Exception as e:
                console.print(f"  [red]✗[/red] {target_file.name}: {e}")
                continue

        count += 1

    if not dry_run:
        # Backup old file
        backup_file = source_file.with_suffix(".yaml.bak")
        shutil.copy2(source_file, backup_file)
        console.print(f"  [dim]Backup: {backup_file}[/dim]")

    return count, template_names


def validate_migration(directory: Path) -> List[str]:
    """
    Validate migrated templates for conflicts.

    Args:
        directory: Root templates directory (e.g., config/templates/)

    Returns:
        List of validation error messages
    """
    errors = []

    # Check case-insensitive duplicates in each type
    for template_type in ["model", "preprocess"]:
        type_dir = directory / template_type
        if not type_dir.exists():
            continue

        seen = {}
        for file in type_dir.glob("*.yaml"):
            name_lower = file.stem.lower()
            if name_lower in seen:
                errors.append(
                    f"Case-insensitive conflict in {template_type}/: "
                    f"'{file.stem}' vs '{seen[name_lower]}'"
                )
            seen[name_lower] = file.stem

    # Global name uniqueness removed - same names allowed in model/ and preprocess/
    return errors


def detect_name_conflicts(
    model_templates: List[str],
    preprocess_templates: List[str]
) -> List[str]:
    """
    Detect name conflicts between model and preprocess templates.

    Returns:
        List of conflicting template names
    """
    return list(set(model_templates) & set(preprocess_templates))


def migrate_global_templates(dry_run: bool = True) -> Dict[str, int]:
    """Migrate global templates."""
    global_template_dir = REPO_ROOT / "config" / "templates"

    counts = {}
    all_model_names = []
    all_preprocess_names = []

    for template_type in ["model", "preprocess"]:
        source_file = global_template_dir / f"{template_type}.yaml"
        target_dir = global_template_dir / template_type

        count, names = migrate_template_file(source_file, target_dir, template_type, dry_run)
        counts[template_type] = count

        if template_type == "model":
            all_model_names = names
        else:
            all_preprocess_names = names


    return counts


def migrate_project_templates(project_name: str, dry_run: bool = True) -> Dict[str, int]:
    """Migrate project-local templates."""
    project_root = REPO_ROOT / "projects" / "kaggle" / project_name
    template_dir = project_root / "templates"

    if not template_dir.exists():
        console.print(f"[yellow]No templates directory for project {project_name}[/yellow]")
        return {}

    counts = {}
    for template_type in ["model", "preprocess"]:
        source_file = template_dir / f"{template_type}.yaml"
        target_dir = template_dir / template_type

        count, _ = migrate_template_file(source_file, target_dir, template_type, dry_run)
        counts[template_type] = count

    return counts


def validate_all_templates() -> int:
    """Validate all templates (global and projects)."""
    console.print("[bold]Validating migration...[/bold]\n")

    all_errors = []

    # Validate global
    global_dir = REPO_ROOT / "config" / "templates"
    errors = validate_migration(global_dir)

    if errors:
        console.print("[red]Global template validation errors:[/red]")
        for error in errors:
            console.print(f"  [red]✗[/red] {error}")
        all_errors.extend(errors)
    else:
        console.print("[green]✓ Global templates validated successfully[/green]")

    # Validate all projects
    projects_dir = REPO_ROOT / "projects" / "kaggle"
    if projects_dir.exists():
        for project_dir in sorted(projects_dir.iterdir()):
            if not project_dir.is_dir():
                continue

            template_dir = project_dir / "templates"
            if not template_dir.exists():
                continue

            errors = validate_migration(template_dir)
            if errors:
                console.print(f"\n[red]Validation errors in {project_dir.name}:[/red]")
                for error in errors:
                    console.print(f"  [red]✗[/red] {error}")
                all_errors.extend(errors)
            else:
                console.print(f"[green]✓[/green] {project_dir.name}")

    if all_errors:
        console.print(f"\n[red]Validation failed with {len(all_errors)} error(s)[/red]")
        return 1
    else:
        console.print("\n[green]All templates validated successfully![/green]")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Migrate template YAMLs to individual files")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")
    parser.add_argument("--project", help="Migrate specific project only (e.g., playground-series-s5e12)")
    parser.add_argument("--all-projects", action="store_true", help="Migrate all projects")
    parser.add_argument("--validate", action="store_true", help="Validate migration (run after migration)")

    args = parser.parse_args()

    if args.validate:
        return validate_all_templates()

    mode = "[yellow]DRY RUN[/yellow]" if args.dry_run else "[red]EXECUTING[/red]"
    console.print(f"\n[bold]Template Migration {mode}[/bold]\n")

    total_counts = {"model": 0, "preprocess": 0}

    # Migrate global
    if not args.project:
        console.print("[bold green]Global Templates[/bold green]")
        counts = migrate_global_templates(args.dry_run)

        # If conflicts detected in execute mode, abort
        if not counts and not args.dry_run:
            return 1

        for k, v in counts.items():
            total_counts[k] += v

    # Migrate projects
    if args.project:
        console.print(f"\n[bold green]Project: {args.project}[/bold green]")
        counts = migrate_project_templates(args.project, args.dry_run)
        for k, v in counts.items():
            total_counts[k] += v

    elif args.all_projects:
        console.print("\n[bold green]All Projects[/bold green]")
        projects_dir = REPO_ROOT / "projects" / "kaggle"
        if projects_dir.exists():
            for project_dir in sorted(projects_dir.iterdir()):
                if project_dir.is_dir():
                    counts = migrate_project_templates(project_dir.name, args.dry_run)
                    for k, v in counts.items():
                        total_counts[k] += v

    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Model templates: {total_counts['model']}")
    console.print(f"  Preprocess templates: {total_counts['preprocess']}")
    console.print(f"  Total: {sum(total_counts.values())}")

    if args.dry_run:
        console.print(f"\n[yellow]This was a dry run. Run without --dry-run to execute.[/yellow]")
    else:
        console.print(f"\n[green]Migration complete! Old files backed up with .bak extension.[/green]")
        console.print(f"\nNext steps:")
        console.print(f"  1. Run: python scripts/migrate_templates.py --validate")
        console.print(f"  2. Test with: uv run python scripts/mla.py model --project <project> --model-template list")
        console.print(f"  3. If successful, commit changes")

    return 0


if __name__ == "__main__":
    exit(main())
