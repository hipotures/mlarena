#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

def get_module_type(dir_path: Path) -> str:
    """Tries to guess module type from directory name or summary.json."""
    # 1. Try summary.json
    summary_path = dir_path / "artifacts" / "preprocess" / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                data = json.load(f)
                if "template" in data:
                    # Often template name contains module name, but let's check config/module if available
                    # Actually standard summary.json has "template"
                    # Let's clean the dirname
                    pass
        except:
            pass

    # 2. Clean dirname
    name = dir_path.name
    # Remove index (e.g. "0-")
    if "-" in name and name.split("-")[0].isdigit():
        name = name.split("-", 1)[1]
    
    # Remove project prefixes
    prefixes = ["test_c_01_0306-", "mcts-"]
    for p in prefixes:
        if name.startswith(p):
            name = name.replace(p, "")
            
    # Handle "mcts" mapped to sanity_check or similar if needed? 
    # Actually based on user logs: "0-mcts" -> Module: train_fraction
    if name == "mcts":
        return "train_fraction" # Based on user logs
        
    return name

def load_summary(step_dir: Path):
    """Finds and loads summary.json from a step directory."""
    # Try recursive search inside artifacts/preprocess
    start_dir = step_dir / "artifacts" / "preprocess"
    if start_dir.exists():
        found = list(start_dir.rglob("summary.json"))
        # Prefer the one in submodules if multiple exist, or just the first one
        # Usually there is only one relevant summary per step
        if found:
            # Sort by length to pick the deepest one (often the module specific one) or shallowest?
            # Actually, standard flow saves transformation summary at .../preprocess/summary.json
            # and module report at .../preprocess/submodules/.../summary.json
            # We want the transformation summary usually, but let's see.
            
            # Let's verify content. We want the one with "shapes" or "config".
            for p in found:
                try:
                    with open(p, "r") as f:
                        data = json.load(f)
                        if "shapes" in data or "config" in data:
                            return data
                except:
                    continue
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare summaries of two experiments module-by-module.")
    parser.add_argument("path_a", help="Path to NFS trial (e.g. .../trial_4728)")
    parser.add_argument("path_b", help="Path to Local chain (e.g. .../04cbee354b9b)")
    args = parser.parse_args()

    path_a = Path(args.path_a)
    path_b = Path(args.path_b)
    
    console = Console()

    if not path_a.exists():
        console.print(f"[red]Path A not found:[/red] {path_a}")
        return
    if not path_b.exists():
        console.print(f"[red]Path B not found:[/red] {path_b}")
        return

    # Helper to get directories sorted by index
    def get_dirs(path):
        dirs = [d for d in path.iterdir() if d.is_dir() and d.name[0].isdigit()]
        return sorted(dirs, key=lambda d: int(d.name.split("-")[0]))

    dirs_a = get_dirs(path_a)
    dirs_b = get_dirs(path_b)

    # Group by module type
    map_a = {get_module_type(d): d for d in dirs_a}
    map_b = {get_module_type(d): d for d in dirs_b}
    
    all_modules = sorted(list(set(map_a.keys()) | set(map_b.keys())))

    console.print(Panel(f"[bold]Comparing:[/bold]\n A (NFS): {path_a.name}\n B (Local): {path_b.name}", style="magenta", expand=False))

    keys_of_interest = [
        "use_original_features_only", "poly_degree", "poly_interaction_only", 
        "max_generated_features", "interaction_types", "n_features", 
        "selection_method", "encoding_method", "outlier_method", 
        "action", "scaling_method", "train_fraction", "valid_fraction"
    ]

    for mod in all_modules:
        d_a = map_a.get(mod)
        d_b = map_b.get(mod)
        
        sum_a = load_summary(d_a) if d_a else None
        sum_b = load_summary(d_b) if d_b else None
        
        # Determine presence
        status_a = f"[green]{d_a.name}[/]" if d_a else "[red]MISSING[/]"
        status_b = f"[green]{d_b.name}[/]" if d_b else "[red]MISSING[/]"
        
        table = Table(title=f"Module: [bold cyan]{mod}[/]", box=box.ROUNDED, show_lines=True)
        table.add_column("Property", style="dim")
        table.add_column(f"NFS ({status_a})", style="yellow")
        table.add_column(f"Local ({status_b})", style="green")
        
        has_content = False

        # 1. Compare Shapes (Train After)
        def get_shape(s):
            if not s: return "N/A"
            sh = s.get("shapes", {}).get("train_after")
            if sh: return f"{sh[0]:,} x {sh[1]}"
            return "N/A"

        sh_a = get_shape(sum_a)
        sh_b = get_shape(sum_b)
        
        if sh_a != sh_b:
            table.add_row("Output Shape", f"[bold red]{sh_a}[/]", f"[bold red]{sh_b}[/]")
            has_content = True
        else:
            table.add_row("Output Shape", sh_a, sh_b)

        # 2. Compare Config
        cfg_a = sum_a.get("config", {}) if sum_a else {}
        cfg_b = sum_b.get("config", {}) if sum_b else {}
        
        all_cfg = sorted(list(set(cfg_a.keys()) | set(cfg_b.keys())))
        
        for k in all_cfg:
            v_a = cfg_a.get(k, "N/A")
            v_b = cfg_b.get(k, "N/A")
            
            s_a = str(v_a)
            s_b = str(v_b)
            
            # Highlight differences
            if s_a != s_b:
                # Highlight if it's a key of interest
                style = "bold red" if k in keys_of_interest else "white"
                table.add_row(k, f"[{style}]{s_a}[/]", f"[{style}]{s_b}[/]")
                has_content = True
            elif k in keys_of_interest:
                table.add_row(k, s_a, s_b)
                
        if has_content or d_a or d_b:
            console.print(table)
            console.print()

if __name__ == "__main__":
    main()
