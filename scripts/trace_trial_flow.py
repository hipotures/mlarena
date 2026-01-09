#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.tree import Tree
from rich import box

def get_detailed_shapes(module_dir: Path):
    """Wyciąga wymiary wszystkich plików z state.json."""
    state_path = module_dir / "state.json"
    if not state_path.exists():
        return {}

    try:
        with open(state_path, "r") as f:
            data = json.load(f)
        
        payload = {}
        if "modules" in data and "preprocess" in data["modules"]:
            payload = data["modules"]["preprocess"].get("payload", {})
        elif "payload" in data:
            payload = data["payload"]

        shapes = payload.get("shapes", {})
        cms = payload.get("custom_module_state", {})
        
        results = {}
        # Lista wszystkich możliwych plików
        files = ["train", "test", "tuning", "eval", "orig"]
        
        for f in files:
            # 1. Próbujemy z shapes (after)
            s_after = shapes.get(f"{f}_after")
            if s_after and isinstance(s_after, list) and len(s_after) == 2:
                results[f] = s_after
                continue
            
            # 2. Próbujemy z shapes (before) jako fallback dla liczby kolumn
            s_before = shapes.get(f"{f}_before")
            
            # 3. Próbujemy wyciągnąć liczbę wierszy z custom_module_state
            row_key = f"{f}_rows"
            rows = cms.get(row_key)
            
            if rows is not None:
                # Kolumny bierzemy z before jeśli po jest null
                cols = s_before[1] if s_before else "??"
                results[f] = [rows, cols]
        
        return results
    except Exception:
        return {}

def main():
    parser = argparse.ArgumentParser(description="Szczegółowa wizualizacja wymiarów plików w trialu MLArena")
    parser.add_argument("trial_path", help="Ścieżka do katalogu trialu (np. trial_4728)")
    args = parser.parse_args()

    trial_path = Path(args.trial_path).resolve()
    if not trial_path.exists():
        print(f"Błąd: Ścieżka {trial_path} nie istnieje.")
        return

    console = Console()

    # Znajdź foldery modułów i posortuj numerycznie
    module_dirs = sorted(
        [d for d in trial_path.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda d: int(d.name.split("-")[0])
    )

    flow_elements = []

    # Nagłówek
    flow_elements.append(Panel(f"[bold magenta]TRIAL FLOW: {trial_path.name}[/]", box=box.DOUBLE, expand=False, border_style="magenta"))
    flow_elements.append(Text("↓", style="bold yellow"))

    for i, m_dir in enumerate(module_dirs):
        module_name = m_dir.name
        shapes = get_detailed_shapes(m_dir)
        
        # Tworzymy drzewo dla plików
        tree = Tree(f"[bold cyan]{module_name}[/]", guide_style="dim green")
        
        if shapes:
            for f_type, dims in shapes.items():
                rows, cols = dims
                rows_str = f"{rows:,}" if rows is not None else "???"
                tree.add(f"📄 [white]{f_type:6}[/] [dim]([yellow]{rows_str}[/] × [bold green]{cols}[/])")
        else:
            tree.add("[italic dim]brak danych o wymiarach[/]")

        flow_elements.append(Panel(tree, box=box.ROUNDED, expand=False, border_style="blue"))
        
        if i < len(module_dirs) - 1:
            flow_elements.append(Text("↓", style="bold yellow"))

    # Wyświetlenie wyśrodkowane
    aligned_flow = [Align.center(el) for el in flow_elements]
    
    console.print()
    console.print(Panel(
        Group(*aligned_flow),
        title="[bold white]Preprocessing Data Dimensions Trace[/bold white]",
        border_style="bright_blue",
        padding=(1, 2)
    ))
    console.print()

if __name__ == "__main__":
    main()