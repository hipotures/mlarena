from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Tree, Static
from rich.json import JSON

DATA = {
    "model_template": "autogluon_thin_fast",
    "preprocess": {"template": "optuna_trial", "force": True, "seed": 42},
    "use_gpu": False,
}

class UI(App):
    CSS = """
    Horizontal { height: 100%; }
    #nav { width: 30; border: solid; }
    #json { width: 1fr; border: solid; overflow: auto; }
    """

    def compose(self) -> ComposeResult:
        tree = Tree("Root", id="nav")
        tree.root.add("config.json")  # tylko gałąź/etykieta
        tree.root.expand()

        # JSON jest renderowany w normalnym widżecie, nie w etykiecie Tree
        viewer = Static(JSON.from_data(DATA), id="json")

        yield Horizontal(tree, viewer)

if __name__ == "__main__":
    UI().run()