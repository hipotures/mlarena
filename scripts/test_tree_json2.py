from textual.app import App, ComposeResult
from textual.widgets import Tree
import json

DATA = {
    "model_template": "autogluon_thin_fast",
    "preprocess": {"template": "optuna_trial", "force": True, "seed": 42},
    "use_gpu": False,
}

class TreeJsonApp(App):
    def compose(self) -> ComposeResult:
        self.tree_widget = Tree("Root")
        self.cfg_node = self.tree_widget.root.add("config.json", expand=False)
        self.tree_widget.root.expand()
        yield self.tree_widget

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        if event.node is not self.cfg_node:
            return

        self.cfg_node.remove_children()
        pretty = json.dumps(DATA, indent=2, ensure_ascii=False)
        for line in pretty.splitlines():
            self.cfg_node.add_leaf(line)

    def on_tree_node_collapsed(self, event: Tree.NodeCollapsed) -> None:
        if event.node is not self.cfg_node:
            return

        self.cfg_node.remove_children()

if __name__ == "__main__":
    TreeJsonApp().run()

