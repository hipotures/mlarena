from pathlib import Path
from mlarena.modules.mcts.space import SuperChainActionSpace

conf_path = Path("conf/preprocess/mla_super_chain.yaml")
space = SuperChainActionSpace(conf_path)

print(f"Total steps: {len(space.steps)}")
for i, step in enumerate(space.steps):
    print(f"Index {i}: {step['name']} (Group: {step.get('group')})")
