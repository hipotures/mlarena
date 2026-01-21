from pathlib import Path
from mlarena.modules.mcts.space import SuperChainActionSpace
from mlarena.modules.mcts.node import PipelineState

# Setup
conf_path = Path("conf/preprocess/mla_super_chain.yaml")
space = SuperChainActionSpace(conf_path)

# Mock State matching the error
# last_index=8, groups=['encoding', 'rank_pre', 'feature_group_agg', 'missingness_features']
used_groups_list = ['encoding', 'rank_pre', 'feature_group_agg', 'missingness_features']
used_groups = {g: "debug" for g in used_groups_list}
last_index = 8

state = PipelineState(
    steps=[],
    depth=4,
    used_groups=used_groups,
    last_step_index=last_index
)

print(f"State last_index: {state.last_step_index}")
print(f"Space total steps: {len(space.steps)}")

# Re-implement loop to see what happens
start_index = state.last_step_index + 1
end_index = len(space.steps)
print(f"Scanning range: {start_index} to {end_index}")

for i in range(start_index, end_index):
    step_def = space.steps[i]
    step_name = step_def.get("name")
    group = step_def.get("group") or step_name
    template_name = step_def.get("template") or step_name
    
    print(f"Checking index {i}: {step_name} (Group: {group})")
    
    if group in state.used_groups:
        print(f"  -> Skipped: Group {group} already used")
        continue
        
    space_def = space.search_spaces.get(template_name, {})
    variants = space_def.get("variants", [])
    print(f"  -> Variants found: {len(variants)}")
    
    for v in variants:
        vname = v.get("name")
        reqs = v.get("requires_preproc", [])
        met = True
        missing = []
        for req in reqs:
            rg = req.get("group")
            if rg and rg not in state.used_groups:
                met = False
                missing.append(rg)
        
        print(f"    -> Variant {vname}: Met={met} (Missing: {missing})")

# Call actual method
actions = space.next_actions(state, lookahead=1)
print(f"Total actions returned: {len(actions)}")
