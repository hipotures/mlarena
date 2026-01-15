import pytest
from tests.unit.preprocess.utils import expand_param_spec

def test_search_spaces_structure(all_search_spaces):
    """Weryfikuje strukture plikow YAML w search_spaces."""
    assert len(all_search_spaces) > 0, "Nie znaleziono plikow search spaces"
    
    required_keys = ["base_config", "variants"]
    
    for module, space in all_search_spaces.items():
        # 1. Glowne klucze
        for k in required_keys:
            assert k in space, f"Module {module} missing required key: {k}"
            
        # 2. Base config musi byc slownikiem
        assert isinstance(space["base_config"], dict), f"{module}: base_config must be dict"
        
        # 3. Variants
        variants = space["variants"]
        assert isinstance(variants, (dict, list)), f"{module}: variants must be dict or list"
        assert len(variants) > 0, f"{module}: variants list is empty"
        
        # Normalize to dict for validation
        if isinstance(variants, list):
            variants_map = {v.get("name", f"idx_{i}"): v for i, v in enumerate(variants)}
        else:
            variants_map = variants

        for v_name, v_spec in variants_map.items():
            # Params (opcjonalne, ale jesli sa to dict)
            if "params" in v_spec:
                assert isinstance(v_spec["params"], dict), f"{module}.{v_name}: params must be dict"
                
                # Weryfikacja typow parametrow
                for p_name, p_spec in v_spec["params"].items():
                    known_types = ["choice", "int_range", "float_range", "bool", "fixed", "subset"]
                    
                    # Sprawdz czy mamy klucz 'type'
                    if "type" in p_spec:
                        p_type = p_spec["type"]
                        assert p_type in known_types, \
                            f"{module}.{v_name}.{p_name}: unknown param type '{p_type}'. Expected one of {known_types}"
                    else:
                        # Sprawdz czy klucze slownika zawieraja znany typ (stary format)
                        assert any(t in p_spec for t in known_types), \
                            f"{module}.{v_name}.{p_name}: unknown param type structure. Expected one of {known_types} keys or 'type' field."
                    
                    # Proba rozwiniecia (czy nie rzuca bledem)
                    vals = expand_param_spec(p_spec)
                    assert len(vals) > 0, f"{module}.{v_name}.{p_name}: param spec generated no values"
