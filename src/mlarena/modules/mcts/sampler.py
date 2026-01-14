import json
import random
import numpy as np
from typing import Any, Dict, List, Optional

class ParameterSampler:
    def __init__(self, seed: int = 42):
        self.seed = seed
        # Default RNGs
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def sample_variant(self, template_name: str, variant_name: str, search_spaces: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Sample a full configuration for a specific variant using a provided or default RNG."""
        space = search_spaces.get(template_name, {})
        variants = space.get("variants", [])
        
        # Find the specific variant spec
        variant_spec = next((v for v in variants if v.get("name") == variant_name), None)
        if not variant_spec:
            return {}
            
        params_spec = variant_spec.get("params", {})
        config = {}
        
        for p_name, p_spec in params_spec.items():
            config[p_name] = self.sample_param(p_spec, rng=rng)
            
        return config

    def sample_param(self, spec: Dict[str, Any], rng: Optional[random.Random] = None) -> Any:
        """Sample a single parameter based on its specification."""
        # Use provided local RNG or fallback to self.rng
        active_rng = rng if rng is not None else self.rng
        
        ptype = spec.get("type")
        
        if ptype == "choice":
            values = spec.get("values", [])
            if not values:
                return None
            return active_rng.choice(values)
            
        elif ptype == "int_range":
            min_val = int(spec.get("min", 0))
            max_val = int(spec.get("max", 10))
            step = spec.get("step")
            if step:
                possible = list(range(min_val, max_val + 1, int(step)))
                if not possible:
                    return min_val
                return active_rng.choice(possible)
            return active_rng.randint(min_val, max_val)
            
        elif ptype == "float_range":
            min_val = float(spec.get("min", 0.0))
            max_val = float(spec.get("max", 1.0))
            log = bool(spec.get("log", False))
            
            if log:
                # For log sampling with local RNG, we use uniform and exp
                if min_val <= 0: min_val = 1e-10
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                # random.uniform is available in random.Random
                val = np.exp(active_rng.uniform(log_min, log_max))
            else:
                val = active_rng.uniform(min_val, max_val)
                
            step = spec.get("step")
            if step:
                val = round(val / step) * step
                
            return float(val)
            
        elif ptype == "bool":
            return active_rng.choice([True, False])
            
        elif ptype == "fixed":
            return spec.get("value")
            
        elif ptype == "subset":
            values = spec.get("values", [])
            if not values:
                return []
            min_items = int(spec.get("min_items", 1))
            max_items = int(spec.get("max_items", len(values)))
            
            max_items = min(max_items, len(values))
            min_items = min(min_items, max_items)
            
            k = active_rng.randint(min_items, max_items)
            chosen = active_rng.sample(values, k)
            
            if bool(spec.get("sort", False)):
                try:
                    chosen.sort()
                except Exception:
                    pass
            return chosen
            
        else:
            raise ValueError(f"Unsupported param type: {ptype}")

    def sample(self, template_name: str, variant_name: str, search_spaces: Dict[str, Any], rng: Optional[random.Random] = None) -> Dict[str, Any]:
        """Legacy/Bridge method to match the call site in tree.py."""
        return self.sample_variant(template_name, variant_name, search_spaces, rng=rng)