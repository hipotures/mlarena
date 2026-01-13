import json
import random
import numpy as np
from typing import Any, Dict, List

class ParameterSampler:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def sample(self, name: str, spec: Dict[str, Any]) -> Any:
        ptype = spec.get("type")
        
        if ptype == "choice":
            values = spec.get("values", [])
            if not values:
                return None
            return self.rng.choice(values)
            
        elif ptype == "int_range":
            min_val = int(spec.get("min", 0))
            max_val = int(spec.get("max", 10))
            step = spec.get("step")
            if step:
                # Use range logic for steps
                possible = list(range(min_val, max_val + 1, int(step)))
                if not possible:
                    return min_val
                return self.rng.choice(possible)
            return self.rng.randint(min_val, max_val)
            
        elif ptype == "float_range":
            min_val = float(spec.get("min", 0.0))
            max_val = float(spec.get("max", 1.0))
            log = bool(spec.get("log", False))
            
            if log:
                # Log-uniform sampling
                # Handle non-positive range for log sampling carefully, assuming user inputs valid >0 for log
                if min_val <= 0: min_val = 1e-10
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                val = np.exp(self.np_rng.uniform(log_min, log_max))
            else:
                val = self.rng.uniform(min_val, max_val)
                
            step = spec.get("step")
            if step:
                # Quantize to step
                val = round(val / step) * step
                
            return float(val)
            
        elif ptype == "bool":
            return self.rng.choice([True, False])
            
        elif ptype == "fixed":
            return spec.get("value")
            
        else:
            # Unknown type, return None or raise? 
            # For robustness, returning None or ignoring might be safer than crashing search.
            # But let's raise to be explicit.
            raise ValueError(f"Unsupported param type: {ptype}")
