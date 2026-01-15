import importlib
import pytest
from typing import Any, Dict, List
import math

class FunctionalModuleWrapper:
    """Wrapper dla modulow zaimplementowanych jako funkcje fit_transform."""
    def __init__(self, mod, config):
        self.mod = mod
        self.config = config
        
    def fit_transform(self, train, val, test):
        # Sygnatura funkcyjna moze sie roznic:
        # fit_transform(train_df, val_df, test_df, config, orig_df=None) -> tuple
        # fit_transform(train, val, test, config) -> tuple
        
        # Sprawdzamy czy funkcja przyjmuje orig_df
        from inspect import signature
        sig = signature(self.mod.fit_transform)
        
        kwargs = {"config": self.config}
        if "orig_df" in sig.parameters:
            kwargs["orig_df"] = None
            
        return self.mod.fit_transform(train, val, test, **kwargs)

def get_processor_class(module_name: str):
    """Dynamicznie importuje klase procesora lub tworzy wrapper dla funkcji."""
    try:
        class_name = "".join(x.title() for x in module_name.split("_"))
        if module_name == "sanity_check":
            class_name = "SanityCheck"
            
        module_path = f"mlarena.defaults.preprocessing.{module_name}"
        mod = importlib.import_module(module_path)
        
        # 1. Szukaj klasy
        if hasattr(mod, class_name):
            return getattr(mod, class_name)
            
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and attr_name.lower() == class_name.lower():
                return attr
                
        # 2. Szukaj funkcji fit_transform (podejscie funkcyjne)
        if hasattr(mod, "fit_transform") and callable(mod.fit_transform):
            # Zwracamy fabryke, ktora zachowuje sie jak konstruktor klasy
            return lambda config: FunctionalModuleWrapper(mod, config)
                
        raise ImportError(f"Could not find class {class_name} or fit_transform function in {module_path}")
    except ImportError as e:
        print(f"WARNING: Module {module_name} not available or import failed: {e}")
        return None

def expand_param_spec(spec: Dict[str, Any]) -> List[Any]:
    """Generuje liste wartosci testowych dla danej specyfikacji parametru."""
    vals = []
    
    # Normalizacja specyfikacji: jesli jest 'type', mapujemy na stary format dla uproszczenia
    spec_type = spec.get("type")
    
    if spec_type == "choice" or "choice" in spec:
        values = spec.get("values", spec.get("choice"))
        if values is not None:
            vals.extend(values)
            
    elif spec_type == "int_range" or "int_range" in spec:
        # Obsluga formatu {type: int_range, min: X, max: Y} oraz {int_range: [min, max]}
        if "int_range" in spec and isinstance(spec["int_range"], list):
            low, high = spec["int_range"][0], spec["int_range"][1]
        else:
            low, high = spec.get("min"), spec.get("max")
            
        if low is not None and high is not None:
            vals.append(low)
            vals.append(high)
            if high - low > 1:
                vals.append((low + high) // 2)
            
    elif spec_type == "float_range" or "float_range" in spec:
        if "float_range" in spec and isinstance(spec["float_range"], list):
            low, high = spec["float_range"][0], spec["float_range"][1]
        else:
            low, high = spec.get("min"), spec.get("max")
            
        if low is not None and high is not None:
            vals.append(low)
            vals.append(high)
            vals.append((low + high) / 2.0)
        
    elif spec_type == "bool" or "bool" in spec:
        vals.extend([True, False])
        
    elif spec_type == "subset":
        # Generujemy kilka przykladowych podzbiorow
        possible_values = spec.get("values", [])
        if possible_values:
            # 1. Pojedynczy element (pierwszy)
            vals.append([possible_values[0]])
            
            # 2. Wszystkie elementy (jesli max_items pozwala)
            max_items = spec.get("max_items", len(possible_values))
            vals.append(possible_values[:max_items])
            
            # 3. Jesli lista ma wiecej niz 1 element, wez tez drugi (dla roznorodnosci)
            if len(possible_values) > 1:
                 vals.append([possible_values[1]])
        
    elif spec_type == "fixed" or "fixed" in spec:
        val = spec.get("value", spec.get("fixed"))
        # Fixed moze byc 0 lub False, wiec sprawdzamy in keys
        if val is not None or "value" in spec or "fixed" in spec:
            vals.append(val)
        
    else:
        # Fallback dla nieznanych typow lub zagniezdzonych struktur
        return []

    # Odsiewanie duplikatow (np. choice: [1, 1])
    # Ale uwaga na typy niehashowalne (listy, dicty)
    unique_vals = []
    seen = set()
    for v in vals:
        try:
            if v not in seen:
                unique_vals.append(v)
                seen.add(v)
        except TypeError:
            # Wartosc niehashowalna, dodajemy zawsze
            unique_vals.append(v)
            
    return unique_vals

def generate_test_cases(all_search_spaces):
    """Generator zwracajacy krotki (module_name, variant_name, param_name, param_value, base_config)."""
    cases = []
    for module_name, space in all_search_spaces.items():
        base_config = space.get("base_config", {})
        variants_data = space.get("variants", {})
        
        # Normalize variants to dictionary {name: spec}
        variants_map = {}
        if isinstance(variants_data, list):
            for v in variants_data:
                v_name = v.get("name", "unknown")
                variants_map[v_name] = v
        elif isinstance(variants_data, dict):
            variants_map = variants_data
        
        for variant_name, variant_spec in variants_map.items():
            params = variant_spec.get("params", {})
            if not params:
                # Wariant bez parametrow
                cases.append((module_name, variant_name, {}, base_config))
                continue
                
            # Dla uproszczenia testow "exhaustive", generujemy configi zmieniajac JEDEN parametr na raz
            # wzgledem wartosci domyslnych (lub pierwszych z listy).
            # To strategia "One-Factor-At-A-Time" dla weryfikacji crashy.
            
            # 1. Najpierw zbuduj "default config" dla wariantu (biorac pierwsze wartosci)
            default_variant_cfg = {}
            expanded_params = {}
            
            for p_name, p_spec in params.items():
                possible_vals = expand_param_spec(p_spec)
                if not possible_vals:
                    continue # Skip invalid/empty specs
                expanded_params[p_name] = possible_vals
                default_variant_cfg[p_name] = possible_vals[0]
            
            # 2. Generuj przypadki testowe
            # Case 0: Default config
            cases.append((module_name, variant_name, default_variant_cfg.copy(), base_config))
            
            # Case N: Zmieniaj kazdy parametr na jego inne mozliwe wartosci
            for p_name, vals in expanded_params.items():
                for val in vals[1:]: # Pomin pierwszy, bo juz jest w default
                    test_cfg = default_variant_cfg.copy()
                    test_cfg[p_name] = val
                    cases.append((module_name, variant_name, test_cfg, base_config))
                    
    return cases
