"""
AutoGluon HPO Search Space Parser

Converts YAML list notation to autogluon.common.space objects.

Supported formats:
- [min, max, 'log'] → space.Real(min, max, log=True)
- [min, max, 'int'] → space.Int(min, max)
- [min, max] (int types) → space.Int(min, max)
- [min, max] (float types) → space.Real(min, max)
- [val1, val2, val3] (categorical) → space.Categorical(val1, val2, val3)

Example:
    >>> from mlarena.utils.hpo_space import parse_search_space
    >>> space_dict = {
    ...     'GBM': {
    ...         'learning_rate': [0.01, 0.3, 'log'],
    ...         'num_leaves': [20, 150],
    ...     }
    ... }
    >>> parsed = parse_search_space(space_dict)
    >>> # parsed['GBM']['learning_rate'] = space.Real(0.01, 0.3, log=True)
    >>> # parsed['GBM']['num_leaves'] = space.Int(20, 150)
"""
from typing import Any, Dict, List, Union

from autogluon.common import space


def parse_search_space(space_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Convert YAML search space notation to autogluon.common.space objects.

    Args:
        space_dict: Nested dict {model_type: {param: [min, max, type]}}

    Returns:
        Nested dict with autogluon.common.space objects

    Raises:
        ValueError: If search space specification is invalid
    """
    result = {}

    for model_type, params in space_dict.items():
        result[model_type] = {}

        for param_name, param_spec in params.items():
            try:
                result[model_type][param_name] = _parse_param_spec(param_name, param_spec)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse search space for {model_type}.{param_name}: {param_spec}. "
                    f"Error: {str(e)}"
                ) from e

    return result


def _parse_param_spec(param_name: str, param_spec: Union[List, Any]) -> Any:
    """
    Parse a single parameter specification into autogluon.common.space object.

    Args:
        param_name: Parameter name (for error messages)
        param_spec: List specification or already-converted space object

    Returns:
        autogluon.common.space object

    Raises:
        ValueError: If specification is invalid
    """
    # Pass through non-list values (already converted or static)
    if not isinstance(param_spec, list):
        return param_spec

    # Validate minimum length
    if len(param_spec) < 2:
        raise ValueError(
            f"Search space spec must have at least 2 elements [min, max], got: {param_spec}"
        )

    # Check if last element is a type specifier
    type_hint = None
    values = param_spec
    if len(param_spec) >= 3:
        last_elem = param_spec[-1]
        if isinstance(last_elem, str) and last_elem in ['log', 'int', 'float']:
            # Last element is type hint
            type_hint = last_elem
            values = param_spec[:-1]

    # Categorical: more than 2 values (after removing type hint) OR non-numeric
    if len(values) > 2:
        # Categorical with multiple options
        return space.Categorical(*values)

    if len(values) == 2 and not isinstance(values[0], (int, float)):
        # Categorical with 2 non-numeric values
        return space.Categorical(*values)

    # Numeric range: [min, max] (with optional type hint already extracted)
    lower, upper = values[0], values[1]

    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        raise ValueError(
            f"Numeric range must have numeric min/max values, got: {param_spec}"
        )

    if lower >= upper:
        raise ValueError(
            f"Search space min ({lower}) must be less than max ({upper})"
        )

    # Determine space type
    if type_hint == 'log':
        # Real with log scale
        return space.Real(lower, upper, log=True)
    elif type_hint == 'int' or (isinstance(lower, int) and isinstance(upper, int) and type_hint != 'float'):
        # Integer space (explicit 'int' or both values are int and no 'float' hint)
        return space.Int(lower, upper)
    else:
        # Real (linear) - default for floats or explicit 'float'
        return space.Real(lower, upper)


def _all_numeric(values: List) -> bool:
    """Check if all values in list are numeric."""
    return all(isinstance(v, (int, float)) for v in values)


def validate_search_space(space_dict: Dict[str, Any]) -> List[str]:
    """
    Validate search space format and return warnings.

    Args:
        space_dict: Search space dictionary to validate

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    known_model_types = {'GBM', 'XGB', 'CAT', 'NN_TORCH', 'FASTAI', 'RF', 'XT', 'KNN'}

    for model_type, params in space_dict.items():
        if model_type not in known_model_types:
            warnings.append(f"Unknown model type: {model_type}")

        if not isinstance(params, dict):
            warnings.append(f"Parameters for {model_type} should be a dict, got {type(params)}")
            continue

        for param_name, spec in params.items():
            if isinstance(spec, list):
                if len(spec) < 2:
                    warnings.append(
                        f"Invalid spec for {model_type}.{param_name}: {spec} "
                        "(should have at least 2 elements)"
                    )

    return warnings
