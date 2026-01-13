import pytest
import numpy as np
from mlarena.modules.mcts.sampler import ParameterSampler

def test_sample_choice():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "choice", "values": ["a", "b", "c"]}
    
    # Run multiple times to ensure we get valid values
    for _ in range(10):
        val = sampler.sample("p1", spec)
        assert val in ["a", "b", "c"]

def test_sample_int_range():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "int_range", "min": 1, "max": 10}
    
    val = sampler.sample("p2", spec)
    assert isinstance(val, int)
    assert 1 <= val <= 10

def test_sample_float_range():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "float_range", "min": 0.1, "max": 0.5}
    
    val = sampler.sample("p3", spec)
    assert isinstance(val, float)
    assert 0.1 <= val <= 0.5

def test_sample_bool():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "bool"}
    
    val = sampler.sample("p4", spec)
    assert isinstance(val, bool)

def test_determinism():
    s1 = ParameterSampler(seed=123)
    s2 = ParameterSampler(seed=123)
    spec = {"type": "int_range", "min": 1, "max": 100}
    
    assert s1.sample("p", spec) == s2.sample("p", spec)
