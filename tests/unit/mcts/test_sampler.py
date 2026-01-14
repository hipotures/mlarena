import pytest
import numpy as np
import random
from mlarena.modules.mcts.sampler import ParameterSampler

def test_sample_choice():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "choice", "values": ["a", "b", "c"]}
    
    # Run multiple times to ensure we get valid values
    for _ in range(10):
        val = sampler.sample_param(spec)
        assert val in ["a", "b", "c"]

def test_sample_int_range():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "int_range", "min": 1, "max": 10}
    
    val = sampler.sample_param(spec)
    assert isinstance(val, int)
    assert 1 <= val <= 10

def test_sample_float_range():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "float_range", "min": 0.1, "max": 0.5}
    
    val = sampler.sample_param(spec)
    assert isinstance(val, float)
    assert 0.1 <= val <= 0.5

def test_sample_bool():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "bool"}
    
    val = sampler.sample_param(spec)
    assert isinstance(val, bool)

def test_determinism():
    s1 = ParameterSampler(seed=123)
    s2 = ParameterSampler(seed=123)
    spec = {"type": "int_range", "min": 1, "max": 100}
    
    # We must use a local RNG to ensure independence from state if desired,
    # but here we test the internal state consistency
    assert s1.sample_param(spec) == s2.sample_param(spec)

def test_local_rng_isolation():
    sampler = ParameterSampler(seed=42)
    spec = {"type": "int_range", "min": 1, "max": 1000000}
    
    rng1 = random.Random(100)
    rng2 = random.Random(100)
    
    val1 = sampler.sample_param(spec, rng=rng1)
    val2 = sampler.sample_param(spec, rng=rng2)
    
    assert val1 == val2