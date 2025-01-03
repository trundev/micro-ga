"""Test arithmetic operations"""
import numpy as np
import pytest
import micro_ga
from . import pos_sig, layout, operation, dtype # pylint: disable=W0611
# pylint: disable=W0621


@pytest.fixture
def neg_sig():
    """Skip tests with basis-vectors of negative signature"""
    return 0

@pytest.fixture
def zero_sig():
    """Skip tests with basis-vectors of zero signature"""
    return 0

def test_operation(layout, operation):
    """Compare results from arithmetic operations with scalar vs. python integer"""
    # Pick the middle blade
    blade = tuple(layout.blades.values())[layout.gaDims//2]
    assert operation(layout.scalar, blade) == operation(1, blade)
    assert operation(blade, layout.scalar) == operation(blade, 1)
    # Unsupported operand type (right and left side)
    with pytest.raises(TypeError):
        _ = operation(blade, None)
    with pytest.raises(TypeError):
        _ = operation(None, blade)

def test_blade_dtype(dtype):
    """Check the internal `numpy` array `dtype` of all blades"""
    layout = micro_ga.Cl(3, dtype=dtype)
    # Note: `dtype('O') == object`
    assert layout.scalar.value.dtype == dtype, 'Internal numpy array must use requested dtype'
    for blade in layout.blades.values():
        assert blade.value.dtype == dtype
    # Check type of individual values from the scalar-blade
    exp_t = int if dtype is object else dtype
    for v in layout.scalar.value:
        # pylint: disable=C0123     # Here we expect exactly the same type
        assert type(v) is exp_t, 'Individual values must be of requested type'

    # Check if both string representations work
    _ = repr(layout.scalar)
    _ = str(layout.scalar)

def test_operation_dtype(operation, dtype):
    """Check the internal `numpy` array `dtype` of operation result"""
    layout = micro_ga.Cl(3, dtype=dtype)
    mv = operation(layout.mvector(12345).astype(dtype), layout.scalar)
    exp_dt = np.result_type(layout.scalar.value.dtype, dtype)
    assert mv.value.dtype is exp_dt, 'Result dtype must come from numpy conversion rules'
    # Check type of individual values
    exp_t = int if dtype is object else dtype
    for v in mv.value:
        # pylint: disable=C0123     # Here we expect exactly the same type
        assert type(v) is exp_t, 'Individual values of result must be of requested type'

def test_unbounded_int():
    """Test python unbounded `int` operation"""
    # With `object`, `numpy` falls-back to original python unbounded operation
    layout = micro_ga.Cl(2, dtype=object)
    mv = layout.scalar + (1<<100)
    assert (mv.value[0] - (1<<100)) == 1

    # With `int`, `numpy` uses `int64`, which is 64-bit only
    layout = micro_ga.Cl(2, dtype=int)
    with pytest.raises(OverflowError):
        mv = layout.scalar + (1<<100)

def test_round(dtype):
    """Test `round` operation"""
    layout = micro_ga.Cl(2, dtype=dtype)
    if dtype is object:
        exp_dtype = int         # Default `micro_ga` type
    else:
        exp_dtype = dtype
    # Pick a convenient number, which after rounding has finite binary representation
    val = exp_dtype(1.2456) + layout.I
    val = round(val, 2)
    assert val == exp_dtype(1.25) + layout.I
