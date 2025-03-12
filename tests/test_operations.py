"""Test arithmetic operations"""
import operator
import inspect
import numpy as np
import pytest
import micro_ga
from . import rng, pos_sig, layout, operation, dtype, exp_dtype, \
        mvector_gen   # pylint: disable=W0611
# pylint: disable=W0621


@pytest.fixture
def neg_sig():
    """Skip tests with basis-vectors of negative signature"""
    return 0

@pytest.fixture
def zero_sig():
    """Skip tests with basis-vectors of zero signature"""
    return 0

def test_operation(layout, operation, mvector_gen):
    """Compare results from arithmetic operations with scalar vs. python integer"""
    # Test built-in operators using a multi-vector as the second argument
    isbuiltin = inspect.isbuiltin(operation)
    # Iterate over some picked values
    for mv_val in mvector_gen(layout):
        if isbuiltin:
            assert operation(layout.scalar, mv_val) == operation(1, mv_val)
        assert operation(mv_val, layout.scalar) == operation(mv_val, 1)
    # Unsupported operand type (right and left side)
    with pytest.raises(TypeError):
        _ = operation(layout.scalar, None)
    if isbuiltin:
        with pytest.raises(TypeError):
            _ = operation(None, layout.scalar)

def test_astype(dtype, exp_dtype):
    """Check conversion of internal `numpy` array `dtype`"""
    layout = micro_ga.Cl(3)
    # Check type of individual blades before and after type conversion
    for blade in layout.blades.values():
        assert isinstance(blade.value[0], np.integer), 'Internal blade numpy array must use int'
        blade = blade.astype(dtype)
        # Note: `dtype('O') == object`
        assert blade.value.dtype == dtype, 'Internal numpy array must use requested dtype'
        assert blade.subtype == exp_dtype, 'Reported subtype must match'
    # Check type of individual values from the scalar-blade
    scalar = layout.scalar.astype(dtype)
    for v in scalar.value:
        assert isinstance(v, exp_dtype), 'Individual values must be of requested type'

def test_operation_dtype(operation, dtype, exp_dtype):
    """Check the internal `numpy` array `dtype` of operation result"""
    layout = micro_ga.Cl(3)
    mv = operation(layout.mvector(12345).astype(dtype), layout.scalar)
    exp_dt = np.result_type(dtype)
    assert mv.value.dtype is exp_dt, 'Result dtype must match requested type'
    # Check type of individual values
    for v in mv.value:
        assert isinstance(v, exp_dtype), 'Individual values of result must be of requested type'

def test_unbounded_int():
    """Test python unbounded `int` operation"""
    layout = micro_ga.Cl(2)
    # When converted to `object`, `numpy` falls-back to original python unbounded operation
    scalar = layout.scalar.astype(object)
    mv = scalar + (1<<100)
    assert (mv.value[0] - (1<<100)) == 1

    # Default type promotes to `numpy.int64` (64-bit only)
    layout = micro_ga.Cl(2)
    with pytest.raises(OverflowError):
        mv = layout.scalar + (1<<100)
    mv = layout.scalar + (1<<40)

def test_round(dtype):
    """Test `round` operation"""
    layout = micro_ga.Cl(2)
    if dtype is object:
        exp_type = int          # Default `micro_ga` type
    else:
        exp_type = dtype
    # Pick a convenient number, which after rounding has finite binary representation
    val = exp_type(1.2456) + layout.I
    # Workaround: Add `0.` to suppress `sympy` precision difference checks
    val = round(val, 2) + exp_type(0.)
    assert val == exp_type(1.25) + layout.I

@pytest.mark.parametrize('operation', [operator.add, operator.mul])
def test_associativity(layout, operation, mvector_gen):
    """Test operation associativity: (a * b) * c == a * (b * c)"""
    for mv_val_1 in mvector_gen(layout):
        for mv_val_2 in mvector_gen(layout):
            mv_val_12 = operation(mv_val_1, mv_val_2)
            for mv_val_3 in mvector_gen(layout):
                mv_val_23 = operation(mv_val_2, mv_val_3)
                assert operation(mv_val_12, mv_val_3) == operation(mv_val_1, mv_val_23)

@pytest.mark.parametrize('operation', [operator.mul])
def test_distributivity(layout, operation, mvector_gen):
    """Test operation distributivity: a * (b + c) == a * b + a * c"""
    for mv_val_1 in mvector_gen(layout):
        for mv_val_2 in mvector_gen(layout):
            for mv_val_3 in mvector_gen(layout):
                # Test distributivity from the left
                mv_res_1 = operation(mv_val_1, mv_val_2 + mv_val_3)
                mv_res_2 = operation(mv_val_1, mv_val_2) + operation(mv_val_1, mv_val_3)
                assert mv_res_1 == mv_res_2, 'Distributivity from the left failed'
                # Test distributivity from the right
                mv_res_1 = operation(mv_val_1 + mv_val_2, mv_val_3)
                mv_res_2 = operation(mv_val_1, mv_val_3) + operation(mv_val_2, mv_val_3)
                assert mv_res_1 == mv_res_2, 'Distributivity from the right failed'
