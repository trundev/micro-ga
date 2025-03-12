"""Test algebra/layout creation"""
import math
from typing import Any
import pytest
import numpy as np
import micro_ga
from . import pos_sig, neg_sig, zero_sig, layout, dtype, exp_dtype  # pylint: disable=W0611
# pylint: disable=W0621


def test_dimensions(pos_sig, neg_sig, zero_sig):
    """Test signature vs. dimensions, blade number"""
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig)
    assert layout.sig.size == pos_sig + neg_sig + zero_sig
    assert layout.dims == layout.sig.size
    assert layout.gaDims == 1<<layout.dims
    assert len(layout.blades) == layout.gaDims
    # Combine expected grades for all blades
    exp_grades = []
    for g in range(layout.dims + 1):
        exp_grades.extend([g]*math.comb(layout.dims, g))
    np.testing.assert_equal(layout.gradeList, exp_grades, 'Unexpected gradeList content')

    # Construct layer from incorrect parameters
    with pytest.raises(ValueError):
        micro_ga.Cl()
    with pytest.raises(ValueError):
        micro_ga.Cl(pos_sig, neg_sig, zero_sig, sig=[1]*pos_sig)
    # Construct multi-vector from wrong number of coefficients
    with pytest.raises(ValueError):
        layout.mvector([1, 2, 3])

def test_null_sig():
    """Test scalar-only algebra"""
    test_dimensions(0, 0, 0)
    layout = micro_ga.Cl(0)
    assert layout.scalar is layout.I
    assert layout.blades == {'': layout.scalar}

def test_blades(layout):
    """Test blade names"""
    _ = layout.scalar
    _ = layout.I
    for k, v in layout.blades.items():
        mv = getattr(layout, k) if k else layout.scalar
        assert mv is v, 'The layout attribute and blade value must be identical'
        assert sum(v.value) == 1 and np.count_nonzero(v.value) == 1, \
               'Blade must have a single value set to 1'

def test_comparison(pos_sig, neg_sig):
    """Test algebra and multi-vector comparison operators"""
    layout = micro_ga.Cl(pos_sig, neg_sig)
    assert layout == micro_ga.Cl(pos_sig, neg_sig)
    assert layout != 'BAD'
    assert layout.scalar != 'BAD'
    # Scalar-blade with multi-vector from a scalar-value
    assert layout.scalar == layout.mvector(1)
    # Different layout with different dimensions
    layout2 = micro_ga.Cl(neg_sig, pos_sig+1)
    assert layout != layout2
    assert layout.scalar != layout2.scalar
    # Layout created from a signature
    assert layout == micro_ga.Cl(sig=[1]*pos_sig + [-1]*neg_sig)

def test_repr(dtype, exp_dtype):
    """Test `repr()` and `str()` results"""
    layout = micro_ga.Cl(3)
    # Algebra representation
    assert str(layout) == f'Cl(sig={[1]*layout.dims})'
    if dtype is object:
        exp_type = int          # Default `micro_ga` type
    else:
        exp_type = dtype
    py_type = exp_type
    if issubclass(py_type, np.number):
        # String representation works on python type, but not on `numpy` type
        py_type: Any = type(np.zeros(1, dtype=dtype).item(0))
    # Basic multi-vector representations
    scalar = layout.scalar.astype(dtype)
    pscalar = layout.I.astype(dtype)
    assert str(scalar) == f'{exp_type(1)}'
    assert str(scalar - scalar) == '0'
    assert str(-scalar + pscalar) == f'{py_type(-1)} + {py_type(1)}*e123'
    assert repr(-scalar + pscalar) == \
           f'MVector({py_type(-1)!r} + {py_type(1)!r}*e123, subtype={exp_dtype.__name__})'
