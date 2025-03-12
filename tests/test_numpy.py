"""Test `numpy` array integration"""
import pytest
import numpy as np
import micro_ga
from . import pos_sig, layout, dtype, exp_dtype # pylint: disable=W0611
# pylint: disable=W0621


@pytest.fixture
def neg_sig():
    """Skip tests with basis-vectors of negative signature"""
    return 0

@pytest.fixture
def zero_sig():
    """Skip tests with basis-vectors of zero signature"""
    return 0

def test_ndarray(layout):
    """Test integration with `numpy.ndarray`"""
    res = np.arange(10) + layout.scalar
    np.testing.assert_equal(res, np.arange(10) + 1, 'Compare ndarray of MVector-s vs. int-s failed')
    # Check types of all elements and multi-vector subtype-s
    for item in res.flat:
        assert isinstance(item, micro_ga.MVector), f'Unexpected ndarray element type: {type(item)}'
        assert issubclass(item.subtype, np.integer), f'Unexpected MVector.subtype: {item.subtype}'

def test_from_to_ndarray(layout, dtype, exp_dtype):
    """Test `Cl.from_ndarray` functionality"""
    vals = np.arange(10 * layout.gaDims, dtype=dtype).reshape(-1, layout.gaDims)
    # Convert all coefficients to requested type (for Fraction and Decimal)
    if vals.dtype == object and (dtype is not object):
        vals = np.vectorize(dtype, otypes=[micro_ga.MVector])(vals)
    # Create array of multi-vectors, check each subtype
    res = layout.from_ndarray(vals)
    for item in res.flat:
        assert item.subtype == exp_dtype, f'Unexpected MVector.subtype: {item.subtype}'
    # Convert back to coefficients, check array and element types
    vals_back = layout.to_ndarray(res)
    np.testing.assert_array_equal(vals_back, vals, 'To-array conversion failed')
    assert vals_back.dtype == vals.dtype, 'To-array conversion dtype mismatch'
    assert type(vals_back.flat[0]) is type(vals.flat[0]), 'To-array element type mismatch'

    # Convert a scalar array, i.e. empty shape (`to_ndarray` handles this separately)
    res = layout.from_ndarray(vals[0])
    vals_back = layout.to_ndarray(res)
    np.testing.assert_array_equal(vals_back, vals[0], 'Scalar array conversion failed')

    # Create array of multi-vectors using different axis
    res = layout.from_ndarray(vals.T, axis=0)
    with pytest.raises(ValueError):
        layout.from_ndarray(vals, axis=0)

def test_vectorize(dtype, exp_dtype):
    """Test `numpy.vectorize` on a multi-vector method"""
    layout = micro_ga.Cl(2)
    res = np.array(layout.scalar) + np.arange(10)
    # Convert underlying type of all multi-vectors
    res = np.vectorize(micro_ga.MVector.astype, otypes=[micro_ga.MVector])(res, dtype=dtype)
    for item in res.flat:
        assert issubclass(item.subtype, exp_dtype), f'Unexpected MVector.subtype: {item.subtype}'
