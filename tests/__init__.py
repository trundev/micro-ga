"""Generic `pytest` fixtures"""
import operator
import fractions
import decimal
import numpy as np
import pytest
import micro_ga


# pylint: disable=W0621
@pytest.fixture
def rng():
    """Random number generator fixture"""
    default_test_seed = 1  # the default seed to start pseudo-random tests
    return np.random.default_rng(default_test_seed)

@pytest.fixture(params=[2, 3])
def pos_sig(request):
    """Number of basis-vectors with positive signature fixture"""
    return request.param

@pytest.fixture(params=[0, 1])
def neg_sig(request):
    """Number of basis-vectors with negative signature fixture"""
    return request.param

@pytest.fixture(params=[0, 1])
def zero_sig(request):
    """Number of basis-vectors with zero signature fixture"""
    return request.param

@pytest.fixture
def layout(pos_sig, neg_sig, zero_sig):     # pylint: disable=W0621
    """Geometric algebra object fixture"""
    return micro_ga.Cl(pos_sig, neg_sig, zero_sig)

@pytest.fixture(params=[ np.int32, np.float64, np.complex64, object,
                         fractions.Fraction, decimal.Decimal])
def dtype(request):
    """Multi-vector underlying data-type fixture"""
    return request.param

@pytest.fixture(params=[
        operator.add,
        operator.sub,
        operator.mul,
        #operator.xor,  # outer product
        #operator.or_,  # inner product
    ])
def operation(request):
    """Arithmetic operation fixture"""
    return request.param
