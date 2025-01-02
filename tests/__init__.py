"""Generic `pytest` fixtures"""
import operator
import fractions
import decimal
import enum
from typing import Iterator
import numpy as np
import numpy.typing as npt
import pytest
import micro_ga


# pylint: disable=W0621
@pytest.fixture
def rng():
    """Random number generator fixture"""
    default_test_seed = 1  # the default seed to start pseudo-random tests
    return np.random.default_rng(default_test_seed)

#
# Layout related fixtures
#
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
def layout(pos_sig, neg_sig, zero_sig):
    """Geometric algebra object fixture"""
    return micro_ga.Cl(pos_sig, neg_sig, zero_sig)

#
# Arithmetic operation / `dtype` related fixtures
#
@pytest.fixture(params=[ np.int32, np.float64, np.complex64, object,
                         fractions.Fraction, decimal.Decimal])
def dtype(request):
    """Multi-vector underlying data-type fixture"""
    return request.param

@pytest.fixture(params=[
        operator.add,
        operator.sub,
        operator.mul,
        #operator.xor,   # outer product
        #operator.or_,   # inner product
    ])
def operation(request):
    """Arithmetic operation fixture"""
    return request.param

#
# Random multi-vector generators to test various operations
#
class MVType(enum.Enum):
    """Multi-vector types"""
    GRADE = enum.auto()     # Multi-vector of single grade
    VERSOR = enum.auto()    # A `versor` multi-vector (product of pure-vectors)
    GEN = enum.auto()       # Generic multi-vector (random coefficients)

def rng_mvector(rng, layout: micro_ga.Cl, blade_mask: npt.NDArray[np.bool] | bool):
    """Multi-vector of random coefficients in given blades only"""
    # Take all blades as `ndarray`
    blade_vals = np.fromiter((blade for blade in layout.blades.values()),
                             dtype=micro_ga.MVector)
    blade_vals = blade_vals[blade_mask]
    # Use prime numbers as multi-vector coefficients
    prime_nums = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                  31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    return (blade_vals * rng.choice(prime_nums, blade_vals.size)).sum()

@pytest.fixture(params=list(MVType), ids=list(t.name for t in MVType))
def mvector_gen(request, rng):
    """Generator to return multi-vectors of specific type / layout"""
    mv_type = request.param
    def iterator(layout: micro_ga.Cl) -> Iterator[micro_ga.MVector]:
        match mv_type:
            case MVType.GRADE:
                # Yield multi-vectors of each grade
                grade_list = layout.gradeList
                for grade in range(layout.dims + 1):
                    yield rng_mvector(rng, layout, grade_list == grade)
            case MVType.VERSOR:
                blade_mask = layout.gradeList == 1
                # Multiply some of random pure-vectors
                # (max grade of the result increases from vector to pseudo-scalar)
                for num in range(1, layout.dims + 1):
                    res = layout.scalar
                    for _ in range(num):
                        res = res * rng_mvector(rng, layout, blade_mask)
                    yield res
            case MVType.GEN:
                # Single generic multi-vector of random coefficients
                yield rng_mvector(rng, layout, True)
            case _:
                assert False, 'Unsupported multi-vector type'
    return iterator

@pytest.fixture
def mvector_2_gen(mvector_gen):
    """Generator to return pairs of multi-vectors of specific type / layout"""
    def iterator(layout: micro_ga.Cl) -> Iterator[tuple[micro_ga.MVector, micro_ga.MVector]]:
        for l_vals in mvector_gen(layout):
            for r_vals in mvector_gen(layout):
                yield l_vals, r_vals
    return iterator
