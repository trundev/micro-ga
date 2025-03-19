"""Test extra tools"""
import operator
from typing import Iterator
import numpy as np
import pytest
import micro_ga
import micro_ga.tools as tools
from . import rng, neg_sig, zero_sig, layout, mvector_gen, mvector_2_gen   # pylint: disable=W0611
# pylint: disable=W0621


# Test with as many as possible positive signatures, including none
@pytest.fixture(params=range(4))
def pos_sig(request):
    """Test with various number of basis-vectors of positive signature"""
    return request.param

@pytest.fixture
def mvector_complex_2_gen(mvector_2_gen):
    """Generator to return pairs of multi-vectors with complex coefficients"""
    def iterator(layout: micro_ga.Cl) -> Iterator:
        # Generate separate values for real and imaginary parts
        for vals_real, vals_imag in zip(mvector_2_gen(layout), mvector_2_gen(layout)):
            vals = layout.to_ndarray(vals_real) + layout.to_ndarray(vals_imag)*1j
            yield layout.from_ndarray(vals)
    return iterator

@pytest.mark.parametrize('operation', [operator.add, operator.mul])
def test_sympify_equivalence(layout, operation, mvector_2_gen):
    """Check operation results using sympy engine"""
    # Iterate over some picked values
    for l_val, r_val in mvector_2_gen(layout):
        our_res = operation(l_val, r_val)
        # Run operation on multi-vectors converted to `sympy` expressions
        sp_l_val = tools.mvector_sympify(l_val)
        sp_r_val = tools.mvector_sympify(r_val)
        sp_res = operation(sp_l_val, sp_r_val)
        # Apply GA rules
        sp_res = tools.sympy_blade_rules(layout, sp_res)
        assert sp_res == tools.mvector_sympify(our_res), 'Sympy result does NOT match'

@pytest.mark.parametrize('operation', [operator.add, operator.mul])
def test_expand_complex_equivalence(layout, operation, mvector_complex_2_gen):
    """Compare expand_vector equivalence before and after some operation"""
    base_basis = 0  #HACK: non-scalars may not make sense
    # Iterate over some picked values
    for l_val, r_val in mvector_complex_2_gen(layout):
        # Run operation on original values, then expand
        res = operation(l_val, r_val)
        ex_res = tools.expand_complex(res, base_basis)
        np.testing.assert_equal(ex_res.value.imag, 0, 'Imaginary component in expanded result')
        # Expand original values, then run operation
        l_val_ex = tools.expand_complex(l_val, base_basis)
        r_val_ex = tools.expand_complex(r_val, base_basis)
        res_ex = operation(l_val_ex, r_val_ex)
        assert ex_res == res_ex, f'Equivalence failed for {operation}'
