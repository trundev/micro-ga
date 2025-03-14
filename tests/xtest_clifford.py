"""Test using `clifford` module as a reference"""
import pytest
import numpy as np
import clifford
import micro_ga.matrix
from . import rng, neg_sig, zero_sig, layout, operation, \
        mvector_gen, mvector_2_gen  # pylint: disable=W0611
# pylint: disable=W0621


# Test with single positive signature only,
# but multiple `neg_sig` and `zero_sig`signatures
@pytest.fixture(params=[2])
def pos_sig(request):
    """Single test with basis-vectors of positive signature"""
    return request.param

def test_blades(layout):
    """Check if our layout has the same blades in the same order

    Other tests expect the that the blades in our and `clifford` layout are in the
    same ordered. Thus, multi-vectors are the same when `value` match.
    """
    # Create `clifford` algebra of same signature
    cl_layout = clifford.Cl(sig=layout.sig)[0]
    assert layout.blades.keys() == cl_layout.blades.keys(), 'Blades are different'
    assert tuple(layout.blades.keys()) == tuple(cl_layout.blades.keys()), \
           'Blade order is different'

def test_operations(layout, operation, mvector_2_gen):
    """Check our results vs. `clifford` ones"""
    # Create `clifford` algebra of same signature, convert operation
    cl_layout = clifford.Cl(sig=layout.sig)[0]
    ref_op = getattr(clifford.MultiVector, operation.__name__, operation)

    # Iterate over some picked value combinations
    for our_l_val, our_r_val in mvector_2_gen(layout):
        # Convert values to `clifford` ones
        ref_l_val = clifford.MultiVector(cl_layout, our_l_val.value)
        ref_r_val = clifford.MultiVector(cl_layout, our_r_val.value)
        # Test results from `clifford` and `micro-ga`
        ref_res = ref_op(ref_l_val, ref_r_val)
        our_res = operation(our_l_val, our_r_val)
        np.testing.assert_equal(our_res.value, ref_res.value)
        # Swap operands to test commutativity
        ref_res = ref_op(ref_r_val, ref_l_val)
        our_res = operation(our_r_val, our_l_val)
        np.testing.assert_equal(our_res.value, ref_res.value)

def test_matrix_form(pos_sig, neg_sig, zero_sig, mvector_gen):
    """Check multi-vector matrix-form conversion"""
    layout = micro_ga.matrix.Cl(pos_sig, neg_sig, zero_sig)
    # Create `clifford` algebra of same signature
    cl_layout = clifford.Cl(pos_sig, neg_sig, zero_sig)[0]

    # Iterate over some picked values
    for our_val in mvector_gen(layout):
        # Convert value to `clifford` ones
        ref_val = clifford.MultiVector(cl_layout, our_val.value)
        # Test "left multiplication", matches column-order form
        ref_left_mtx = cl_layout.get_left_gmt_matrix(ref_val)
        layout.set_conversion_type(col_order=True)
        our_mtx = layout.to_matrix(our_val)
        np.testing.assert_equal(our_mtx, ref_left_mtx)
        # Test "right multiplication", matches transposed row-order form
        ref_right_mtx = cl_layout.get_right_gmt_matrix(ref_val)
        layout.set_conversion_type(col_order=False)
        our_mtx = layout.to_matrix(our_val)
        np.testing.assert_equal(our_mtx, ref_right_mtx.T)
