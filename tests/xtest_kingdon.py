"""Test using `kingdon` module as a reference

See https://github.com/tBuLi/kingdon
"""
import pytest
import numpy as np
import kingdon
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

def test_blades(pos_sig, neg_sig, zero_sig):
    """Check if our layout has the same blades in the same order

    Other tests expect the that the blades in our and `kingdon` layout are in the
    same ordered. Thus, multi-vectors are the same when `value` and `values()` match.
    """
    # `kingdon` uses zero-based indices for degenerate metric
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig, first_index=0 if zero_sig else 1)
    # Create `kingdon` algebra of same signature
    kn_layout = kingdon.Algebra(pos_sig, neg_sig, zero_sig)
    ref_basis = np.asarray(tuple(kn_layout.blades.blades.keys()))
    # `kingdon` blades use 'e' as the scalar
    ref_basis[ref_basis == 'e'] = ''
    np.testing.assert_equal(tuple(layout.blades.keys()), ref_basis, 'Blades are different')

def test_operations(layout, operation, mvector_2_gen):
    """Check our results vs. `kingdon` ones"""
    # Create `kingdon` algebra of same signature, convert operation
    kn_layout = kingdon.Algebra(signature=layout.sig)
    ref_op = getattr(kingdon.MultiVector, operation.__name__, operation)

    # Iterate over some picked value combinations
    for our_l_val, our_r_val in mvector_2_gen(layout):
        # Convert values to `kingdon` ones
        ref_l_val = kn_layout.multivector(our_l_val.value)
        ref_r_val = kn_layout.multivector(our_r_val.value)
        # Test results from `kingdon` and `micro-ga`
        ref_res = ref_op(ref_l_val, ref_r_val)
        our_res = operation(our_l_val, our_r_val)
        np.testing.assert_equal(our_res.value, ref_res.values())

def test_matrix_form(pos_sig, neg_sig, zero_sig, mvector_gen):
    """Check multi-vector matrix-form conversion"""
    layout = micro_ga.matrix.Cl(pos_sig, neg_sig, zero_sig)
    # `kingdon` uses column based order
    layout.set_conversion_type(col_order=True)
    # Create `kingdon` algebra of same signature
    kn_layout = kingdon.Algebra(pos_sig, neg_sig, zero_sig)

    # Iterate over some picked values
    for our_val in mvector_gen(layout):
        # Convert value to `kingdon` ones
        ref_val = kn_layout.multivector(our_val.value)
        ref_mtx = ref_val.asmatrix()
        #HACK: Ensure the `kingdon` matrix-form is correct
        if not np.array_equal((ref_val * ref_val).asmatrix(), ref_mtx @ ref_mtx):
            pytest.xfail('Reference matrix is NOT multiplication equivalent')
        our_mtx = layout.to_matrix(our_val)
        np.testing.assert_equal(our_mtx, ref_mtx)
