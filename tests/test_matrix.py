"""Test multi-vector to/from equivalent matrix conversion"""
import numpy as np
import pytest
import micro_ga.matrix
from . import rng, pos_sig, neg_sig, zero_sig, \
        mvector_gen, mvector_2_gen      # pylint: disable=W0611
# pylint: disable=W0621


@pytest.fixture
def matrix_layout(pos_sig, neg_sig, zero_sig):
    """Geometric algebra with matrix-from conversion fixture"""
    return micro_ga.matrix.Cl(pos_sig, neg_sig, zero_sig)

def test_to_from_matrix(matrix_layout, mvector_gen):
    """Test multi-vector to matrix conversion"""
    layout = matrix_layout
    for order in False, True:
        for mv_val in mvector_gen(layout):
            # Convert multi-vector to matrix and back
            mv_mtx = layout.to_matrix(mv_val, col_order=order)
            assert layout.from_matrix(mv_mtx, col_order=order) == mv_val, \
                    'Backward conversion failed'

    # Ensure matrix conversion works on compatible multi-vectors only
    with pytest.raises(ValueError):
        layout.to_matrix(micro_ga.Cl(1).scalar)
    # Ensure non multi-vector equivalent matrices are detected
    inval_mtx = np.arange(layout.gaDims**2).reshape([layout.gaDims]*2)
    with pytest.raises(ValueError):
        layout.from_matrix(inval_mtx)
    with pytest.raises(ValueError):
        layout.from_matrix(inval_mtx[np.newaxis])

def test_ndarray_to_matrix(matrix_layout, mvector_gen):
    """Test `numpy.ndarray` of multi-vectors to matrix conversion"""
    layout = matrix_layout
    # Combine a 2D array of multi-vectors
    mv_arr = np.stack(2 * [np.fromiter(mvector_gen(layout), micro_ga.MVector)])
    # Convert multi-vectors to matrices and back
    mtx_arr = layout.to_matrix_ndarray(mv_arr)
    np.testing.assert_equal(layout.from_matrix_ndarray(mtx_arr), mv_arr)
    np.testing.assert_equal(layout.from_matrix_ndarray(mtx_arr, draft=True), mv_arr)

    # Ensure matrix conversion works on compatible multi-vectors only
    with pytest.raises(ValueError):
        layout.to_matrix_ndarray(np.asarray(micro_ga.Cl(1).scalar))
    # Ensure non multi-vector equivalent matrices are detected
    inval_mtx = np.arange(layout.gaDims**2).reshape([layout.gaDims]*2)
    with pytest.raises(ValueError):
        layout.from_matrix_ndarray(inval_mtx)

def test_mult_equivalence(matrix_layout, mvector_2_gen):
    """Test multi-vector vs matrix multiplication equivalence"""
    layout = matrix_layout
    for order in False, True:
        for l_val, r_val in mvector_2_gen(layout):
            ref_res = l_val * r_val
            # Convert multi-vectors to matrices and multiply
            l_mtx = layout.to_matrix(l_val, col_order=order)
            r_mtx = layout.to_matrix(r_val, col_order=order)
            mtx_res = l_mtx @ r_mtx
            # Compare to matrix from multiplication result
            mtx_ref_res = layout.to_matrix(ref_res, col_order=order)
            np.testing.assert_equal(mtx_res, mtx_ref_res, 'Multiplication equivalence failed')
            # Convert back to multi-vector and compare to multiplication result
            mv_res = layout.from_matrix(mtx_res, col_order=order)
            if np.abs(ref_res.value).max() * np.finfo(float).resolution < 1 / layout.gaDims:
                assert mv_res == ref_res, 'Multiplication equivalence failed'
            else:   # Too large values may cause float underflow
                np.testing.assert_allclose(mv_res.value, ref_res.value)

def test_mult_inverse(matrix_layout, mvector_gen):
    """Test multi-vector matrix multiplicative inverse"""
    layout = matrix_layout
    for mv_val in mvector_gen(layout):
        # Convert multi-vector to matrix, invert and check
        mv_mtx = layout.to_matrix(mv_val)
        if 0 in layout.sig and np.linalg.det(mv_mtx) == 0:
            pytest.xfail('Can not invert singular matrix (degenerate metric)')
        mv_inv = np.linalg.inv(mv_mtx)
        mv_inv = layout.from_matrix(mv_inv)
        assert round(mv_inv * mv_val, 10) == 1, 'Inverse failed'
        assert round(mv_val * mv_inv, 10) == 1, 'Inverse must be commutative'
