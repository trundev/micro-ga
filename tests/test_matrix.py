"""Test multi-vector to/from equivalent matrix conversion"""
import numpy as np
import pytest
import micro_ga.matrix
import micro_ga.matrix_ganja
from . import rng, pos_sig, neg_sig, zero_sig, \
        rng_mvector, mvector_gen, mvector_2_gen      # pylint: disable=W0611
# pylint: disable=W0621


@pytest.fixture(params=[micro_ga.matrix, micro_ga.matrix_ganja],
                ids=['matrix', 'matrix_ganja'])
def layout(pos_sig, neg_sig, zero_sig, request):
    """Geometric algebra with matrix-from conversion support fixture"""
    return request.param.Cl(pos_sig, neg_sig, zero_sig)

def test_to_from_matrix(layout, mvector_gen):
    """Test multi-vector to matrix conversion"""
    for order in False, True:
        layout.set_conversion_type(col_order=order)
        for mv_val in mvector_gen(layout):
            # Convert multi-vector to matrix and back
            mv_mtx = layout.to_matrix(mv_val)
            assert layout.from_matrix(mv_mtx) == mv_val, 'Backward conversion failed'

    # Ensure matrix conversion works on compatible multi-vectors only
    with pytest.raises(ValueError):
        layout.to_matrix(micro_ga.Cl(1).scalar)
    # Ensure non multi-vector equivalent matrices are detected
    inval_mtx = np.arange(layout.gaDims**2).reshape([layout.gaDims]*2)
    with pytest.raises(ValueError):
        layout.from_matrix(inval_mtx)
    with pytest.raises(ValueError):
        layout.from_matrix(inval_mtx[np.newaxis])

def test_from_vector_matrix(rng, layout):
    """Test multi-vector from single row/column matrix-vector conversion"""
    for order in False, True:
        layout.set_conversion_type(col_order=order)
        # Create multi-vector equivalent matrix, iterate over all of its rows/columns
        mv_val = rng_mvector(rng, layout, True)
        mv_mtx = layout.to_matrix(mv_val)
        for idx in range(mv_mtx.shape[0]):
            # Row conversion
            no_zeros = (mv_mtx[idx, :] != 0).all()
            mv_res = layout.from_vector_matrix(mv_mtx[idx, :], row=idx, strict=no_zeros)
            if no_zeros:
                assert mv_res == mv_val, f'Single-row conversion failed: {order=}, row={idx}'
            else:   # Vector with degenerate signature, expect partial result
                mtx_res = layout.to_matrix(mv_res)
                np.testing.assert_equal(mtx_res[idx, :], mv_mtx[idx, :],
                                        f'Degenerate-row conversion failed: {order=}, row={idx}')
            # Column conversion
            no_zeros = (mv_mtx[:, idx] != 0).all()
            mv_res = layout.from_vector_matrix(mv_mtx[:, idx], column=idx, strict=no_zeros)
            if no_zeros:
                assert mv_res == mv_val, f'Single-column conversion failed: {order=}, row={idx}'
            else:   # Vector with degenerate signature, expect partial result
                mtx_res = layout.to_matrix(mv_res)
                np.testing.assert_equal(mtx_res[:, idx], mv_mtx[:, idx], \
                                        f'Degenerate-column conversion failed: {order=}, row={idx}')

    # Ensure failure with ambiguous row/column parameters
    with pytest.raises(ValueError):
        layout.from_vector_matrix(np.ones(layout.gaDims), row=0, column=0)
    with pytest.raises(ValueError):
        layout.from_vector_matrix(np.ones(layout.gaDims))
    # Ensure failure from vectors with degenerate signatures (last column in column ordered matrix)
    if 0 in layout.sig:
        # Workaround: For `ganja` this is valid for row ordered matrices
        if isinstance(layout, micro_ga.matrix_ganja.Cl):
            layout.set_conversion_type(False)
        with pytest.raises(ValueError):
            layout.from_vector_matrix(np.zeros(layout.gaDims), column=-1, strict=True)

def test_ndarray_to_matrix(layout, mvector_gen):
    """Test `numpy.ndarray` of multi-vectors to matrix conversion"""
    # Combine a 2D array of multi-vectors
    mv_arr = np.stack(2 * [np.fromiter(mvector_gen(layout), micro_ga.MVector)])
    # Convert multi-vectors to matrices and back
    mtx_arr = layout.to_matrix_ndarray(mv_arr)
    np.testing.assert_equal(layout.from_matrix_ndarray(mtx_arr), mv_arr)

    # Ensure matrix conversion works on compatible multi-vectors only
    with pytest.raises(ValueError):
        layout.to_matrix_ndarray(np.asarray(micro_ga.Cl(1).scalar))
    # Ensure non multi-vector equivalent matrices are detected
    inval_mtx = np.arange(layout.gaDims**2).reshape([layout.gaDims]*2)
    with pytest.raises(ValueError):
        layout.from_matrix_ndarray(inval_mtx)

def test_mult_equivalence(layout, mvector_2_gen):
    """Test multi-vector vs matrix multiplication equivalence"""
    for order in False, True:
        layout.set_conversion_type(col_order=order)
        for l_val, r_val in mvector_2_gen(layout):
            ref_res = l_val * r_val
            # Convert multi-vectors to matrices and multiply
            l_mtx = layout.to_matrix(l_val)
            r_mtx = layout.to_matrix(r_val)
            mtx_res = l_mtx @ r_mtx
            # Compare to matrix from multiplication result
            mtx_ref_res = layout.to_matrix(ref_res)
            np.testing.assert_equal(mtx_res, mtx_ref_res, 'Multiplication equivalence failed')
            # Convert back to multi-vector and compare to multiplication result
            mv_res = layout.from_matrix(mtx_res)
            if np.abs(ref_res.value).max() * np.finfo(float).resolution < 1 / layout.gaDims:
                assert mv_res == ref_res, 'Multiplication equivalence failed'
            else:   # `Cl.from_matrix` from huge values may cause float underflow
                np.testing.assert_allclose(mv_res.value, ref_res.value)

def test_mult_inverse(layout, mvector_gen):
    """Test multi-vector matrix multiplicative inverse"""
    for mv_val in mvector_gen(layout):
        # Convert multi-vector to matrix, invert and check
        mv_mtx = layout.to_matrix(mv_val)
        if 0 in layout.sig and np.linalg.det(mv_mtx) == 0:
            pytest.xfail('Can not invert singular matrix (degenerate metric)')
        mv_inv = np.linalg.inv(mv_mtx)
        mv_inv = layout.from_matrix(mv_inv)
        assert round(mv_inv * mv_val, 10) == 1, 'Inverse failed'
        assert round(mv_val * mv_inv, 10) == 1, 'Inverse must be commutative'
