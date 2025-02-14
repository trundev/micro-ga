"""Test multi-vector to/from equivalent matrix conversion"""
import math
import numpy as np
import numpy.typing as npt
import pytest
import numpy_determinant as npdet
import micro_ga.matrix
from . import rng, pos_sig, neg_sig, zero_sig, \
        rng_mvector, mvector_gen, mvector_2_gen      # pylint: disable=W0611
# pylint: disable=W0621


@pytest.fixture
def layout(pos_sig, neg_sig, zero_sig):
    """Geometric algebra with matrix-from conversion support fixture"""
    return micro_ga.matrix.Cl(pos_sig, neg_sig, zero_sig)

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
    # Ensure from vectors with degenerate signatures (last column in row ordered matrix)
    if 0 in layout.sig:
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
            else:   # Too large values may cause float underflow
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

def nz_grades(mv_val: micro_ga.MVector) -> npt.NDArray[np.bool]:
    """Boolean for each grade, indicating if it contains any non-zero coefficients"""
    layout = mv_val.layout
    res = np.zeros(dtype=bool, shape=(layout.gaDims, layout.dims + 1))
    np.put_along_axis(res, layout.gradeList[:, np.newaxis], mv_val.value[:, np.newaxis], axis=-1)
    return res.any(0)

def nested_isqrt(val: int) -> list[int]:
    """List of all successive "exact" integer square roots (every entry is $v^{-2^i}$)"""
    sqrts = []
    while val:
        res = math.isqrt(val)
        if res * res != val:
            return sqrts
        sqrts.append(res)
        val = res
    return sqrts

@pytest.mark.slow('Exact matrix determinant calculation')
def test_determinant_sqrts(layout, mvector_gen):
    """Test if multi-vector matrix determinant is some power of an integer

    Conjecture:
    Determinant of matrix from a multi-vector is its magnitude squared `dims` times, i.e.
    raised to the power of `2^{dims}`. The magnitude of multi-vector with positive integer
    coefficients, when squared a specific number of times, results in an integer, like:
    - scalar/pseudo-scalars - `magnitude` is integer
    - vectors/pseudo-vectors (also all `versors`) - `magnitude^2` is integer
    - others - TODO: stick with `magnitude^4` is integer

    Note:
    This test requires exact determinant value of big matrices, which is a huge value,
    from which must find the exact square root (repeatedly). Thus, test is limited to
    python unbounded integers only, no `np.linalg.det()`.
    """
    if layout.gaDims >= 32:
        pytest.skip('Exact matrix determinant calculation may take too much memory')
    for order in False, True:
        layout.set_conversion_type(col_order=order)
        for mv_val in mvector_gen(layout):
            # Convert multi-vector to matrix, invert and check
            mv_mtx = layout.to_matrix(mv_val)
            # Exact integer determinant using unbounded python int-s, no float underflow
            # Avoid `np.linalg.det()` as it lacks precision for large values
            try:    # `numpy-determinant` is memory irresponsible for big matrices
                int_det = npdet.det(mv_mtx.astype(object))
            except MemoryError as ex:
                pytest.xfail(str(ex))
            if int_det == 0:    # Ignore singular matrices
                continue
            # Simplification: always expect magnitude squared twice to be an integer
            num_sqrts = layout.dims - 2
            mask = nz_grades(mv_val)
            # Increase expectations
            num_sqrts += not mask[::2].any()        # Missing even grades
            num_sqrts += 2 * (not mask[1:].any())   # Scalar-only
            sqrts = nested_isqrt(int_det)
            assert len(sqrts) >= num_sqrts, \
                   f'Determinant of ({mv_val}) is not a {2**num_sqrts}-power of integer'
