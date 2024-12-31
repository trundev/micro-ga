"""Matrix representation of a multi-vector (matrix-form)

Inspired by this Mathoma video: https://www.youtube.com/watch?v=3Ki14CsP_9k
and this article: https://discourse.bivector.net/t/matrix-representation

Geometric algebra multi-vectors have corresponding matrices, with equivalent matrix
multiplication behavior, similar to Pauli matrices. Such matrix representation, even
less efficient, allows use of all matrix tricks, like inverse, power to real-number,
exponentiation, eigenvector, diagonalize, etc.

The matrix is square one, combined from the multi-vector coefficient, in row or column
based order. Here is an example for a row based order:
Each row is combined from multi-vector coefficients:
- The first row consist of unchanged multi-vector coefficients
- The second row consist of coefficients after multiplication by 'e1'
- etc.
- The last row consist of coefficients after multiplication by pseudo-scalar
Note:
* The multi-vector coefficients in each matrix-row are in different order. Some are
  negated or even set to zero in case of degenerate metrics.
* With degenerate metric, https://github.com/enkimute/ganja.js gives different matrices,
  with the same multiplication behavior (difference in the zero-element locations only).
  This version allows easier conversion from matrix-form to multi-vector.
"""
import numpy as np
import numpy.typing as npt
from . import layout
from .multivector import MVector

class Cl(layout.Cl):
    """Clifford algebra generator with matrix-form conversion support"""
    _expand_axis: int
    _res_idx_axis: int

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        # Default to row based order
        self.set_conversion_type(False)

    def set_conversion_type(self, col_order: bool) -> None:
        """Select multi-vector to/from matrix conversion flavor"""
        self._expand_axis = -1 if col_order else -2
        self._res_idx_axis = -2 if col_order else -1

    @property
    def _mtx_table(self) -> layout.NDMultTableType:
        """Multiplication table to be used by to/from matrix conversion"""
        return self._mult_table

    @property
    def _mtx_table_res_idx(self) -> layout.NDResultIdxType:
        """Multiplication result index table to be used by to/from matrix conversion"""
        return self._mult_table_res_idx

    def to_matrix(self, mvector: MVector, **kw_args) -> npt.NDArray:
        """Convert multi-vector to the equivalent square matrix"""
        return self.to_matrix_ndarray(np.asarray(mvector), **kw_args)

    def to_matrix_ndarray(self, mvector_arr: npt.NDArray[np.object_]) -> npt.NDArray:
        """Convert `numpy.ndarray` of multi-vectors to array of equivalent square matrices"""
        #HACK: Check first array element only
        if self != mvector_arr.item(0).layout:
            raise ValueError('Multi-vector of incompatible layout')
        data = self.to_ndarray(mvector_arr)
        # Apply signatures, treat data as left (col_order=True) or right argument, see `Cl.do_mul()`
        data = np.expand_dims(data, axis=self._expand_axis)
        data = data * self._mtx_table
        # Reorder data to matrix-form positions
        return np.take_along_axis(*np.broadcast_arrays(data, self._mtx_table_res_idx),
                                  axis=self._res_idx_axis)

    def from_matrix(self, mvect_mtx: npt.NDArray, **kw_args) -> MVector:
        """Convert square matrix to a multi-vector (must be multi-vector equivalent matrix)"""
        if mvect_mtx.shape == mvect_mtx.shape[:1] * 2:
            return self.from_matrix_ndarray(mvect_mtx, **kw_args).item(0)
        raise ValueError('Input must be single square matrix')

    def from_matrix_ndarray(self, mvect_mtx: npt.NDArray, *,
                            strict: bool=True) -> npt.NDArray[np.object_]:
        """Convert `numpy.ndarray` of square matrices to array of multi-vectors"""
        # Reorder data back from matrix-form positions
        data = np.empty_like(mvect_mtx)
        np.put_along_axis(data, *np.broadcast_arrays(self._mtx_table_res_idx, mvect_mtx),
                            axis=self._res_idx_axis)
        if not np.allclose(data[..., self._mtx_table == 0], 0):
            raise ValueError('Matrix do not match degenerate metric')
        # Reapply signatures to get initial values back
        data = data * self._mtx_table
        # Take averaged coefficients among non-zero signatures (avoid int to float conversion)
        data = np.mean(data, axis=self._expand_axis, where=self._mtx_table != 0)
        mvect = self.from_ndarray(data.astype(mvect_mtx.dtype))
        # Check if the result is consistent (type-conversion is to support Fraction and Decimal)
        if strict and not np.allclose(self.to_matrix_ndarray(mvect).astype(complex),
                                      mvect_mtx.astype(complex)):
            raise ValueError('Matrix is not multi-vector equivalent')
        return mvect

    def from_vector_matrix(self, mvect_vmtx: npt.NDArray, *, strict: bool=True,
                           row: int|None=None, column: int|None=None) -> npt.NDArray[np.object_]:
        """Convert `numpy.ndarray` of 1D vector-matrices to array of multi-vectors"""
        if (row is None) ^ (column is not None):
            raise ValueError('Either row or column must be value')
        # Prepare indices for this row- or column- matrix-vector
        if column is None:
            index = row
            res_idx = self._mtx_table_res_idx[row]
            mult_table = self._mtx_table
        else:
            index = column
            res_idx = self._mtx_table_res_idx[:, column]
            mult_table = self._mtx_table.T
        if (self._expand_axis == -2) ^ (column is None):
            # Vector orientation is orthogonal to the matrix order,
            # like row-vector from matrix with column based order
            index = np.arange(mult_table.shape[0])
            mult_table = mult_table.T
        signature = mult_table[index, res_idx]
        # Ensure coefficients at degenerate signatures are zero
        z_mask = signature == 0
        if (z_mask if strict else mvect_vmtx[z_mask] != 0).any():
            raise ValueError('Matrix do not match degenerate metric')
        data = (mvect_vmtx * signature)[res_idx]
        return self.from_ndarray(data, axis=-1)
