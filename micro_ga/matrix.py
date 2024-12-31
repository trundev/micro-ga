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
    def to_matrix(self, mvector: MVector, **kw_args) -> npt.NDArray:
        """Convert multi-vector to the equivalent square matrix"""
        return self.to_matrix_ndarray(np.asarray(mvector), **kw_args)

    def to_matrix_ndarray(self, mvector_arr: npt.NDArray[np.object_], *, col_order: bool=False
                          ) -> npt.NDArray:
        """Convert `numpy.ndarray` of multi-vectors to array of equivalent square matrices"""
        #HACK: Check first array element only
        if self != mvector_arr.item(0).layout:
            raise ValueError('Multi-vector of incompatible layout')
        data = self.to_ndarray(mvector_arr)
        # Apply signatures, treat data as left (col_order=True) or right argument, see `Cl.do_mul()`
        data = np.expand_dims(data, axis=-1 if col_order else -2)
        data = data * self._mult_table
        # Reorder data to matrix-form positions
        return np.take_along_axis(*np.broadcast_arrays(data, self._mult_table_res_idx),
                                  axis=-2 if col_order else -1)

    def from_matrix(self, mvect_mtx: npt.NDArray, **kw_args) -> MVector:
        """Convert square matrix to a multi-vector (must be multi-vector equivalent matrix)"""
        if mvect_mtx.shape == mvect_mtx.shape[:1] * 2:
            return self.from_matrix_ndarray(mvect_mtx, **kw_args).item(0)
        raise ValueError('Input must be single square matrix')

    def from_matrix_ndarray(self, mvect_mtx: npt.NDArray, *, col_order: bool=False,
                            draft: bool=False) -> npt.NDArray[np.object_]:
        """Convert `numpy.ndarray` of square matrices to array of multi-vectors"""
        if draft:
            # The first row (col_order=False) of each matrix matches the multi-vector, just take it
            mvect = self.from_ndarray(mvect_mtx[..., *(np.s_[:, 0] if col_order else np.s_[0, :])],
                                      axis=-1)
        else:
            # Reorder data back from matrix-form positions
            data = np.empty_like(mvect_mtx)
            np.put_along_axis(data, *np.broadcast_arrays(self._mult_table_res_idx, mvect_mtx),
                              axis=-2 if col_order else -1)
            if not np.allclose(data[..., self._mult_table == 0], 0):
                raise ValueError('Matrix do not match degenerate metric')
            # Reapply signatures to get initial values back
            data = data * self._mult_table
            # Take averaged coefficients among non-zero signatures (avoid int to float conversion)
            data = np.mean(data, axis=-1 if col_order else -2, where=self._mult_table != 0)
            mvect = self.from_ndarray(data.astype(mvect_mtx.dtype))
        # Check if the result is consistent (type-conversion is  to support Fraction and Decimal)
        if np.allclose(self.to_matrix_ndarray(mvect, col_order=col_order).astype(complex),
                       mvect_mtx.astype(complex)):
            return mvect
        raise ValueError('Matrix is not multi-vector equivalent')
