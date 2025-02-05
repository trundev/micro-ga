"""Matrix representation of a multi-vector in `ganja.js` format

See https://github.com/enkimute/ganja.js/blob/master/ganja.js:
- Algebra.Element.describe().matrix
"""
import numpy as np
import numpy.typing as npt
from . import matrix, layout

class Cl(matrix.Cl):
    """Clifford algebra generator with `ganja.js` matrix-form conversion support

    This version uses matrix-form rules, that give the same result as `ganja.js`,
    but for signatures of zero or one "zero" basis-vectors only. For others like
    Cl(2, 0, 1) or Cl(1, 1, 2), the result is multiplication equivalent unlike
    the original.
    """
    _nz_mult_table: layout.NDMultTableType

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)

        # Create signatures for a non-degenerate algebra (if necessary)
        if 0 in self.sig:
            self._nz_mult_table = self._build_mult_table(np.where(self.sig, self.sig, 1))
        else:
            self._nz_mult_table = self._mult_table

    def to_matrix_ndarray(self, mvector_arr: npt.NDArray[np.object_], *, col_order: bool=False
                          ) -> npt.NDArray:
        """Convert `numpy.ndarray` of multi-vectors to array of equivalent square matrices"""
        #HACK: Check first array element only
        if self != mvector_arr.item(0).layout:
            raise ValueError('Multi-vector of incompatible layout')
        data = np.expand_dims(self.to_ndarray(mvector_arr), axis=-1 if col_order else -2)
        data = data * self._nz_mult_table
        data = np.take_along_axis(*np.broadcast_arrays(data, self._mult_table_res_idx),
                                  axis=-2 if col_order else -1)
        # Apply zero-vectors using original signatures, but rolled along other axis
        zero_mask = np.take_along_axis(self._mult_table == 0, self._mult_table_res_idx,
                                       axis=-1 if col_order else -2)
        data[..., zero_mask] = 0
        return data

    def from_matrix_ndarray(self, mvect_mtx: npt.NDArray, *, col_order: bool=False,
                            draft: bool=False) -> npt.NDArray[np.object_]:
        """Convert `numpy.ndarray` of square matrices to array of multi-vectors"""
        if draft:
            # The first column (col_order=False) of each matrix contains the multi-vector
            # coefficients, but with applied signatures
            mvect = self.from_ndarray(mvect_mtx[..., *(np.s_[0, :] if col_order else np.s_[:, 0])]
                                      * np.diag(self._nz_mult_table), axis=-1)
        else:
            # Check expected zeros for zero-versions
            zero_mask = np.take_along_axis(self._mult_table == 0, self._mult_table_res_idx,
                                           axis=-1 if col_order else -2)
            if not np.allclose(mvect_mtx[..., zero_mask], 0):
                raise ValueError('Matrix do not match degenerate signature')
            # Reorder data back from matrix-form positions
            data = np.empty_like(mvect_mtx)
            np.put_along_axis(data, *np.broadcast_arrays(self._mult_table_res_idx, mvect_mtx),
                              axis=-2 if col_order else -1)
            # Reapply signatures to get initial values back
            data = data * self._nz_mult_table
            #pylint: disable=duplicate-code #HACK: The same as in 'matrix.Cl'
            # Take averaged coefficients among non-zero signatures (avoid int to float conversion)
            data = np.mean(data, axis=-1 if col_order else -2, where=~zero_mask)
            mvect = self.from_ndarray(data.astype(mvect_mtx.dtype))
        # Check if the result is consistent (type-conversion is  to support Fraction and Decimal)
        if np.allclose(self.to_matrix_ndarray(mvect, col_order=col_order).astype(complex),
                       mvect_mtx.astype(complex)):
            return mvect
        raise ValueError('Matrix is not multi-vector equivalent')
