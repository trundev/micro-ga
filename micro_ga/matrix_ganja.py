"""Matrix representation of a multi-vector in `ganja.js` format

See https://github.com/enkimute/ganja.js/blob/master/ganja.js:
- Algebra.Element.describe().matrix
"""
import numpy as np
import numpy.typing as npt
from . import matrix

class Cl(matrix.Cl):
    """Clifford algebra generator with `ganja.js` matrix-form conversion support

    This version uses the exact rules from the latest version (as of 2025-2-4),
    which seems to have issue with signatures of 2 or more "zero" basis-vectors,
    like Cl(2, 0, 1) or Cl(1, 1, 2) - e.g. multiplication equivalence test fails.
    """
    _ganja_sign_adjust: npt.NDArray[np.int_]

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)

        # Mimic results from https://github.com/enkimute/ganja.js/blob/6e97cb4/ganja.js#L751
        metric_x = np.diag(self._mult_table)[self._mult_table_res_idx]
        if 0 in self.sig[:1]:   # Bases include `e0`
            # This acts as `...basis.indexOf(basis[x].replace("0", ""))`
            inv_sort = np.arange(self._blade_basis_masks.size)
            inv_sort[self._blade_basis_masks] = inv_sort
            basis_xd = inv_sort[self._blade_basis_masks & ~np.int_(1)]
            metric_xd = np.diag(self._mult_table)[basis_xd[self._mult_table_res_idx]]
        else:
            metric_xd = metric_x
        grades_x = self.gradeList[self._mult_table_res_idx]
        odd_grades = (-1)**grades_x.astype(int)
        negate_mask = (metric_x == -1) | (
                        (metric_x == 0)
                        & (grades_x > 1)
                        & ((metric_xd == 0) | (odd_grades == metric_xd)))
        # Convert to a sign-adjustment table
        self._ganja_sign_adjust = np.where(negate_mask, -1, 1).astype(self._mult_table.dtype)

    def to_matrix_ndarray(self, mvector_arr: npt.NDArray[np.object_], *, col_order: bool=True
                          ) -> npt.NDArray:
        """Convert `numpy.ndarray` of multi-vectors to array of equivalent square matrices"""
        #HACK: Check first array element only
        if self != mvector_arr.item(0).layout:
            raise ValueError('Multi-vector of incompatible layout')
        data = np.expand_dims(self.to_ndarray(mvector_arr), axis=-1 if col_order else -2)
        data = data * self._mult_table.T
        data = np.take_along_axis(*np.broadcast_arrays(data, self._mult_table_res_idx),
                                  axis=-2 if col_order else -1)
        return data * self._ganja_sign_adjust

    def from_matrix_ndarray(self, mvect_mtx: npt.NDArray, *, col_order: bool=True,
                            draft: bool=False) -> npt.NDArray[np.object_]:
        """Convert `numpy.ndarray` of square matrices to array of multi-vectors"""
        if draft:
            # The first column (col_order=False) of each matrix contains the multi-vector
            # coefficients, but with adjust the signs
            mvect = mvect_mtx * self._ganja_sign_adjust
            mvect = mvect[..., *(np.s_[:, 0] if col_order else np.s_[0, :])]
            mvect = self.from_ndarray(mvect, axis=-1)
        else:
            # Reorder data back from matrix-form positions
            data = mvect_mtx * self._ganja_sign_adjust
            np.put_along_axis(data, *np.broadcast_arrays(self._mult_table_res_idx, data),
                              axis=-2 if col_order else -1)
            # Reapply signatures to get initial values back
            data = data * self._mult_table.T
            #pylint: disable=duplicate-code #HACK: The same as in 'matrix.Cl'
            # Take averaged coefficients among non-zero signatures (avoid int to float conversion)
            data = np.mean(data, axis=-1 if col_order else -2, where=self._mult_table != 0)
            mvect = self.from_ndarray(data.astype(mvect_mtx.dtype))
        # Check if the result is consistent (type-conversion is  to support Fraction and Decimal)
        if np.allclose(self.to_matrix_ndarray(mvect, col_order=col_order).astype(complex),
                       mvect_mtx.astype(complex)):
            return mvect
        raise ValueError('Matrix is not multi-vector equivalent')
