"""Matrix representation of a multi-vector in `ganja.js` format

See https://github.com/enkimute/ganja.js/blob/master/ganja.js:
- Algebra.Element.describe().matrix
"""
import numpy as np
from . import layout, matrix

class Cl(matrix.Cl):
    """Clifford algebra generator with `ganja.js` matrix-form conversion support

    This version uses the exact rules from the latest version (as of 2025-2-4),
    which seems to have issue with signatures of 2 or more "zero" basis-vectors,
    like Cl(2, 0, 2) or Cl(1, 1, 2) - e.g. multiplication equivalence test fails.
    """
    _ganja_sign_adjust: layout.NDMultTableType

    def __init__(self, *arg, **kwarg):
        super().__init__(*arg, **kwarg)
        # `ganja` uses column based order
        self.set_conversion_type(True)

    def set_conversion_type(self, col_order: bool) -> None:
        """Select multi-vector to/from matrix conversion flavor"""
        super().set_conversion_type(col_order)

        # Mimic results from https://github.com/enkimute/ganja.js/blob/6e97cb4/ganja.js#L751
        metric_x = np.diag(super()._mtx_table)[super()._mtx_table_res_idx]
        if 0 in self.sig[:1]:   # Bases include `e0`
            # This acts as `...basis.indexOf(basis[x].replace("0", ""))`
            inv_sort = np.arange(self._blade_basis_masks.size)
            inv_sort[self._blade_basis_masks] = inv_sort
            basis_xd = inv_sort[self._blade_basis_masks & ~np.int_(1)]
            metric_xd = np.diag(super()._mtx_table)[basis_xd[super()._mtx_table_res_idx]]
        else:
            metric_xd = metric_x
        grades_x = self.gradeList[super()._mtx_table_res_idx]
        odd_grades = (-1)**grades_x.astype(int)
        negate_mask = (metric_x == -1) | (
                        (metric_x == 0)
                        & (grades_x > 1)
                        & ((metric_xd == 0) | (odd_grades == metric_xd)))
        # Convert to a sign-adjustment table
        sign_adjust = np.where(negate_mask, -1, 1).astype(super()._mtx_table.dtype)
        # Create sign-adjustment table for original multiplication table
        np.put_along_axis(sign_adjust, super()._mtx_table_res_idx, sign_adjust,
                          axis=self._res_idx_axis)
        self._ganja_sign_adjust = sign_adjust

    @property
    def _mtx_table(self) -> layout.NDMultTableType:
        """Multiplication table to be used by to/from matrix conversion"""
        return super()._mtx_table.T * self._ganja_sign_adjust
