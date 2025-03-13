"""Matrix representation of a multi-vector in `ganja.js` format

See https://github.com/enkimute/ganja.js/blob/master/ganja.js:
- Algebra.Element.describe().matrix
"""
import numpy as np
from . import layout, matrix

class Cl(matrix.Cl):
    """Clifford algebra generator with `ganja.js` matrix-form conversion support

    This version uses matrix-form rules, that give the same result as `ganja.js`,
    but for signatures of zero or one "zero" basis-vectors only. For others,
    like Cl(2, 0, 2) or Cl(1, 1, 2), the result is multiplication equivalent
    unlike the original.

    Note:
    `ganja` uses a mixed order, where non-degenerate signatures are in row based
    order, but zeros are in column based order. This is the default mode, selected
    by `matrix.Cl()`.
    """
    _mtx_mult_table: layout.NDMultTableType

    def set_conversion_type(self, col_order: bool) -> None:
        """Select multi-vector to/from matrix conversion flavor"""
        super().set_conversion_type(col_order)

        if 0 in self.sig:
            # Degenerate signature: build the mixed order table
            # Start with non-degenerate signature table, ordered as selected by `col_order`
            mult_table = self._build_sig_table(np.where(self.sig, self.sig, 1))

            # Apply zero coefficients, reordered in a way, so as at the end are in `not col_order`
            zero_mask = np.take_along_axis(super()._mtx_table == 0, self._mult_table_res_idx,
                                           axis=self._expand_axis)
            np.put_along_axis(zero_mask, self._mult_table_res_idx, zero_mask,
                              axis=self._res_idx_axis)
            mult_table[zero_mask] = 0
        else:
            mult_table = super()._mtx_table

        self._mtx_mult_table = mult_table

    @property
    def _mtx_table(self) -> layout.NDMultTableType:
        """Multiplication table to be used by to/from matrix conversion"""
        return self._mtx_mult_table
