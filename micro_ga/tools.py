"""Extra Geometric Algebra tools

- Expand complex coefficients (expand_complex):
  Inspired by this "Zundamon's Theorem" video: https://www.youtube.com/watch?v=HbUewIIpl6I

  Convert a multi-vector with complex coefficients to one with real coefficient, but
  from new algebra with extra blade. The extra blade doubles the number of coefficients,
  which makes room for real and imaginary components separately.
  The imaginary unit is replaced by a basis-vector, which is a merge from the extra blade
  and some existing basis-vector. The signature of the extra blade is selected, to make
  this new basis-vector to square to -1.
  For example:
  - 2D multi-vector (`s + x*e1 + y*e2 + a*e12`):
    is expanded based on `e1` blade, by adding positive `e3`, so imaginary unit is `e13`:
    real components: `s_r + x_r*e1 + y_r*e2 + a_r*e12`
    imaginary components:
      `(s_i + x_i*e1 + y_i*e2 + a_i*e12)*e13 = s_i*e13 + x_i*e3 - y_i*e123 - a_i*e23`
"""
from typing import Sequence
import numpy as np
import numpy.typing as npt
from . import layout
from .multivector import MVector

# Use `sympy` to extract real/imaginary components and simplify (if installed)
try:
    from sympy import re as sympy_re, im as sympy_im, simplify as sympy_simplify
    #TODO: This converts all coefficients to `sympy.Basic` type
    wrap_real = np.vectorize(sympy_re, otypes=[object])
    wrap_imag = np.vectorize(sympy_im, otypes=[object])
except ImportError as _:
    wrap_real = lambda arr: arr.real
    wrap_imag = lambda arr: arr.imag
    sympy_simplify = lambda v: v


#
# Conversion to sympy expression
#
if 'sympy_re' in globals():   # sympy is installed
    from sympy import Basic as SympyBasic, Expr as SympyExpr, Symbol as SympySymbol, S as SympyS

    def _blade_symbols(layout: layout.Cl) -> npt.NDArray[np.object_]:
        """Create symbols for all blades (multiplication non-commutative)"""
        return np.fromiter((SympySymbol(k, commutative=False) if k else SympyS.One
                            for k in layout.blades.keys()), dtype=SympyExpr)

    def mvector_sympify(mvect: MVector) -> SympyExpr:
        """Convert multi-vector to a `sympy` expression"""
        return (mvect.value * _blade_symbols(mvect.layout)).sum()

    def sympy_blade_rules(layout: layout.Cl, mvect: SympyExpr) -> SympyBasic:
        """Apply layout's blade-product rules on `sympy` expression"""
        mult_table = layout._mult_table
        res_idx = layout._mult_table_res_idx
        symbols = _blade_symbols(layout)
        # Must expand to make `subs()` working
        mv_res: SympyBasic = mvect.expand()
        for l_idx, r_idx in np.ndindex(mult_table.shape):
            l_sym, r_sym = symbols[[l_idx, r_idx]]
            sign = mult_table[l_idx, r_idx]
            res_sym = symbols[res_idx[l_idx, r_idx]]
            mv_res = mv_res.subs(l_sym * r_sym, sign * res_sym)
        return mv_res

#
# Complex expansion
#
def _layout_add_basis(layout: layout.Cl, extra_sig: Sequence
                      ) -> tuple[layout.Cl, npt.NDArray[np.bool], MVector]:
    """Create algebra with extra blade(s)"""
    sig = layout.sig.tolist() + np.asarray(extra_sig).flatten().tolist()
    new_layout = type(layout)(sig=sig)
    # Prepare info about original blades
    new_basis_idx = layout.dims + np.arange(len(extra_sig))
    new_basis_mask = np.bitwise_or.reduce(1<<new_basis_idx)
    orig_blades_mask = ~(new_layout._blade_basis_masks & new_basis_mask).astype(bool)
    # Prepare info about newly create blade
    new_blade = np.argmax(new_layout._blade_basis_masks == new_basis_mask)
    new_blade = tuple(new_layout.blades.values())[new_blade]
    return new_layout, orig_blades_mask, new_blade

def expand_complex(mvector: MVector, base_blade: int|str, *, simplify: bool=False) -> MVector:
    """Expand complex coefficients in a multi-vector to real coefficients in larger algebra

    Example:
    - 2D complex multi-vector to 3D real: `s + x*e1 + y*e2 + a*e12`,
      `base_basis='e1'` -> `imag_blade='e13'`

      blade |  1  | e1  | e2  | e3  | e12 | e13 | e23 | e123
      ------|-----|-----|-----|-----|-----|-----|-----|-----
      real  | s_r | x_r | y_r |     | a_r |     |     |
      imag  |     |     |     | x_i |     | s_i |-a_i |-y_r
    """
    layout = mvector.layout
    if isinstance(base_blade, str):
        base_blade = tuple(layout.blades.keys()).index(base_blade)

    #
    # Select signature for the new basis-vector, the new blade must:
    # - commute with all blades -> even number of basis-vector
    # - square to -1 -> number of basis-vector is 2, 3, 6, 7...
    # i.e. need to add 2, 6, 10.... basis-vectors
    new_basis_sig = 1, 1

    # Create algebra with extra basis-vector(s)
    new_layout, orig_blades_mask, new_blade = _layout_add_basis(layout, new_basis_sig)
    blades_arr = np.asarray(tuple(new_layout.blades.values()))
    # Isolate the blade to represent the imaginary uint
    imag_blade = blades_arr[base_blade] * new_blade

    # Combine expanded multi-vector values:
    # - real components go to original blades
    # - imaginary components - imaginary unit is replaced by `imag_blade`
    orig_blades = blades_arr[orig_blades_mask]
    new_mvector = (wrap_real(mvector.value) * orig_blades).sum()
    new_mvector += (wrap_imag(mvector.value) * orig_blades * imag_blade).sum()
    if simplify:
        new_mvector.value[...] = sympy_simplify(new_mvector.value)
    return new_mvector
