"""Geometric algebra multi-vector basic implementation"""
import numbers
from typing import Union, Callable, ForwardRef, TYPE_CHECKING
import numpy as np
import numpy.typing as npt

# Treat `sympy` types as "numbers" (if installed)
try:
    import sympy.core.basic
    SympyTypes = (sympy.core.basic.Basic,)
except ImportError as _:
    SympyTypes: tuple[type, ...] = ()

# Avoid circular imports, while type checking works
if TYPE_CHECKING:
    from .layout import Cl  # pragma: no cover # pylint: disable=wrong-spelling-in-comment
else:
    Cl = ForwardRef('Cl')

# Multi-vector operation `other` argument type
NumberTypes = (numbers.Number,) + SympyTypes
OtherArg = Union['MVector', int, complex, *NumberTypes]

class MVector:
    """Multi-vector representation"""
    layout: Cl
    value: npt.NDArray

    def __init__(self, layout: Cl, value: npt.ArrayLike|numbers.Number) -> None:
        # Handle various input type options
        if isinstance(value, NumberTypes):
            # Ensure blade-type will be up-scaled for integers
            val = layout.scalar.value * np.asarray(value)
        else:
            val = np.array(value)
            if layout.gaDims != val.size:
                raise ValueError('Coefficients do not match layout signature')
        self.layout = layout
        self.value = val

    @property
    def subtype(self) -> type:
        """Type of underlying objects"""
        # Start with native `numpy` data-type
        subtype = self.value.dtype.type
        if subtype == np.object_:
            # Type of underlying object
            subtype = type(self.value.item(0))
            # Report all `sympy` types as `Basic`
            if issubclass(subtype, SympyTypes):
                subtype = SympyTypes[0]
        return subtype

    def _value_astype(self, dtype: type) -> npt.NDArray:
        """Convert `value` to a different data type, original array if type matches"""
        value = np.asarray(self.value, dtype=dtype)
        if value.dtype.type == np.object_ and dtype != object:
            # Convert individual values to requested non `numpy` type
            value = np.vectorize(dtype, otypes=[object])(value)
        return value

    def astype(self, dtype: type) -> 'MVector':
        """Convert to a multi-vector of different data type"""
        return MVector(self.layout, self._value_astype(dtype))

    def _to_string(self, str_fn: Callable, *, tol: float=0, mult_sym='*') -> str:
        """String representation, strips near-zero blades"""
        vals = self.value
        nz_mask = np.abs(vals) > tol
        if not nz_mask.any():
            return '0'
        vals = vals[nz_mask]

        # Start with strings to join blades (`object` is to allow `+=`)
        el_strs = np.full(vals.shape, ' + ', dtype=object)
        el_strs[0] = ''
        if issubclass(self.subtype, (int, float, np.integer, np.floating)):
            # Only primitive scalar types are joined using their sign
            el_strs[1:][vals[1:] < 0] = ' - '
            vals[1:] = abs(vals[1:])

        # Convert each coefficient to string using `str_fn`
        el_strs += np.vectorize(str_fn, otypes=[str])(vals)

        # Add blade-names
        blade_strs = np.array(tuple(self.layout.blades.keys()))[nz_mask]
        el_strs += np.where(blade_strs, mult_sym, '') + blade_strs
        return el_strs.sum()

    def __str__(self) -> str:
        return self._to_string(str)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._to_string(repr)}, subtype={self.subtype.__name__})'

    def __round__(self, ndigits: int=0) -> 'MVector':
        """Implement built-in round(), esp. for `dtype=object`"""
        vals = self.value
        #HACK: `numpy.round()` crashes if `dtype=object` and underlying object has no `rint` method
        # But, `round()` does NOT work with complex: "type complex doesn't define __round__ method"
        if vals.dtype == object:
            vals = np.vectorize(round, otypes=[object])(vals, ndigits=ndigits)
        else:
            vals = vals.round(decimals=ndigits)
        return MVector(self.layout, vals)

    def _get_other_value(self, other: OtherArg) -> npt.NDArray:
        """Convert values of an operation argument to match ours"""
        # Check if it is scalar
        if isinstance(other, NumberTypes):
            # Ensure result `dtype` matches our and `other` types:
            # - type `object` must persist (allows unbounded integers)
            # - when `other` is integer, scalar-blade type will be up-scaled
            dtype = np.promote_types(self.value.dtype, type(other))
            return self.layout.scalar._value_astype(dtype.type) * other # pylint: disable=protected-access
        if not isinstance(other, MVector):
            return NotImplemented
        if self.layout != other.layout:
            return NotImplemented
        return other.value

    def __eq__(self, other) -> bool:
        """Multi-vector comparison"""
        value = self._get_other_value(other)
        if value is NotImplemented:
            return NotImplemented
        return (self.value == value).all()

    def __add__(self, other: OtherArg) -> 'MVector':
        """Left-side addition"""
        value = self._get_other_value(other)
        if value is NotImplemented:
            return NotImplemented
        return MVector(self.layout, self.value + value)

    __radd__ = __add__

    def __neg__(self) -> 'MVector':
        """Negation"""
        return MVector(self.layout, -self.value)

    def __sub__(self, other: OtherArg) -> 'MVector':
        """Left-side subtraction"""
        return self.__add__(-other)

    def __rsub__(self, other: OtherArg) -> 'MVector':
        """Right-side subtraction"""
        return self.__neg__().__add__(other)

    def __mul__(self, other: OtherArg) -> 'MVector':
        """Left-side multiplication"""
        value = self._get_other_value(other)
        if value is NotImplemented:
            return NotImplemented
        return self.layout.do_mul(self.value, value)

    def __rmul__(self, other: OtherArg) -> 'MVector':
        """Right-side multiplication"""
        value = self._get_other_value(other)
        if value is NotImplemented:
            return NotImplemented
        return self.layout.do_mul(value, self.value)
