"""Geometric algebra multi-vector basic implementation"""
import numbers
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import numpy.typing as npt
from .multivector import MVector

#
# Geometric algebra signature elements (0, +1, or -1)
# Use minimum if 8-bits
#
SigType = np.int8
#
# Multiplication/sign table (0, +1, or -1)
# Combined signature and euclidean-sign swap
#
MultTableType = np.int8
#
# Basis-index to represent the basis-vectors, included in a multi-vector blade
# 0: not-included, 1: included - to use basis-vector boolean mask as index
#
BasisIdxType = np.uint8

NDSigType = npt.NDArray[SigType]
NDMultTableType = npt.NDArray[MultTableType]
NDResultIdxType = npt.NDArray[BasisIdxType]

class ClBase(ABC):
    """Clifford algebra blade container"""
    # Basis-vector dimensions, similar to `clifford.Layout.dims`
    dims: int
    # Blade-name to multi-vector map
    blades: dict[str, MVector]
    # Individual blades, also include 'e1', 'e2', etc.
    scalar: MVector
    I: MVector
    #
    # Masks for basis-vectors in each multi-vector blade
    #
    _blade_masks: npt.NDArray[np.bool]
    #
    # Representation order: map `blades` dict-value index to multi-vector values and v.s.
    #
    _blade_indices: npt.NDArray[np.integer]
    _value_indices: npt.NDArray[np.integer]

    def __init__(self, dims: int, name_prefix: str='e', first_index: int=1) -> None:
        self.dims = dims

        # Select basis-vector masks using binary index
        self._blade_masks = np.indices((2,)*dims, dtype=np.bool)

        # Update blade names, add object attributes
        self._add_blades(name_prefix, first_index)

    def _add_blades(self, name_prefix, first_index) -> None:
        """Assign blade-names as the object attributes"""
        #
        # Select blade names
        #
        blade_names = np.where(self._blade_masks.T, np.arange(self.dims) + first_index, '').T
        blade_names = blade_names.astype(object).sum(0) if self.dims else np.asarray('')
        self.blades = {}

        # Sort blades by grades - number of one-indices, which is the number of basis-vectors
        # like: (0,0,0); (0,0,1), (0,1,0), (1,0,0); (0,1,1), (1,0,1), (1,1,0); (1,1,1)
        # Then, by the smallest basis vector: `e14` (idx:1,0,0,1) is before `e23` (idx:0,1,1,0))
        blade_indices = self._blade_masks.reshape(self.dims, 1<<self.dims).astype(int)
        argsort = np.lexsort(list(-blade_indices)[::-1] + [blade_indices.sum(0)])
        blade_indices = blade_indices.T[argsort]

        # Keep the sort order as map from `nd-value` to `blades` and v.s.
        self._blade_indices = blade_indices.astype(BasisIdxType).T
        self._value_indices = np.empty((2,)*self.dims, dtype=int)
        # The `...` forces l-value to be at least 1‑D (in 0-D scenario it is scalar)
        self._value_indices[*self._blade_indices, ...] = np.arange(self.gaDims)

        # Create blade array of `dtype` minimal integer
        blade_val = np.empty(blade_names.shape, dtype=SigType)
        for idx in blade_indices:
            # Create multi-vector for this blade
            blade_val[...] = 0
            blade_val[*idx] = 1
            blade_mvec = self.mvector(blade_val)
            # Add to `blades` map, the scalar is ''
            name = blade_names[*idx]
            name = name_prefix+name if name else ''
            self.blades[name] = blade_mvec
            # Add it as object attribute, the scalar is 'scalar'
            if name == '':
                name = 'scalar'
            setattr(self, name, blade_mvec)
            # Extra pseudo-scalar property from the last blade
            if all(idx):
                setattr(self, 'I', blade_mvec)

    @property
    def gaDims(self) -> int:    # pylint: disable=invalid-name #HACK: match `clifford` naming
        """Multi-vector dimensions, similar to `clifford.Layout.gaDims`"""
        return 1 << self.dims

    @property
    def gradeList(self) -> npt.NDArray[np.int_]:    # pylint: disable=invalid-name #HACK: match `clifford` naming
        """Map blade-index to its grade, similar to `clifford.Layout.gradeList`"""
        return self._blade_masks.sum(0)

    @abstractmethod
    def mvector(self, value: npt.ArrayLike|numbers.Number) -> MVector:
        """Create a multi-vector from this layout"""

    def from_ndarray(self, value: npt.NDArray, *, axis=-1) -> npt.NDArray[np.object_]:
        """Helper to create array of multi-vectors from array of sorted coefficients"""
        if value.shape[axis] != self.gaDims:
            raise ValueError('Array axis size do not match layout signature')
        return np.apply_along_axis(lambda v: np.asarray(self.mvector(v[self._value_indices])),
                                   axis, value)

    @staticmethod
    def to_ndarray(mvector_arr: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]:
        """Helper to create array of sorted coefficients from array of multi-vectors"""
        # Extract multi-vector coefficients (HACK: use first one to select `dtype`)
        value0 = mvector_arr.item(0).value_sorted
        # Workaround for `np.vectorize()` on scalars for numpy<2.3 (before PR #28624):
        # The `[np.newaxis]`, then `[0]` trick ensures input is non-scalar (for uniform behavior).
        # NOTE: Drop this trick, when "<2.3" becomes obsolete
        otypes = (value0.dtype, value0.shape)
        return np.vectorize(lambda mv: mv.value_sorted, otypes=[otypes])(mvector_arr[np.newaxis])[0]

class Cl(ClBase):
    """Clifford algebra generator (similar to `clifford.Cl()`)"""
    #
    # Algebra signature, similar to `clifford.Layout.sig`
    #
    sig: NDSigType
    #
    # Multiplication tables
    #
    _mult_table: NDMultTableType
    _mult_table_res_idx: NDResultIdxType

    def __init__(self, pos_sig: int|None=None, neg_sig: int=0, zero_sig: int=0, *,
                 sig: npt.ArrayLike|None=None, **kwargs) -> None:
        if sig is None:
            if pos_sig is None:
                raise ValueError('Either pos_sig or sig must be valid')
            # Build signature
            sig = np.array([0] * zero_sig + [1] * pos_sig + [-1] * neg_sig, dtype=SigType)
        elif pos_sig is not None:
            raise ValueError('Both pos_sig and sig are valid')
        else:
            sig = np.asarray(sig, dtype=SigType)

        self.sig = sig
        super().__init__(sig.size, **kwargs)

        #
        # Create multiplication `Cayley` table (result is non-overlapping blades)
        #
        self._mult_table_res_idx = self._build_res_idx_table(np.logical_xor)
        self._mult_table = self._build_sig_table(sig)

    def _build_res_idx_table(self, combine_masks: Callable) -> NDResultIdxType:
        """Table of result indices after combining individual component pairs"""
        # Basis-vector masks of non-overlapping blades for each component combination
        align = (np.newaxis,)*self.dims
        return combine_masks(self._blade_masks[..., *align],
                             self._blade_masks[:, *align]).astype(BasisIdxType)

    def _build_signature_table(self, sig: npt.ArrayLike) -> NDMultTableType:
        """Table to apply basis-vector signatures during component multiplication"""
        # Basis-vector masks of overlapping blades for each component combination
        align = (np.newaxis,)*self.dims
        overlap_mask = self._blade_masks[..., *align] & self._blade_masks[:, *align]
        return np.where(overlap_mask.T, sig, 1).T.prod(axis=0, dtype=SigType)

    def _build_sign_swap_table(self) -> NDMultTableType:
        """Table to apply anti-commutativity of basis-vector swaps"""
        # Count number of basis-swaps in left-component to match the right-component
        # Masks of basis-vectors preceding each component's bases:
        # "(1<<basis_index) - 1" where basis is included, or "0" otherwise
        pre_basis_mask = (self._blade_masks.T << np.arange(self.dims)).T
        pre_basis_mask = np.where(pre_basis_mask, pre_basis_mask - 1, 0)
        # Mask of basis-vectors from right-component preceding bases from left-component
        # (each bit in this mask correspond to a swap operation)
        # shape: <basis>, <left-components>, <right-components>
        swap_bit_mask = (self._blade_masks.T << np.arange(self.dims)).T.sum(0)
        align = (np.newaxis,)*self.dims
        swap_bit_mask = pre_basis_mask[..., *align] & swap_bit_mask
        # Count total numbers of swaps, in order left-component to align to right one
        swap_cnt_table = np.bitwise_count(swap_bit_mask).sum(0, dtype=MultTableType)
        # Select the sign based on swap parity: `-1` odd number of swaps, `1` even number
        return np.where(swap_cnt_table & 1, MultTableType(-1), MultTableType(1))

    def _build_sig_table(self, sig: npt.ArrayLike) -> NDMultTableType:
        """Combined basis anti-commutativity and signature tables"""
        return self._build_signature_table(sig) * self._build_sign_swap_table()

    #@override  #HACK: Python 3.11 compatibility
    def mvector(self, value: npt.ArrayLike|numbers.Number) -> MVector:
        """Create a multi-vector from this layout"""
        return MVector(self, value)

    def __repr__(self) -> str:
        """String representation"""
        return f'{type(self).__name__}(sig={self.sig.tolist()})'

    def __eq__(self, other) -> bool:
        """Algebra comparison"""
        if self is other:   # The algebra-objects are often identical
            return True
        if not isinstance(other, type(self)):
            return False
        return np.array_equal(self.sig, other.sig)

    def do_mul(self, l_value: npt.NDArray, r_value: npt.NDArray) -> MVector:
        """Multi-vector multiplication"""
        # Flatten `nd-values` and `Cayley` tables
        l_shape = l_value.shape[:-self.dims] + (self.gaDims, 1)
        r_shape = r_value.shape[:-self.dims] + (1, self.gaDims)
        val_shape = self.scalar.value.shape
        mult_table = self._mult_table.reshape(self.gaDims, self.gaDims)
        res_idx = np.ravel_multi_index(self._mult_table_res_idx, val_shape)
        res_idx = res_idx.reshape(*mult_table.shape)
        # Row based order: `_mult_table` is rolled along first axis, result is summed along second
        result = l_value.reshape(l_shape) * mult_table
        result = np.take_along_axis(result, res_idx, axis=-2) * r_value.reshape(r_shape)
        result = result.sum(-1, dtype=result.dtype)
        # Un-flatten the result to `nd-value` shape
        result = result.reshape(result.shape[:-self.dims] + val_shape)
        return MVector(self, result)
