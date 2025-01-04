"""Geometric algebra multi-vector basic implementation"""
import numbers
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
# Bit-mask to represent the basis-vectors, included in a multi-vector blade
# 16-bits allow max of 16 basis-vectors (2**16 == 65536 blades) - ought to be enough!
#
BasisBitmaskType = np.uint16

NDSigType = npt.NDArray[SigType]
NDMultTableType = npt.NDArray[MultTableType]
NDResultIdxType = npt.NDArray[np.int_]

class Cl:
    """Clifford algebra generator (similar to `clifford.Cl()`)"""
    #
    # Algebra signature, similar to `clifford.Layout.sig`
    #
    sig: NDSigType
    # Basis-vector dimensions, similar to `clifford.Layout.dims`
    dims: int
    # Blade-name to multi-vector map
    blades: dict[str, MVector]
    # Individual blades, also include 'e1', 'e2', etc.
    scalar: MVector
    I: MVector
    #
    # Bit-masks for basis-vectors in each multi-vector blade
    #
    _blade_basis_masks: npt.NDArray[BasisBitmaskType]
    #
    # Multiplication tables
    #
    _mult_table: NDMultTableType
    _mult_table_res_idx: NDResultIdxType

    def __init__(self, pos_sig: int|None=None, neg_sig: int=0, zero_sig: int=0, *,
                 sig: npt.ArrayLike|None=None) -> None:
        if sig is None:
            if pos_sig is None:
                raise ValueError('Either pos_sig or sig must be valid')
            # Build signature
            self.sig = np.array([0] * zero_sig + [1] * pos_sig + [-1] * neg_sig,
                                dtype=SigType)
        elif pos_sig is not None:
            raise ValueError('Both pos_sig and sig are valid')
        else:
            self.sig = np.asarray(sig, dtype=SigType)
        self.dims = self.sig.size
        #
        # Select bit-masks for all available blades
        #
        blade_masks = np.arange(1<<self.dims, dtype=BasisBitmaskType)
        # Sort by grades - number of set-bits, which is the number of basis-vectors
        # like: 000b; 001b, 010b, 100b; 011b, 101b, 110b; 111b
        # Then, by the smallest basis vector: `e14` (mask 9) is before `e23` (mask 6)
        argsort = np.lexsort(list(-(blade_masks & 1<<np.arange(self.dims)[:, np.newaxis]))[::-1]
                             + [np.bitwise_count(blade_masks)])
        self._blade_basis_masks = blade_masks[argsort]
        # Update blade names, add object attributes
        self._add_blades()
        #
        # Create multiplication tables
        #
        self._build_mult_table()

    def _add_blades(self) -> None:
        """Assign blade-names as the object attributes"""
        #
        # Select blade names
        #
        blade_names = np.where(
                self._blade_basis_masks[:, np.newaxis] & 1<<np.arange(self.dims),
                np.arange(self.dims) + 1, '').astype(object).sum(-1)
        self.blades = {}
        # Create blade array of `dtype` minimal integer
        blade_val = np.empty(blade_names.size, dtype=SigType)
        for idx, n in enumerate(blade_names):
            # Create multi-vector for this blade
            blade_val[...] = 0
            blade_val[idx] = 1
            blade_mvec = self.mvector(blade_val)
            # Add to `blades` map, the scalar is ''
            name = 'e'+n if n else ''
            self.blades[name] = blade_mvec
            # Add it as object attribute, the scalar is 'scalar'
            if name == '':
                name = 'scalar'
            setattr(self, name, blade_mvec)
            # Extra pseudo-scalar property from the last blade
            if idx + 1 == 1 << self.dims:
                setattr(self, 'I', blade_mvec)

    def _build_mult_table(self) -> None:
        """Create multiplication tables"""
        #
        # Select component-indices for each element in table
        #
        # Bit-masks of non-overlapping blades for each component combination
        result_masks = self._blade_basis_masks[:, np.newaxis] ^ self._blade_basis_masks
        # Convert bit-masks to component-indices
        inv_sort = np.arange(self._blade_basis_masks.size)
        inv_sort[self._blade_basis_masks] = inv_sort
        self._mult_table_res_idx = inv_sort[result_masks]

        #
        # Table to apply basis-vector signatures during component multiplication
        #
        # Bit-masks of overlapping blades for each component combination
        overlap_mask = self._blade_basis_masks[:, np.newaxis] & self._blade_basis_masks
        # Convert to Boolean-mask where each basis-vector overlap
        # shape: <left-component>, <right-component>, <basis>
        overlap_mask = (overlap_mask[..., np.newaxis] & 1<<np.arange(self.dims)).astype(bool)
        signature_table = np.where(overlap_mask, self.sig, 1).prod(axis=-1, dtype=SigType)

        #
        # Table to apply anti-commutativity of basis-vector swaps
        #
        # Count number of basis-swaps in left-component to match the right-component
        # Bit-masks of basis-vectors preceding each component's bases:
        # "1<<(basis_index - 1)" where basis is included, or "0" otherwise
        pre_basis_mask = self._blade_basis_masks[:, np.newaxis] \
                & 1<<np.arange(self.dims, dtype=self._blade_basis_masks.dtype)
        pre_basis_mask = np.where(pre_basis_mask, pre_basis_mask - 1, 0)
        # Bit-mask of basis-vectors from right-component preceding bases from left-component
        # (each bit in this mask correspond to a swap operation)
        # shape: <left-component>, <basis>, <right-component>
        swap_mask = pre_basis_mask[..., np.newaxis] & self._blade_basis_masks
        # Count total numbers of swaps, in order left-component to align to right one
        swap_cnt_table = np.bitwise_count(swap_mask).sum(1, dtype=MultTableType)
        # Select the sign based on swap parity
        np.testing.assert_equal(swap_cnt_table & 1,
                                np.logical_xor.reduce(np.bitwise_count(swap_mask) & 1, axis=1))
        sign_table = np.where(swap_cnt_table & 1,
                              MultTableType(-1),    # need odd number of swaps
                              MultTableType(1))     # need even number of swaps
        #
        # Combine sign-swap and signature tables
        #
        self._mult_table = sign_table * signature_table

    @property
    def gaDims(self) -> int:    # pylint: disable=invalid-name #HACK: match `clifford` naming
        """Multi-vector dimensions, similar to `clifford.Layout.gaDims`"""
        return 1 << self.dims

    @property
    def gradeList(self) -> npt.NDArray[np.int_]:    # pylint: disable=invalid-name #HACK: match `clifford` naming
        """Map blade-index to its grade, similar to `clifford.Layout.gradeList`"""
        return np.bitwise_count(self._blade_basis_masks)

    def mvector(self, value: npt.ArrayLike|numbers.Number) -> MVector:
        """Create a multi-vector from this layout"""
        return MVector(self, value)

    def from_ndarray(self, value: npt.ArrayLike, *, axis=-1) -> npt.NDArray[np.object_]:
        """Helper to create array of multi-vectors from array of coefficients"""
        return np.apply_along_axis(lambda v: np.asarray(self.mvector(v)), axis, value)

    @staticmethod
    def to_ndarray(mvector_arr: npt.NDArray[np.object_]) -> npt.NDArray[np.object_]:
        """Helper to create array of coefficients from array of multi-vectors"""
        # Extract multi-vector coefficients (HACK: use first one to select `dtype`)
        value0 = mvector_arr.item(0).value
        # Note: `vectorize()` on scalars do not need `otype` dimensions
        otypes = (value0.dtype, value0.shape) if mvector_arr.shape else value0.dtype
        return np.vectorize(lambda mv: mv.value, otypes=[otypes])(mvector_arr)

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
        product = l_value[:, np.newaxis] * self._mult_table * r_value
        result = np.zeros_like(product, shape=l_value.shape)
        np.add.at(result, self._mult_table_res_idx, product)
        return MVector(self, result)
