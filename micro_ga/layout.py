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

    def __init__(self, pos_sig: int, neg_sig: int=0, zero_sig: int=0, *,
                 dtype: type|None=None) -> None:
        # Build signature
        self.sig = np.array([0] * zero_sig + [1] * pos_sig + [-1] * neg_sig,
                            dtype=SigType)
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
        self._add_blades(dtype)

    def _add_blades(self, dtype: type|None) -> None:
        """Assign blade-names as the object attributes"""
        #
        # Select blade names
        #
        blade_names = np.where(
                self._blade_basis_masks[:, np.newaxis] & 1<<np.arange(self.dims),
                np.arange(self.dims) + 1, '').astype(object).sum(-1)
        self.blades = {}
        blade_val = np.empty(blade_names.size, dtype=dtype)
        # Create 0 and 1 objects of type `dtype`
        zero, one = (0, 1) if dtype in (None, object) else (dtype(0), dtype(1))
        for idx, n in enumerate(blade_names):
            # Create multi-vector for this blade
            blade_val[...] = zero
            blade_val[idx] = one
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

    def __repr__(self) -> str:
        """String representation"""
        dtype_str = self.scalar.subtype.__name__
        return f'{type(self).__name__}(sig={self.sig.tolist()}, dtype={dtype_str})'

    def __eq__(self, other) -> bool:
        """Algebra comparison"""
        if self is other:   # The algebra-objects are often identical
            return True
        if not isinstance(other, type(self)):
            return False
        return np.array_equal(self.sig, other.sig)
