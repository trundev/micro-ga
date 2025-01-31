"""Test using `Grassmann.jl` Julia package as a reference

See https://github.com/chakravala/Grassmann.jl
"""
import pytest
import numpy as np
from juliacall import Main as jl
from . import rng, pos_sig, neg_sig, zero_sig, layout, operation, \
        mvector_gen, mvector_2_gen  # pylint: disable=W0611
# pylint: disable=W0621


@pytest.fixture(autouse=True)
def grassmann_layout(layout):
    """`Grassmann` algebra object auto-use fixture"""
    jl.seval(f"""using Grassmann; layout, blades... = @basis"{layout.sig}" layout e""")

@pytest.fixture
def grassmann_op(operation):
    """Convert operation to `Grassmann` string"""
    return {'add': ' + ', 'sub': ' - ', 'mul': ' * '}[operation.__name__]

def test_blades(layout):
    """Check if our layout has the same blades in the same order"""
    for name, blade in layout.blades.items():
        # `0e` is to prevent scalar-only collapse
        gr_value = jl.seval(f'Multivector({blade} + 0e).v')
        np.testing.assert_equal(blade.value, gr_value, f'Different blade {name} values')

@pytest.mark.parametrize('operation', ['*'], ids=['mul'])
def test_mul_table(layout, operation):
    """Check if multiplication table matches `Grassmann.cayley` one"""
    # Get the `cayley` table, basis-vectors inside and algebra blades
    cayley, cayley_bases, ref_blades = jl.seval(f"""
        table = cayley(layout, {operation}).v.v
        table = map(row->row.v, table)
        bases = map(row -> map(basis, row), table)
        table, bases, blades
        """)
    cayley = np.array(list(cayley))
    cayley_bases = np.array(list(cayley_bases))
    ref_blades = np.fromiter(ref_blades, dtype=object)
    # Convert basis-objects to indices in list of algebra blades
    gr_res_idx = np.argmax(cayley_bases[..., np.newaxis] == ref_blades, axis=-1)
    np.testing.assert_equal(layout._mult_table_res_idx, gr_res_idx.T,   # pylint: disable=protected-access
                            'Grassman multiplication table mismatch')
    # Extract signatures only (array includes `Submanifold` and `Single` objects)
    gr_mult_table = np.vectorize(lambda v: getattr(v, 'v', 1))(cayley)
    np.testing.assert_equal(layout._mult_table, gr_mult_table.T,    # pylint: disable=protected-access
                            'Grassman multiplication signature table mismatch')

def test_operations(layout, operation, grassmann_op, mvector_2_gen):
    """Check our results vs. `Grassmann.jl` ones"""
    # Prepare Julia function to run `Grassmann` operation, get the blade objects
    ref_blades = jl.seval(f"""
        function ref_op(l_val, r_val)
            return l_val {grassmann_op} r_val
            end
        return blades
        """)
    ref_blades = np.fromiter(ref_blades, dtype=object)
    # Iterate over some picked value combinations
    for our_l_val, our_r_val in mvector_2_gen(layout):
        # Convert values to Julia / `Grassmann` objects
        ref_l_val = (our_l_val.value * ref_blades).sum()
        ref_r_val = (our_r_val.value * ref_blades).sum()
        ref_res = jl.ref_op(ref_l_val,  ref_r_val)
        # Compare to the result from `micro-ga`
        our_res = operation(our_l_val, our_r_val)
        np.testing.assert_equal(our_res.value, ref_res.v)
