"""Test using `ganja.js` JavaScript library as a reference

See https://github.com/enkimute/ganja.js
"""
import subprocess
import json
import pytest
import numpy as np
import numpy.typing as npt
import micro_ga
import micro_ga.matrix
from . import rng, neg_sig, zero_sig, operation, \
        mvector_gen, mvector_2_gen  # pylint: disable=W0611
# pylint: disable=W0621


GANJA_JS_HDR = 'var Algebra = require("./ganja.js");\n'
RESULT_TOKEN= '*RESULT='

# Test with single positive signature only,
# but multiple `neg_sig` and `zero_sig`signatures
@pytest.fixture(params=[2])
def pos_sig(request):
    """Single test with basis-vectors of positive signature"""
    return request.param

def run_ganja(js_script: str) -> list:
    """Execute `node` to run a JavaScript code from a string"""
    #print('> JavaScript:', js_script)
    result = subprocess.run(['node'], input=js_script, capture_output=True, text=True, check=True)
    print('> result:', result.stdout)
    data = result.stdout.rsplit(RESULT_TOKEN, 1)
    if len(data) < 2:
        raise AssertionError(f'No "{RESULT_TOKEN}" found in JavaScript output')
    return json.loads(data[-1])

def parse_blades(layout: micro_ga.Cl, basis: npt.NDArray[np.str_]
                 ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Convert Algebra.describe() blade structures to more convenient format"""
    basis = np.strings.strip(basis)
    # Identify negative, zero bases and scalar
    sign_table = np.where(basis == '0', 0,
                          np.where(np.strings.startswith(basis, '-'), -1, 1))
    basis = np.strings.lstrip(basis, '-')
    basis[basis == '1'] = ''    # In our list scalar is empty string
    # Convert basis-strings to indices in our blade-list
    found_mask = basis[..., np.newaxis] == np.asarray(tuple(layout.blades.keys()))
    blade_idx = np.argmax(found_mask, axis=-1)
    blade_idx[~found_mask.any(-1)] = -1 # Invalidate not-found indices, like blade is '0'
    return blade_idx, sign_table

def test_blades(pos_sig, neg_sig, zero_sig):
    """Check if our layout has the same blades in the same order"""
    # `ganja.js` uses zero-based indices for degenerate metric
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig, first_index=0 if zero_sig else 1)
    ganja_basis = run_ganja(GANJA_JS_HDR + f"""
        var layout = Algebra({pos_sig}, {neg_sig}, {zero_sig});
        console.log('{RESULT_TOKEN}' + JSON.stringify(layout.describe().basis));
        """)
    # Convert blade-naming to our style
    ganja_res_idx, ganja_sign_table = parse_blades(layout, np.array(ganja_basis))
    np.testing.assert_equal(ganja_sign_table, 1, 'Unexpected minus sign / zero in a base name')
    np.testing.assert_equal(ganja_res_idx, np.arange(layout.gaDims), 'Blade order is different')

def test_mul_table(pos_sig, neg_sig, zero_sig):
    """Check if our layout use the same multiplication table"""
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig, first_index=0 if zero_sig else 1)
    mul_table = run_ganja(GANJA_JS_HDR + f"""
        var layout = Algebra({pos_sig}, {neg_sig}, {zero_sig});
        console.log('{RESULT_TOKEN}' + JSON.stringify(layout.describe().mulTable));
        """)
    # Convert blade-naming to our style
    ganja_res_idx, ganja_sign_table = parse_blades(layout, np.array(mul_table))
    our_sign_table = layout._mult_table     # pylint: disable=protected-access
    our_res_idx = np.where(our_sign_table != 0, layout._mult_table_res_idx, -1) # pylint: disable=protected-access
    np.testing.assert_equal(our_res_idx, ganja_res_idx, 'ganja Algebra.mulTable mismatch')
    np.testing.assert_equal(our_sign_table, ganja_sign_table,
                            'ganja Algebra.mulTable signature mismatch')

def crop_mvector(mv_val: micro_ga.MVector) -> micro_ga.MVector:
    """Crop individual multi-vector coefficients, so it can be handled by `ganja.js`"""
    # Use some large prime number (negative modulo is to keep the sign of negative values)
    return mv_val.layout.mvector(mv_val.value % np.where(mv_val.value < 0, -1999, 1999))

def test_operations(operation, pos_sig, neg_sig, zero_sig, mvector_2_gen):
    """Check our results vs. `ganja.js` ones"""
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig, first_index=0 if zero_sig else 1)
    # Prepare some common JavaScript stuff
    js_script = GANJA_JS_HDR + f"var layout = Algebra({pos_sig}, {neg_sig}, {zero_sig});\n"
    ganja_op = operation.__name__.title()

    # Prepare script to run operation with all value combinations
    js_script += "res_arr = Array();"
    val_list = []
    for l_val, r_val in mvector_2_gen(layout):
        # Limit operation argument to avoid `ganja` rounding
        l_val = crop_mvector(l_val)
        r_val = crop_mvector(r_val)
        val_list.append((l_val, r_val))
        # Update JavaScript to do single operation
        js_script += f"""
            l_val = new layout({l_val.value.tolist()});
            r_val = new layout({r_val.value.tolist()});
            res = l_val.{ganja_op}(r_val);
            console.log('{ganja_op}( ' + l_val.toString() + ', ' + r_val.toString() + ' )'
                        + ' -> ' + res.toString());
            res_arr.push(Array.from(res));
            """

    # Run JavaScript to calculate reference results
    js_script += f"console.log('{RESULT_TOKEN}' + JSON.stringify(res_arr));"
    js_res = run_ganja(js_script)

    # Iterate over some picked value combinations
    for (l_val, r_val), ref_res in zip(val_list, js_res, strict=True):
        ref_res = layout.mvector(ref_res)
        our_res = operation(l_val, r_val)
        np.testing.assert_equal(our_res, ref_res)

@pytest.mark.parametrize('zero_sig', [0, pytest.param(1,
        marks=pytest.mark.xfail(reason="Degenerate metric produce different matrix"))])
def test_matrix_form(pos_sig, neg_sig, zero_sig, mvector_gen):
    """Check multi-vector matrix-form conversion"""
    layout = micro_ga.matrix.Cl(pos_sig, neg_sig, zero_sig, first_index=0 if zero_sig else 1)
    matrix = run_ganja(GANJA_JS_HDR + f"""
        var layout = Algebra({pos_sig}, {neg_sig}, {zero_sig});
        console.log('{RESULT_TOKEN}' + JSON.stringify(layout.describe().matrix));
        """)
    ganja_res_idx, ganja_sign_table = parse_blades(layout, np.array(matrix))

    # Iterate over some picked values
    for our_val in mvector_gen(layout):
        # Convert to matrix-form using `ganja` rules
        ref_mtx = our_val.value[ganja_res_idx] * ganja_sign_table
        # Test if it matches our version
        our_mtx = layout.to_matrix(our_val)
        np.testing.assert_equal(our_mtx, ref_mtx, f'Matrix-form mismatch for {our_val}')
