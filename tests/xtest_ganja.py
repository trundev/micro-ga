"""Test using `ganja.js` JavaScript library as a reference

See https://github.com/enkimute/ganja.js
"""
import subprocess
import json
import pytest
import numpy as np
import micro_ga
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

def test_blades(pos_sig, neg_sig, zero_sig):
    """Check if our layout has the same blades in the same order"""
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig)
    ganja_basis = run_ganja(GANJA_JS_HDR + f"""
        var layout = Algebra({pos_sig}, {neg_sig}, {zero_sig});
        console.log('{RESULT_TOKEN}' + JSON.stringify(layout.describe().basis));
        """)
    # Convert blade-naming to our style
    def map_basis(n):
        if n == '1':    # Scalar
            return ''
        if 'e0' in ganja_basis: # Degenerate metric starts from '0'
            n = n[0] + ''.join(map(lambda v: chr(ord(v)+1), n[1:]))
        return n
    ganja_basis = list(map(map_basis, ganja_basis))
    assert list(layout.blades.keys()) == ganja_basis, 'Blades are different'

def crop_mvector(mv_val: micro_ga.MVector) -> micro_ga.MVector:
    """Crop individual multi-vector coefficients, so it can be handled by `ganja.js`"""
    # Use some large prime number (negative modulo is to keep the sign of negative values)
    return mv_val.layout.mvector(mv_val.value % np.where(mv_val.value < 0, -1999, 1999))

def test_operations(operation, pos_sig, neg_sig, zero_sig, mvector_2_gen):
    """Check our results vs. `ganja.js` ones"""
    layout = micro_ga.Cl(pos_sig, neg_sig, zero_sig)
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
