"""Shared definitions for Part C: 3-bit Carry-Lookahead Adder (CLA) test circuit.

The CLA adds two 3-bit numbers A[2:0] and B[2:0] with carry-in Cin and produces
sum S[2:0] and carry-out Cout. It is built entirely from NAND2, AND2, and OR2 gates
(gate sizing from gate_common.py). XOR is implemented with four NAND2 gates; the carries
c1, c2, cout are carry-lookahead (two-level AND-OR of the generate / propagate bits and cin).

Circuit inputs: a0, a1, a2, b0, b1, b2, cin  (7 primary inputs, 128 combinations)
Circuit outputs: s0, s1, s2, cout_net
Total gates: 12 AND2 + 24 NAND2 + 6 OR2 = 42 gates

Run every script from the repository root, e.g.
    python3 Test_Circuit/Scripts/07_write_cla_netlist.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Gate_Validation" / "Scripts"))

from gate_common import LMIN, MODEL_FILE, NGSPICE, WN_INV, WN_NAND, WP_INV, WP_NAND, WN_NOR, WP_NOR  # noqa: E402

TC = ROOT / "Test_Circuit"
CORNERS_CSV = ROOT / "Gate_Validation" / "Results" / "corners.csv"
GATE_ML_CSV = ROOT / "Gate_Validation" / "Results" / "gate_ml_results.csv"
SPICE_CSV   = TC / "Results" / "cla_spice_results.csv"
ML_CSV      = TC / "Results" / "cla_ml_results.csv"
TABLE_CSV   = TC / "Results" / "cla_results_table.csv"
PARITY_PNG  = TC / "Plots" / "cla_parity.png"
NETLIST_PATH = TC / "Netlists" / "CLA3.cir"

CIRCUIT_NAME = "CLA3"
INPUTS = ["a0", "a1", "a2", "b0", "b1", "b2", "cin"]   # 7 primary inputs

# ===========================================================================
# Gate instance list: (instance_name, gate_type, [in1, in2], output_net)
# Gate types must match gate_common.GATES keys (the gates validated in Part B): NAND2, AND2, OR2
# Carry-lookahead: every carry is computed directly from the generate / propagate bits and cin
# (two-level AND-OR, built from 2-input gates), so no carry waits for the previous one:
#   c1   = g0 + p0.cin
#   c2   = g1 + p1.g0 + p1.p0.cin
#   cout = g2 + p2.g1 + p2.p1.g0 + p2.p1.p0.cin
# ===========================================================================
GATES = [
    # Generate bits: G_i = A_i AND B_i
    ("and_g0",  "AND2",  ["a0", "b0"],       "g0"),
    ("and_g1",  "AND2",  ["a1", "b1"],       "g1"),
    ("and_g2",  "AND2",  ["a2", "b2"],       "g2"),
    # Propagate bits: P_i = A_i XOR B_i, implemented as 4-NAND2 XOR
    ("xp0_n1",  "NAND2", ["a0", "b0"],       "n1p0"),
    ("xp0_n2",  "NAND2", ["a0", "n1p0"],     "n2p0"),
    ("xp0_n3",  "NAND2", ["b0", "n1p0"],     "n3p0"),
    ("xp0_n4",  "NAND2", ["n2p0", "n3p0"],   "p0"),
    ("xp1_n1",  "NAND2", ["a1", "b1"],       "n1p1"),
    ("xp1_n2",  "NAND2", ["a1", "n1p1"],     "n2p1"),
    ("xp1_n3",  "NAND2", ["b1", "n1p1"],     "n3p1"),
    ("xp1_n4",  "NAND2", ["n2p1", "n3p1"],   "p1"),
    ("xp2_n1",  "NAND2", ["a2", "b2"],       "n1p2"),
    ("xp2_n2",  "NAND2", ["a2", "n1p2"],     "n2p2"),
    ("xp2_n3",  "NAND2", ["b2", "n1p2"],     "n3p2"),
    ("xp2_n4",  "NAND2", ["n2p2", "n3p2"],   "p2"),
    # Lookahead carry c1 = g0 + p0.cin
    ("and_c1a", "AND2",  ["p0", "cin"],      "c1a"),
    ("or_c1",   "OR2",   ["g0", "c1a"],      "c1"),
    # Lookahead carry c2 = g1 + p1.g0 + p1.p0.cin
    ("and_c2a", "AND2",  ["p1", "g0"],       "c2a"),
    ("and_p10", "AND2",  ["p1", "p0"],       "p10"),
    ("and_c2b", "AND2",  ["p10", "cin"],     "c2b"),
    ("or_c2a",  "OR2",   ["g1", "c2a"],      "c2o"),
    ("or_c2",   "OR2",   ["c2o", "c2b"],     "c2"),
    # Lookahead carry cout = g2 + p2.g1 + p2.p1.g0 + p2.p1.p0.cin
    ("and_c3a", "AND2",  ["p2", "g1"],       "c3a"),
    ("and_p21", "AND2",  ["p2", "p1"],       "p21"),
    ("and_c3b", "AND2",  ["p21", "g0"],      "c3b"),
    ("and_p210", "AND2", ["p21", "p0"],      "p210"),
    ("and_c3c", "AND2",  ["p210", "cin"],    "c3c"),
    ("or_c3a",  "OR2",   ["g2", "c3a"],      "c3o1"),
    ("or_c3b",  "OR2",   ["c3b", "c3c"],     "c3o2"),
    ("or_cout", "OR2",   ["c3o1", "c3o2"],   "cout_net"),
    # Sum bits: S_i = P_i XOR C_i, implemented as 4-NAND2 XOR (c0 = cin)
    ("xs0_n1",  "NAND2", ["p0", "cin"],      "sn1_0"),
    ("xs0_n2",  "NAND2", ["p0", "sn1_0"],    "sn2_0"),
    ("xs0_n3",  "NAND2", ["cin", "sn1_0"],   "sn3_0"),
    ("xs0_n4",  "NAND2", ["sn2_0", "sn3_0"], "s0"),
    ("xs1_n1",  "NAND2", ["p1", "c1"],       "sn1_1"),
    ("xs1_n2",  "NAND2", ["p1", "sn1_1"],    "sn2_1"),
    ("xs1_n3",  "NAND2", ["c1", "sn1_1"],    "sn3_1"),
    ("xs1_n4",  "NAND2", ["sn2_1", "sn3_1"], "s1"),
    ("xs2_n1",  "NAND2", ["p2", "c2"],       "sn1_2"),
    ("xs2_n2",  "NAND2", ["p2", "sn1_2"],    "sn2_2"),
    ("xs2_n3",  "NAND2", ["c2", "sn1_2"],    "sn3_2"),
    ("xs2_n4",  "NAND2", ["sn2_2", "sn3_2"], "s2"),
]

# Mapping of gate type to boolean evaluation function
_GATE_EVAL = {
    "NAND2": lambda a, b: int(not (a and b)),
    "AND2":  lambda a, b: int(a and b),
    "OR2":   lambda a, b: int(a or b),
}


def evaluate_circuit(a0, a1, a2, b0, b1, b2, cin):
    """Evaluate the CLA combinational logic for the given primary inputs (all 0 or 1).
    Returns a dict mapping every net name to its logic value (0 or 1).
    Primary inputs are included, then every gate output is computed in GATES order.
    """
    nets = {"a0": a0, "a1": a1, "a2": a2, "b0": b0, "b1": b1, "b2": b2, "cin": cin,
            "0": 0, "vdd": 1}
    for (_, gate_type, ins, out_net) in GATES:
        in_vals = [nets[n] for n in ins]
        nets[out_net] = _GATE_EVAL[gate_type](*in_vals)
    return nets


def all_input_combos():
    """Return list of 128 input tuples (a0, a1, a2, b0, b1, b2, cin) in binary order."""
    combos = []
    for code in range(128):
        a0  = (code >> 6) & 1
        a1  = (code >> 5) & 1
        a2  = (code >> 4) & 1
        b0  = (code >> 3) & 1
        b1  = (code >> 2) & 1
        b2  = (code >> 1) & 1
        cin = code & 1
        combos.append((a0, a1, a2, b0, b1, b2, cin))
    return combos


def combo_label(combo):
    """7-character binary string for an input combo tuple."""
    return "".join(str(b) for b in combo)


def fmt(v):
    return f"{float(v):.16g}"


def check_adder():
    """Assert that the gate list adds: s2 s1 s0 and cout equal a + b + cin for all 128 input combinations."""
    for combo in all_input_combos():
        a0, a1, a2, b0, b1, b2, cin = combo
        nets = evaluate_circuit(*combo)
        total = (a0 + 2 * a1 + 4 * a2) + (b0 + 2 * b1 + 4 * b2) + cin
        got = nets["s0"] + 2 * nets["s1"] + 4 * nets["s2"] + 8 * nets["cout_net"]
        assert got == total, f"adder error for {combo_label(combo)}: {got} != {total}"


check_adder()
