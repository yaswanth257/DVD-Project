"""Step 2 - example transistor-level netlists, one per gate (deliverable 2): corner 0, input combination 00 / 0.
Each one runs on its own:  ngspice -b Gate_Validation/Netlists/NAND2.cir   (prints RESULT <inputs> i(Vdd) ... v(out))

    python3 Gate_Validation/Scripts/02_write_gate_netlists.py
"""
import subprocess
import pandas as pd
from gate_common import GV, GATE_ORDER, GATES, NGSPICE, CORNERS_CSV, input_combos, netlist

corner = pd.read_csv(CORNERS_CSV).iloc[0].to_dict()
for gate in GATE_ORDER:
    path = GV / "Netlists" / f"{gate}.cir"
    path.write_text(netlist(gate, corner, "../../45nm_MGK.pm", combo=input_combos(gate)[0]))
    out = subprocess.run([NGSPICE, "-b", str(path)], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    print(f"{path.name:10s}", line[0] if line else "NO RESULT\n" + out[:500])
