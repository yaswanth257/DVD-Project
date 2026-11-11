"""Step 1 - draw the 200 random corners (VDD, temperature, NMOS and PMOS process sets), seed 12175305.
The same corners.csv is reused by Part B (gates) and Part C (test circuit).

    python3 Gate_Validation/Scripts/01_generate_corners.py
"""
import pandas as pd
from gate_common import CORNERS_CSV, draw_corners

df = pd.DataFrame(draw_corners())
df.to_csv(CORNERS_CSV, index=False)
print(f"{len(df)} corners -> {CORNERS_CSV}")
print(df.describe().loc[["min", "max"]].T.to_string())
