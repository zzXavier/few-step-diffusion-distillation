"""Report independent retrain variability; require exact repeatable reference arms."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

one,two=map(Path,sys.argv[1:3])
def rows(folder):
    return {(r["method"],r["steps"]):r for r in csv.DictReader((folder/"summary.csv").open())}
a,b=rows(one),rows(two)
assert len(a)==len(b)==9 and a.keys()==b.keys()
assert all(r["cases"]=="6" for r in [*a.values(),*b.values()])
raw_a=json.loads((one/"measurements.json").read_text());raw_b=json.loads((two/"measurements.json").read_text())
assert len(raw_a)==len(raw_b)==54 and {i["file"] for i in raw_a}=={i["file"] for i in raw_b}
reference_matches=0
for row in raw_a:
    f=row["file"]
    if row["method"]!="trained":
        with Image.open(one/"predictions"/f) as i, Image.open(two/"predictions"/f) as j:
            np.testing.assert_array_equal(np.array(i),np.array(j))
        reference_matches+=1
for root in (one,two):
    t=json.loads((root/"training.json").read_text())
    assert t["updates"]==32 and t["adapter_update_l2"]>0
print("Matched fixed-model images:",reference_matches,"of 36")
for key in sorted(a):
    delta=float(b[key]["mean_mse_to_teacher20"])-float(a[key]["mean_mse_to_teacher20"])
    print(*key, "MSE difference",format(delta,"+.8f"))
