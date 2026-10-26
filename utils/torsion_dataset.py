"""
utils/torsion_dataset.py
------------------------
Loader for the 187-complex bound/unbound torsion dataset written by
script/calculate_187_prdbv3_protein_torsions.py (location: paths key "torsion_dataset").
"""

import json
from pathlib import Path

import pandas as pd

from utils.paths import get_path

TORSION_DATASET = get_path("torsion_dataset")


def load_torsion_dataset(path=None):
    """
    The dataset as a DataFrame; missing angles are NaN.

    The JSON is read with json.load rather than pd.read_json, whose type inference turns
    PDB codes such as '4E78' into numbers. A .csv / .tsv path is read with PDB codes as strings.
    """
    path = Path(path) if path else TORSION_DATASET
    if path.suffix.lower() == ".json":
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    sep = "\t" if path.suffix.lower() == ".tsv" and "\t" in path.open().readline() else ","
    return pd.read_csv(path, sep=sep, dtype={"PDB": str, "U_PDB": str})
