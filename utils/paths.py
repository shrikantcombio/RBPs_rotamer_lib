"""
utils/paths.py
--------------
File locations for RBPs_rotamer_lib scripts.

Defaults below are relative to the repository root; machine-specific locations go in
config/paths.json (git-ignored; template: config/paths.example.json).
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULTS = {
    # Equal-residue bound/unbound PDB pairs (<C_PDB>_<U_PDB>_C.pdb / _U.pdb) and PRince .int/.sur files
    "prdbv3_rotamer_dir": "data/PRDBv3_rotamer",
    # rota_assign checkout providing macromol_torsion
    "rota_assign_dir": "../rota_assign",
    # DSSP executable
    "dssp_bin": "dssp",
    # Non-redundant 288-complex PRDBv3 set (calculate_288_prdbv3_protein_torsions.py)
    "nr288_dir": "data/nr_288_PRDBv3.0",
    # PRince executable
    "prince_bin": "prince",
    # protein-protein project (PPI libraries in derive_all_updated_rotamer_libraries.py)
    "protein_protein_dir": "../protein-protein",
}


def _load():
    cfg = dict(DEFAULTS)
    local = ROOT_DIR / "config" / "paths.json"
    if local.exists():
        cfg.update(json.loads(local.read_text()))
    return cfg


def get_path(key: str) -> Path:
    """
    Configured location for `key`. Relative paths resolve against the repository root; an
    executable given as a bare name (e.g. "dssp") is left as is so it is looked up on PATH.
    """
    value = _load()[key]
    p = Path(value).expanduser()
    if p.is_absolute() or (key.endswith("_bin") and "/" not in value):
        return p
    return ROOT_DIR / p
