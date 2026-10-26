#!/usr/bin/env python3
"""
calculate_187_prdbv3_protein_torsions.py

Bound and unbound protein torsion angles of the 187 PRDBv3 protein-RNA complexes that have an
unbound protein structure. Same calculators as calculate_288_prdbv3_protein_torsions.py
(rota_assign macromol_torsion), plus bound/unbound residue pairing.

Inputs (paths in utils/paths.py / config/paths.json):
  - input_files/PRDBv3_pdbList_chains.json             complex list (C_PDB, U_pro_PDB)
  - <prdbv3_rotamer_dir>/<C>_<U>_C.pdb / _U.pdb        equal-residue bound / unbound protein
  - <prdbv3_rotamer_dir>/<C>P.int, <C>P.sur            PRince interface / surface atoms

Per residue (one row per bound residue with N, CA and C):
  PHI, PSI      : C(i-1)-N-CA-C and N-CA-C-N(i+1); empty at chain ends and breaks (C-N > 2.0 A)
  OMEGA         : CA(i)-C(i)-N(i+1)-CA(i+1), the peptide bond after residue i (IUPAC); a cis
                  X-Pro bond is therefore reported on the residue before PRO
  CHI1-CHI5     : IUPAC side-chain torsions (CHI5: ARG CD-NE-CZ-NH1); empty when an atom is missing
  SS            : 3-state H / S / L from DSSP with the Rost & Sander (1993) reduction used in the
                  original assign_dssp.py: H, G -> H; E -> S; I, T, S, coil -> L; an isolated
                  bridge B followed by coil -> S S, and B coil B -> L L L
  DSSP          : the raw 8-state DSSP code ('-' for coil)
  SASA          : I interface (in PRince .int), N non-interface surface (in .sur, not .int),
                  C core (in neither)
  U_*           : the paired unbound residue. Unbound chains are often relabelled (5X6B: IJEF vs
                  ABCD) or renumbered (2ZZM), so residues are paired by chain order and aligned
                  residue-name sequence (macromol_torsion.pairing), never by label. U_LABEL gives
                  the unbound residue; all U_* are empty when there is no partner.

Missing values are empty in the CSV and null in the JSON, never 0.0.

Outputs (output_files/torsion_dataset/):
  PRDBv3_187_bound_unbound_protein_torsions.csv / .json
  PRDBv3_187_bound_unbound_pairing_log.csv              per-complex pairing statistics

Usage:
    python script/calculate_187_prdbv3_protein_torsions.py
"""

import json
import logging
import math
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.paths import get_path

ROTA_ASSIGN_DIR = get_path("rota_assign_dir")
if str(ROTA_ASSIGN_DIR) not in sys.path:
    sys.path.insert(0, str(ROTA_ASSIGN_DIR))

from macromol_torsion.dssp import DsspAnalyzer
from macromol_torsion.pairing import pair_equal_residues
from macromol_torsion.protein_backbone import ProteinBackboneCalculator
from macromol_torsion.protein_sidechain import ProteinSidechainCalculator
from macromol_torsion.structure_reader import StructureReader

STRUCTURE_DIR = get_path("prdbv3_rotamer_dir")
COMPLEX_LIST = ROOT_DIR / "input_files" / "PRDBv3_pdbList_chains.json"
OUTPUT_DIR = ROOT_DIR / "output_files" / "torsion_dataset"
OUTPUT_STEM = "PRDBv3_187_bound_unbound_protein_torsions"
PAIRING_LOG = OUTPUT_DIR / "PRDBv3_187_bound_unbound_pairing_log.csv"

NAN = float("nan")
ANGLES = ["PHI", "PSI", "OMEGA", "CHI1", "CHI2", "CHI3", "CHI4", "CHI5"]


# ----------------------------------------------------------------------------------------------
# Secondary structure
# ----------------------------------------------------------------------------------------------
def three_state(codes):
    """
    Rost & Sander (1993) 3-state reduction as in the original assign_dssp.corrected_assignment:
    H, G -> H; E -> S; I, '-', S, T -> L; 'B' followed by coil -> 'S S', or 'L L L' when a second
    'B' follows ('B - B'); any other B -> L.
    """
    out, i, n = [], 0, len(codes)
    while i < n:
        c = codes[i]
        if c in ("H", "G"):
            out.append("H")
        elif c == "E":
            out.append("S")
        elif c == "B" and i + 1 < n and codes[i + 1] == "-":
            if i + 2 < n and codes[i + 2] == "B":
                out += ["L", "L", "L"]
                i += 3
            else:
                out += ["S", "S"]
                i += 2
            continue
        else:
            out.append("L")
        i += 1
    return out


def secondary_structure(dssp, structure_file):
    """{(chain, residue number with insertion code): (3-state, raw DSSP code)}."""
    lines = dssp.parse_dssp(dssp.run_dssp(structure_file))
    codes = ["-" if line[16] == " " else line[16] for line in lines]
    states = three_state(codes)
    # Keyed by chain and number, not residue name: DSSP writes disulfide-bonded CYS in lower case
    return {(line[11], line[5:10].strip() + line[10].strip()): (ss, raw)
            for line, ss, raw in zip(lines, states, codes)}


# ----------------------------------------------------------------------------------------------
# Interface / surface
# ----------------------------------------------------------------------------------------------
def prince_residues(path):
    """(chain, residue number with insertion code) of every atom in a PRince .int / .sur file."""
    keys = set()
    if path is None or not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                keys.add((line[21].strip(), line[22:26].strip() + line[26].strip()))
    return keys


def find_prince_file(bound, unbound, ext):
    for name in (f"{bound}P.{ext}", f"{bound}C.{ext}", f"{bound}.{ext}", f"{bound}_{unbound}_C.{ext}"):
        path = STRUCTURE_DIR / name
        if path.exists():
            return path
    return None


# ----------------------------------------------------------------------------------------------
# Torsions of one structure
# ----------------------------------------------------------------------------------------------
def residue_key(label):
    """(chain, residue number with insertion code) of a label such as 'LYS A   72' or 'GLY B  52A'."""
    parts = label.split()
    return parts[1], parts[2]


def structure_torsions(path, bb_calc, sc_calc, dssp):
    """Ordered residue labels and {label: {angle / SS / DSSP: value}} of one structure."""
    atoms = StructureReader.parse_structure(path)
    phi, psi, omega, _, _, labels = bb_calc.calculate_dihedrals(atoms)
    chis = sc_calc.calculate_all_chis(atoms)
    ss = secondary_structure(dssp, path)
    values = {}
    for i, lbl in enumerate(labels):
        ss_3, ss_raw = ss.get(residue_key(lbl), (None, None))
        values[lbl] = {"PHI": phi[i], "PSI": psi[i], "OMEGA": omega[i],
                       **{f"CHI{k}": chis[f"CHI{k}"].get(lbl, NAN) for k in range(1, 6)},
                       "SS": ss_3, "DSSP": ss_raw}
    return labels, values


def calculate_dataset():
    complexes = json.loads(COMPLEX_LIST.read_text())
    bb_calc, sc_calc = ProteinBackboneCalculator(), ProteinSidechainCalculator()
    dssp = DsspAnalyzer(dssp_binary=str(get_path("dssp_bin")))

    records, pairing_log = [], []
    for item in complexes:
        bound = item["C_PDB"].strip()
        unbound = str(item.get("U_pro_PDB") or "").replace("*", "").strip()
        c_file = STRUCTURE_DIR / f"{bound}_{unbound}_C.pdb"
        u_file = STRUCTURE_DIR / f"{bound}_{unbound}_U.pdb"
        if not unbound or not c_file.exists() or not u_file.exists():
            logging.info(f"{bound}: no unbound structure pair, skipped")
            continue

        b_labels, b_vals = structure_torsions(c_file, bb_calc, sc_calc, dssp)
        u_labels, u_vals = structure_torsions(u_file, bb_calc, sc_calc, dssp)
        try:
            pair_map, stats = pair_equal_residues(b_labels, u_labels)
        except ValueError as e:
            logging.warning(f"{bound}_{unbound}: {e}; unbound values left empty")
            pair_map, stats = {}, {"paired": 0, "name_mismatch": 0, "unpaired_bound": len(b_labels),
                                   "chain_length_mismatch": 0}
        pairing_log.append({"PDB": bound, "U_PDB": unbound, "bound_residues": len(b_labels),
                            "unbound_residues": len(u_labels), **stats})

        interface = prince_residues(find_prince_file(bound, unbound, "int"))
        surface = prince_residues(find_prince_file(bound, unbound, "sur"))

        for lbl in b_labels:
            key = residue_key(lbl)
            u_lbl = pair_map.get(lbl)
            b, u = b_vals[lbl], u_vals.get(u_lbl, {})
            rec = {"PDB": bound, "U_PDB": unbound, "LABEL": lbl, "U_LABEL": u_lbl}
            rec.update({f"B_{a}": b[a] for a in ANGLES + ["SS", "DSSP"]})
            rec.update({f"U_{a}": u.get(a, NAN if a in ANGLES else None) for a in ANGLES + ["SS", "DSSP"]})
            rec["SASA"] = "I" if key in interface else ("N" if key in surface else "C")
            records.append(rec)

    df = pd.DataFrame(records)
    df[[f"{s}_{a}" for s in "BU" for a in ANGLES]] = df[[f"{s}_{a}" for s in "BU" for a in ANGLES]].round(2)
    return df, pd.DataFrame(pairing_log)


def write_outputs(df, pairing):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_file, json_file = OUTPUT_DIR / f"{OUTPUT_STEM}.csv", OUTPUT_DIR / f"{OUTPUT_STEM}.json"
    df.to_csv(csv_file, index=False)
    records = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}
               for rec in df.to_dict(orient="records")]
    with open(json_file, "w") as f:
        json.dump(records, f, indent=1)
    pairing.to_csv(PAIRING_LOG, index=False)

    paired = df["U_LABEL"].notna()
    logging.info(f"{len(df)} residues in {df['PDB'].nunique()} complexes; "
                 f"{int(paired.sum())} paired with an unbound residue, {int((~paired).sum())} unpaired")
    logging.info(f"Regions: {df['SASA'].value_counts().to_dict()}; "
                 f"bound SS: {df['B_SS'].value_counts(dropna=False).to_dict()}")
    logging.info(f"Wrote {csv_file}, {json_file} and {PAIRING_LOG}")


if __name__ == "__main__":
    write_outputs(*calculate_dataset())
