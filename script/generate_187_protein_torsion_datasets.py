#!/usr/bin/env python3
"""
generate_187_protein_torsion_datasets.py

Calculates complete protein torsion angle datasets (B_PHI, B_PSI, B_OMEGA, B_CHI1-5, B_SS,
U_PHI, U_PSI, U_OMEGA, U_CHI1-5, U_SS, SASA) for all 187 PRDBv3 protein-RNA complexes.

SASA Classification:
  - 'I': Interface residues (buried upon complexation, from PRince .int)
  - 'N': Non-interface surface residues (surface exposed outside binding interface, from PRince .sur)
  - 'C': Core buried residues (interior residues without surface exposure)

Bound/unbound pairing:
  Residues of <C_PDB>_<U_PDB>_C.pdb are paired with <C_PDB>_<U_PDB>_U.pdb by chain order and
  residue sequence (macromol_torsion.pairing), not by residue label: unbound chains are usually
  relabelled (5X6B: IJEF vs ABCD) or renumbered (2ZZM).

Missing values:
  Angles that cannot be computed (missing atoms, chain termini/breaks, no unbound partner) are
  written as empty (NaN), never 0.0. U_SS is empty when the residue has no unbound partner.

Outputs (paths in utils/paths.py / config/paths.json):
  - output_files/ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187.{csv,tsv,json}
  - input_files/ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv (comma-separated)
  - output_files/ALL_AA_..._187_pairing_log.csv (per-complex pairing statistics)
"""

import os
import sys
import json
import math
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def records_with_null(df):
    """DataFrame rows as dicts with NaN replaced by None, so json.dump writes valid JSON (null)."""
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()}
            for rec in df.to_dict(orient="records")]

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.paths import get_path

ROTA_ASSIGN_DIR = get_path("rota_assign_dir")
if str(ROTA_ASSIGN_DIR) not in sys.path:
    sys.path.insert(0, str(ROTA_ASSIGN_DIR))

from macromol_torsion.protein_backbone import ProteinBackboneCalculator
from macromol_torsion.protein_sidechain import ProteinSidechainCalculator
from macromol_torsion.dssp import DsspAnalyzer
from macromol_torsion.structure_reader import StructureReader
from macromol_torsion.pairing import pair_equal_residues

NAN = float("nan")
CHI_KEYS = ("CHI1", "CHI2", "CHI3", "CHI4", "CHI5")

DATASET_DIR = get_path("prdbv3_rotamer_dir")
JSON_PATH = ROOT_DIR / "input_files" / "PRDBv3_pdbList_chains.json"
OUTPUT_DIR = ROOT_DIR / "output_files"
INPUT_FILES_DIR = ROOT_DIR / "input_files"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_FILES_DIR.mkdir(parents=True, exist_ok=True)


def parse_res_tuples(fpath):
    """Parses PDB/INT/SUR file to extract set of (chain, resSeq, resName) and (chain, resSeq) tuples."""
    f_p = Path(fpath)
    if not f_p.exists():
        return set()

    tuples = set()
    with open(f_p, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain = line[21:22].strip()
                seq = line[22:26].strip()
                res_name = line[17:20].strip()
                tuples.add((chain, seq, res_name))
                tuples.add((chain, seq))
    return tuples


def find_prince_file(c_pdb, u_pdb, ext):
    candidates = [
        DATASET_DIR / f"{c_pdb}P.{ext}",
        DATASET_DIR / f"{c_pdb}C.{ext}",
        DATASET_DIR / f"{c_pdb}.{ext}",
        DATASET_DIR / f"{c_pdb}_{u_pdb}_C.{ext}"
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def calculate_187_dataset():
    logging.info(f"Loading 187 PRDBv3 complex list: {JSON_PATH}")
    with open(JSON_PATH, "r") as f:
        json_items = json.load(f)

    sc_calc = ProteinSidechainCalculator()
    bb_calc = ProteinBackboneCalculator()
    dssp = DsspAnalyzer(dssp_binary=str(get_path("dssp_bin")))

    all_records = []
    pairing_log = []
    processed_count = 0

    for idx, item in enumerate(json_items):
        bound_pdb = item["C_PDB"].strip()
        unbound_pdb = str(item.get("U_pro_PDB") or "").replace("*", "").strip()
        pro_chain = item["C_pro_chain"].strip()

        if not unbound_pdb:
            continue

        c_file = DATASET_DIR / f"{bound_pdb}_{unbound_pdb}_C.pdb"
        u_file = DATASET_DIR / f"{bound_pdb}_{unbound_pdb}_U.pdb"

        if not c_file.exists() or not u_file.exists():
            continue

        processed_count += 1

        # 1. Interface & Surface residue lookup from PRince .int and .sur files
        int_file = find_prince_file(bound_pdb, unbound_pdb, "int")
        sur_file = find_prince_file(bound_pdb, unbound_pdb, "sur")

        int_tuples = parse_res_tuples(int_file)
        sur_tuples = parse_res_tuples(sur_file)

        # 2. Parse Atoms using clean parser
        b_atoms = StructureReader.parse_structure(c_file)
        u_atoms = StructureReader.parse_structure(u_file)

        # 3. DSSP Secondary Structure
        b_ss_map = dssp.get_secondary_structure(c_file)
        u_ss_map = dssp.get_secondary_structure(u_file)

        # 4. Backbone Dihedrals
        b_phi, b_psi, b_omega, _, _, b_labels = bb_calc.calculate_dihedrals(b_atoms)
        u_phi, u_psi, u_omega, _, _, u_labels = bb_calc.calculate_dihedrals(u_atoms)

        # 5. Sidechain Dihedrals
        b_chis = sc_calc.calculate_all_chis(b_atoms)
        u_chis = sc_calc.calculate_all_chis(u_atoms)

        # Build maps per residue label
        b_bb_map = {}
        for i, lbl in enumerate(b_labels):
            parts = lbl.split()
            if len(parts) >= 3:
                res_name, chain, seq = parts[0], parts[1], parts[2]
                b_bb_map[lbl] = {
                    "RESNAME": res_name, "CHAIN": chain, "SEQ": seq,
                    "PHI": b_phi[i], "PSI": b_psi[i], "OMEGA": b_omega[i],
                    "SS": b_ss_map.get(lbl, "L")
                }

        u_bb_map = {}
        for i, lbl in enumerate(u_labels):
            parts = lbl.split()
            if len(parts) >= 3:
                u_bb_map[lbl] = {
                    "PHI": u_phi[i], "PSI": u_psi[i], "OMEGA": u_omega[i],
                    "SS": u_ss_map.get(lbl, "L")
                }

        # Pair bound and unbound residues by chain order and sequence (labels differ)
        try:
            pair_map, pair_stats = pair_equal_residues(b_labels, u_labels)
        except ValueError as e:
            logging.warning(f"{bound_pdb}_{unbound_pdb}: {e}; unbound values left empty")
            pair_map, pair_stats = {}, {"paired": 0, "name_mismatch": 0,
                                         "unpaired_bound": len(b_labels), "chain_length_mismatch": 0}
        pairing_log.append({"PDB": bound_pdb, "U_PDB": unbound_pdb, "bound_residues": len(b_labels),
                            "unbound_residues": len(u_labels), **pair_stats})

        # 6. Build combined residue records
        for lbl, b_info in b_bb_map.items():
            res_name = b_info["RESNAME"]
            chain = b_info["CHAIN"]
            seq = b_info["SEQ"]

            # Bound chi angles lookup (NaN when the chi cannot be computed)
            b_c1, b_c2, b_c3, b_c4, b_c5 = (b_chis[k].get(lbl, NAN) for k in CHI_KEYS)

            # Unbound values via the paired unbound residue label
            u_lbl = pair_map.get(lbl)
            u_info = u_bb_map.get(u_lbl) if u_lbl else None
            if u_info:
                u_phi_val, u_psi_val, u_omega_val = u_info["PHI"], u_info["PSI"], u_info["OMEGA"]
                u_ss_val = u_info["SS"]
                u_c1, u_c2, u_c3, u_c4, u_c5 = (u_chis[k].get(u_lbl, NAN) for k in CHI_KEYS)
            else:
                u_phi_val = u_psi_val = u_omega_val = NAN
                u_ss_val = None
                u_c1 = u_c2 = u_c3 = u_c4 = u_c5 = NAN

            # SASA classification logic:
            # - 'I': Interface (in .int)
            # - 'N': Non-interface Surface (in .sur but not in .int)
            # - 'C': Core buried (not in .sur)
            is_int = (chain, seq, res_name) in int_tuples or (chain, seq) in int_tuples
            is_sur = (chain, seq, res_name) in sur_tuples or (chain, seq) in sur_tuples

            if is_int:
                sasa_val = "I"
            elif is_sur:
                sasa_val = "N"
            else:
                sasa_val = "C"

            record = {
                "PDB": bound_pdb,
                "LABEL": lbl,
                "B_PHI": b_info["PHI"],
                "B_PSI": b_info["PSI"],
                "B_OMEGA": b_info["OMEGA"],
                "B_CHI1": b_c1,
                "B_CHI2": b_c2,
                "B_CHI3": b_c3,
                "B_CHI4": b_c4,
                "B_CHI5": b_c5,
                "B_SS": b_info["SS"],
                "U_PHI": u_phi_val,
                "U_PSI": u_psi_val,
                "U_OMEGA": u_omega_val,
                "U_CHI1": u_c1,
                "U_CHI2": u_c2,
                "U_CHI3": u_c3,
                "U_CHI4": u_c4,
                "U_CHI5": u_c5,
                "U_SS": u_ss_val,
                "SASA": sasa_val,
            }
            all_records.append(record)

    df = pd.DataFrame(all_records)
    pairing_df = pd.DataFrame(pairing_log)
    pairing_df.to_csv(OUTPUT_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_pairing_log.csv", index=False)
    logging.info(f"Pairing: {int(pairing_df.paired.sum())} bound residues paired with an unbound residue, "
                 f"{int(pairing_df.unpaired_bound.sum())} unpaired, across {len(pairing_df)} complexes")
    logging.info(f"Successfully calculated 187 PRDBv3 protein torsion dataset! Total residues: {len(df)}")

    # Export CSV, TSV, and JSON files to output_files/ and input_files/
    csv_file = OUTPUT_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187.csv"
    json_file = OUTPUT_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187.json"
    tsv_file = OUTPUT_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187.tsv"

    df.to_csv(csv_file, index=False)
    df.to_csv(tsv_file, sep="\t", index=False)

    # NaN is not valid JSON; write null instead
    records_json = records_with_null(df)
    with open(json_file, "w") as f_json:
        json.dump(records_json, f_json, indent=2)

    # Sync clean TSV to input_files for rotamer lib generation
    clean_tsv_input = INPUT_FILES_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"
    df.to_csv(clean_tsv_input, index=False)

    logging.info("==================================================")
    logging.info("187 PROTEIN TORSION DATASET GENERATED SUCCESSFULLY!")
    logging.info(f"  CSV Output:  {csv_file} ({os.path.getsize(csv_file)/1e6:.2f} MB)")
    logging.info(f"  TSV Output:  {tsv_file} ({os.path.getsize(tsv_file)/1e6:.2f} MB)")
    logging.info(f"  JSON Output: {json_file} ({os.path.getsize(json_file)/1e6:.2f} MB)")
    logging.info("==================================================")


if __name__ == "__main__":
    calculate_187_dataset()
