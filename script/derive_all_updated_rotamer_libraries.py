#!/usr/bin/env python3
"""
derive_all_updated_rotamer_libraries.py

Re-derives all Backbone-Dependent (BBD) and Backbone-Independent (BBI) Rotamer Libraries
for Protein-RNA (RBPs) and Protein-Protein (PPI) complexes using verified datasets.

Outputs clean, unified CSV, TSV, and JSON files for all amino acids (without individual .lib files).

Output Directories:
  - RBPs BBD: RBPs_rotamer_lib/RBPs_BBD_rotamer_lib/
  - RBPs BBI: RBPs_rotamer_lib/RBPs_BBI_rotamer_lib/
  - PPI Unbound 1: protein-protein/library/unbound1/
  - PPI Unbound 2: protein-protein/library/unbound2/
"""

import os
import sys
import json
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent # /home/labuser/Projects/PhD_projects/RBPs_rotamer_lib
BASE_DIR = ROOT_DIR.parent   # /home/labuser/Projects/PhD_projects

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backbone_dependent_rotamer_lib import BackboneDependentRotamerLibrary
from backbone_independent_rotamer_lib import BackboneIndependentRotamerLibrary

# Dataset paths
RBP_CLEAN_TSV = ROOT_DIR / "input_files" / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"
PPI_UNBOUND1_CSV = BASE_DIR / "protein-protein" / "output_files" / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_PROTEIN_PROTEIN_UNBOUND1_257.csv"
PPI_UNBOUND2_CSV = BASE_DIR / "protein-protein" / "output_files" / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_PROTEIN_PROTEIN_UNBOUND2_257.csv"

# Output directory paths
RBP_BBD_OUT_DIR = ROOT_DIR / "RBPs_BBD_rotamer_lib"
RBP_BBI_OUT_DIR = ROOT_DIR / "RBPs_BBI_rotamer_lib"
PPI_UNBOUND1_OUT_DIR = BASE_DIR / "protein-protein" / "library" / "unbound1"
PPI_UNBOUND2_OUT_DIR = BASE_DIR / "protein-protein" / "library" / "unbound2"

for d in [RBP_BBD_OUT_DIR, RBP_BBI_OUT_DIR, PPI_UNBOUND1_OUT_DIR, PPI_UNBOUND2_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def export_unified_library_files(lib_dict, output_dir, prefix):
    """Exports rotamer library dict to unified CSV, TSV, and JSON files combining all amino acids."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for aa, r_df in lib_dict.items():
        all_dfs.append(r_df)

    if not all_dfs:
        logging.warning(f"No library data generated for {prefix}.")
        return

    df_combined = pd.concat(all_dfs, ignore_index=True)

    csv_path = out_dir / f"{prefix}.csv"
    tsv_path = out_dir / f"{prefix}.tsv"
    json_path = out_dir / f"{prefix}.json"

    df_combined.to_csv(csv_path, index=False)
    df_combined.to_csv(tsv_path, sep="\t", index=False)

    records = df_combined.to_dict(orient="records")
    with open(json_path, "w") as f_json:
        json.dump(records, f_json, indent=2)

    logging.info(f"Saved unified library dataset: {prefix} (CSV: {csv_path.name}, TSV: {tsv_path.name}, JSON: {json_path.name}) | Rows: {len(df_combined)}")


def derive_rbp_rotamer_libraries():
    logging.info("==================================================")
    logging.info("1. DERIVING PROTEIN-RNA (RBPs) ROTAMER LIBRARIES")
    logging.info(f"Loading verified RBP dataset: {RBP_CLEAN_TSV}")
    df_rbp = pd.read_csv(RBP_CLEAN_TSV, sep=",")

    bbd_calc = BackboneDependentRotamerLibrary(bb_sep=30)
    bbi_calc = BackboneIndependentRotamerLibrary()

    # BBD Libraries for RBPs: B_I (Interface), B_N (Non-Interface), B_S (Surface), B (Overall), U_I, U_N, U_S, U
    bbd_configs = [
        ("B", "I", "RBPs_I_bbd_B_rotamer_lib"),
        ("B", "N", "RBPs_N_bbd_B_rotamer_lib"),
        ("B", "S", "RBPs_S_bbd_B_rotamer_lib"),
        ("B", None, "RBPs_bbd_B_rotamer_lib"),
        ("U", "I", "RBPs_I_bbd_U_rotamer_lib"),
        ("U", "N", "RBPs_N_bbd_U_rotamer_lib"),
        ("U", "S", "RBPs_S_bbd_U_rotamer_lib"),
        ("U", None, "RBPs_bbd_U_rotamer_lib"),
    ]

    for state, sasa, prefix in bbd_configs:
        lib_dict = bbd_calc.generate_library(df_rbp, state=state, sasa_filter=sasa)
        export_unified_library_files(lib_dict, RBP_BBD_OUT_DIR, prefix)

    # BBI Libraries for RBPs
    bbi_configs = [
        ("B", "I", "RBPs_I_bbi_B_rotamer_lib"),
        ("B", "N", "RBPs_N_bbi_B_rotamer_lib"),
        ("B", "S", "RBPs_S_bbi_B_rotamer_lib"),
        ("B", None, "RBPs_bbi_B_rotamer_lib"),
        ("U", "I", "RBPs_I_bbi_U_rotamer_lib"),
        ("U", "N", "RBPs_N_bbi_U_rotamer_lib"),
        ("U", "S", "RBPs_S_bbi_U_rotamer_lib"),
        ("U", None, "RBPs_bbi_U_rotamer_lib"),
    ]

    for state, sasa, prefix in bbi_configs:
        lib_dict = bbi_calc.generate_library(df_rbp, state=state, sasa_filter=sasa)
        export_unified_library_files(lib_dict, RBP_BBI_OUT_DIR, prefix)


def derive_ppi_rotamer_libraries():
    logging.info("==================================================")
    logging.info("2. DERIVING PROTEIN-PROTEIN (PPI) ROTAMER LIBRARIES")

    bbd_calc = BackboneDependentRotamerLibrary(bb_sep=30)
    bbi_calc = BackboneIndependentRotamerLibrary()

    partners = [
        (1, PPI_UNBOUND1_CSV, PPI_UNBOUND1_OUT_DIR),
        (2, PPI_UNBOUND2_CSV, PPI_UNBOUND2_OUT_DIR),
    ]

    for p_num, p_csv, p_out_dir in partners:
        logging.info(f"Loading PPI Unbound Partner {p_num} dataset: {p_csv}")
        df_ppi = pd.read_csv(p_csv)

        # BBD Libraries
        bbd_configs = [
            ("B", "I", f"PPI_bbd_B_I_unbound{p_num}"),
            ("B", "N", f"PPI_bbd_B_N_unbound{p_num}"),
            ("B", None, f"PPI_bbd_B_S_unbound{p_num}"),
            ("U", "I", f"PPI_bbd_U_I_unbound{p_num}"),
            ("U", "N", f"PPI_bbd_U_N_unbound{p_num}"),
            ("U", None, f"PPI_bbd_U_S_unbound{p_num}"),
        ]

        bbd_dir = p_out_dir / "bbd"
        for state, sasa, prefix in bbd_configs:
            lib_dict = bbd_calc.generate_library(df_ppi, state=state, sasa_filter=sasa)
            export_unified_library_files(lib_dict, bbd_dir, prefix)

        # BBI Libraries
        bbi_configs = [
            ("B", "I", f"PPI_bbi_B_I_unbound{p_num}"),
            ("B", "N", f"PPI_bbi_B_N_unbound{p_num}"),
            ("B", None, f"PPI_bbi_B_S_unbound{p_num}"),
            ("U", "I", f"PPI_bbi_U_I_unbound{p_num}"),
            ("U", "N", f"PPI_bbi_U_N_unbound{p_num}"),
            ("U", None, f"PPI_bbi_U_S_unbound{p_num}"),
        ]

        bbi_dir = p_out_dir / "bbi"
        for state, sasa, prefix in bbi_configs:
            lib_dict = bbi_calc.generate_library(df_ppi, state=state, sasa_filter=sasa)
            export_unified_library_files(lib_dict, bbi_dir, prefix)


def main():
    derive_rbp_rotamer_libraries()
    derive_ppi_rotamer_libraries()
    logging.info("==================================================")
    logging.info("ALL ROTAMER LIBRARIES RE-DERIVED SUCCESSFULLY IN UNIFIED CSV, TSV, AND JSON FORMATS!")
    logging.info("==================================================")


if __name__ == "__main__":
    main()
