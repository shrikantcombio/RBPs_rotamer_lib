#!/usr/bin/env python3
"""
deltaG_calculation.py

Calculates backbone (Phi, Psi) and side-chain (Chi1..Chi5) torsion potential energy (E_U, E_B, DeltaE = E_U - E_B)
for bound and unbound protein structures using standardized AMBER force-field parameters (parm14SB / ff99SB).

Generates:
1. PDB-level total and average energy JSON and CSV files across 187 structures at Interface (I) and Non-Interface (N).
2. Amino Acid-level total and average energy JSON and CSV files across 17 amino acid types.

Usage:
    python script/deltaG_calculation.py [options]
"""

import argparse
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUT_DIR = ROOT_DIR / "input_files"
OUTPUT_DIR = ROOT_DIR / "output_files"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gas constant * Temperature (RT in kcal/mol at ~298 K)
RT = 0.59225621

# Amino acid groups
AA_ONLY_CHI1 = ["CYS", "SER", "THR", "VAL"]
AA_AROMATIC = ["HIS", "PHE", "TRP", "TYR"]
EXCLUDED_RESIDUES = ["GLY", "ALA", "PRO"]

# Standardized AMBER force field parameters (parm14SB / ff99SB)
# Format: {angle_name: [barrier_divider, K_theta (kcal/mol), phase_shift (deg), periodicity_m]}
DIHEDRAL_PARAMS = {
    "ARG": {
        "CHI1": [9, 1.400, 0.0, 3.0],
        "CHI2": [9, 1.400, 0.0, 3.0],
        "CHI3": [1, 0.180, 0.0, 3.0],
        "CHI4": [1, 0.400, 180.0, 2.0],
        "CHI5": [1, 3.500, 180.0, 2.0]
    },
    "ASN": {
        "CHI1": [1, 0.033, 0.0, 3.0],
        "CHI2": [1, 0.301, 180.0, 3.0]
    },
    "ASP": {
        "CHI1": [1, 0.058, 0.0, 3.0],
        "CHI2": [1, 0.000, 0.0, 3.0]
    },
    "CYS": {
        "CHI1": [1, 0.251, 0.0, 3.0]
    },
    "GLN": {
        "CHI1": [1, 0.033, 0.0, 3.0],
        "CHI2": [1, 0.412, 180.0, 3.0],
        "CHI3": [1, 0.250, 180.0, 2.0]
    },
    "GLU": {
        "CHI1": [1, 0.144, 0.0, 3.0],
        "CHI2": [1, 0.608, 180.0, 3.0],
        "CHI3": [1, 0.250, 180.0, 2.0]
    },
    "HIS": {
        "CHI1": [1, 0.219, 0.0, 3.0],
        "CHI2": [1, 0.122, 180.0, 3.0]
    },
    "ILE": {
        "CHI1": [1, 0.113, 0.0, 3.0],
        "CHI2": [1, 0.107, 0.0, 3.0]
    },
    "LEU": {
        "CHI1": [1, 0.144, 0.0, 3.0],
        "CHI2": [1, 0.164, 0.0, 3.0]
    },
    "LYS": {
        "CHI1": [9, 1.400, 0.0, 3.0],
        "CHI2": [1, 0.180, 0.0, 3.0],
        "CHI3": [1, 0.180, 0.0, 3.0],
        "CHI4": [1, 0.156, 0.0, 3.0]
    },
    "MET": {
        "CHI1": [1, 0.180, 0.0, 3.0],
        "CHI2": [1, 0.016, 0.0, 3.0],
        "CHI3": [1, 0.380, 0.0, 3.0]
    },
    "PHE": {
        "CHI1": [1, 5.728, 180.0, 3.0],
        "CHI2": [1, 0.126, -21.2, 4.0]
    },
    "SER": {
        "CHI1": [1, 0.401, 0.0, 3.0]
    },
    "THR": {
        "CHI1": [1, 0.315, 0.0, 3.0]
    },
    "TRP": {
        "CHI1": [1, 2.475, 0.0, 3.0],
        "CHI2": [1, 1.029, 0.0, 3.0]
    },
    "TYR": {
        "CHI1": [1, 5.728, 180.0, 3.0],
        "CHI2": [1, 0.126, -21.2, 4.0]
    },
    "VAL": {
        "CHI1": [1, 0.148, 0.0, 3.0]
    }
}


def torsion_potential_charmm(theta_deg, K_theta, m, delta_deg):
    """
    Calculate torsion potential energy using CHARMM formulation:
    E = (K_theta / 2) * (1 + cos(m * theta - delta))
    """
    theta_rad = np.radians(theta_deg)
    delta_rad = np.radians(delta_deg)
    return (K_theta / 2.0) * (1.0 + np.cos(m * theta_rad - delta_rad))


def torsion_potential_amber(theta_deg, K_theta, m, delta_deg):
    """
    Calculate torsion potential energy using standard AMBER formulation:
    E = K_theta * (1 + cos(m * theta + delta))
    """
    theta_rad = np.radians(theta_deg)
    delta_rad = np.radians(delta_deg)
    return K_theta * (1.0 + np.cos(m * theta_rad + delta_rad))


def calculate_residue_torsion_energy(row, force_field="amber", include_backbone=True):
    """
    Calculates unbound torsional energy (E_U), bound torsional energy (E_B),
    and energy difference (DeltaE = E_U - E_B) for a single residue row containing Phi, Psi, Chi1..Chi5.
    """
    K_phi = 0.8
    K_psi = 0.4
    m_phi = m_psi = 3.0
    delta_phi = 180.0
    delta_psi = 0.0

    pot_fn = torsion_potential_charmm if str(force_field).lower() == "charmm" else torsion_potential_amber

    lbl = str(row.get("LABEL", "")).strip()
    res_name = lbl.split()[0] if lbl else str(row.get("AA", "GLY")).strip().upper()

    e_u_bb = 0.0
    e_b_bb = 0.0

    if include_backbone and res_name not in ["GLY", "PRO"]:
        u_phi = float(row.get("U_PHI", row.get("B_PHI", 0.0)))
        u_psi = float(row.get("U_PSI", row.get("B_PSI", 0.0)))
        b_phi = float(row.get("B_PHI", 0.0))
        b_psi = float(row.get("B_PSI", 0.0))

        e_u_bb = pot_fn(u_phi, K_phi, m_phi, delta_phi) + pot_fn(u_psi, K_psi, m_psi, delta_psi)
        e_b_bb = pot_fn(b_phi, K_phi, m_phi, delta_phi) + pot_fn(b_psi, K_psi, m_psi, delta_psi)

    e_u_sc = 0.0
    e_b_sc = 0.0

    if res_name in DIHEDRAL_PARAMS:
        params = DIHEDRAL_PARAMS[res_name]
        for chi_name in ["CHI1", "CHI2", "CHI3", "CHI4", "CHI5"]:
            if chi_name in params:
                u_col = f"U_{chi_name}"
                b_col = f"B_{chi_name}"

                u_ang = float(row.get(u_col, 0.0))
                b_ang = float(row.get(b_col, 0.0))

                k_val = params[chi_name][1] / params[chi_name][0]
                d_val = params[chi_name][2]
                m_val = params[chi_name][3]

                e_u_sc += pot_fn(u_ang, k_val, m_val, d_val)
                e_b_sc += pot_fn(b_ang, k_val, m_val, d_val)

    e_u = e_u_bb + e_u_sc
    e_b = e_b_bb + e_b_sc

    return {
        "E_U": round(e_u, 6),
        "E_B": round(e_b, 6),
        "DeltaE": round(e_u - e_b, 6),
        "E_U_sc": round(e_u_sc, 6),
        "E_B_sc": round(e_b_sc, 6),
        "DeltaE_sc": round(e_u_sc - e_b_sc, 6),
    }


def calculate_dataset_pdb_energies(df_torsion, force_field="amber"):
    """
    Calculate PDB-level total and average torsion potential energy for interface (I) and non-interface (N).
    """
    logging.info(f"Computing PDB-level torsion energies across dataset (force_field='{force_field}')...")
    pdb_col = "PDB" if "PDB" in df_torsion.columns else "PDB_ID"
    pdbs = df_torsion[pdb_col].unique()
    records = []

    for pdb in pdbs:
        pdb_str = str(pdb).strip()
        df_pdb = df_torsion[df_torsion[pdb_col] == pdb]

        # Interface
        df_int = df_pdb[df_pdb["SASA"].astype(str).str.strip() == "I"]
        cnt_int = len(df_int)
        e_u_int = sum(calculate_residue_torsion_energy(r, force_field=force_field)["E_U"] for _, r in df_int.iterrows())
        e_b_int = sum(calculate_residue_torsion_energy(r, force_field=force_field)["E_B"] for _, r in df_int.iterrows())

        # Non-interface
        df_nint = df_pdb[df_pdb["SASA"].astype(str).str.strip() == "N"]
        cnt_nint = len(df_nint)
        e_u_nint = sum(calculate_residue_torsion_energy(r, force_field=force_field)["E_U"] for _, r in df_nint.iterrows())
        e_b_nint = sum(calculate_residue_torsion_energy(r, force_field=force_field)["E_B"] for _, r in df_nint.iterrows())

        records.append({
            "PDB_ID": pdb_str,
            "IU_DELG": round(e_u_int, 6),
            "IB_DELG": round(e_b_int, 6),
            "IU_AVG_DELG": round(e_u_int / cnt_int, 6) if cnt_int > 0 else 0.0,
            "IB_AVG_DELG": round(e_b_int / cnt_int, 6) if cnt_int > 0 else 0.0,
            "ICOUNT": cnt_int,
            "NU_DELG": round(e_u_nint, 6),
            "NB_DELG": round(e_b_nint, 6),
            "NU_AVG_DELG": round(e_u_nint / cnt_nint, 6) if cnt_nint > 0 else 0.0,
            "NB_AVG_DELG": round(e_b_nint / cnt_nint, 6) if cnt_nint > 0 else 0.0,
            "NCOUNT": cnt_nint
        })

    return pd.DataFrame(records)


def calculate_dataset_aa_energies(df_torsion, force_field="amber"):
    """
    Calculate Amino Acid-level total and average torsion potential energy for interface (I) and non-interface (N).
    """
    logging.info(f"Computing Amino Acid-level torsion energies across dataset...")
    df_work = df_torsion.copy()
    if "AA" not in df_work.columns:
        df_work["AA"] = df_work["LABEL"].apply(lambda s: str(s).split()[0] if str(s).strip() else "GLY")

    aa_types = sorted(df_work["AA"].unique())
    records = []

    for aa in aa_types:
        if aa in EXCLUDED_RESIDUES or aa not in DIHEDRAL_PARAMS:
            continue

        df_aa = df_work[df_work["AA"] == aa]

        # Interface
        df_int = df_aa[df_aa["SASA"].astype(str).str.strip() == "I"]
        cnt_int = len(df_int)
        e_u_int = sum(calculate_residue_torsion_energy(r, force_field=force_field)["E_U"] for _, r in df_int.iterrows())
        e_b_int = sum(calculate_residue_torsion_energy(r, force_field=force_field)["E_B"] for _, r in df_int.iterrows())

        # Non-interface
        df_nint = df_aa[df_aa["SASA"].astype(str).str.strip() == "N"]
        cnt_nint = len(df_nint)
        e_u_nint = sum(calculate_residue_torsion_energy(r, force_field=force_field)["E_U"] for _, r in df_nint.iterrows())
        e_b_nint = sum(calculate_residue_torsion_energy(r, force_field=force_field)["E_B"] for _, r in df_nint.iterrows())

        records.append({
            "AA": str(aa),
            "IU_DELG": round(e_u_int, 6),
            "IB_DELG": round(e_b_int, 6),
            "IU_AVG_DELG": round(e_u_int / cnt_int, 6) if cnt_int > 0 else 0.0,
            "IB_AVG_DELG": round(e_b_int / cnt_int, 6) if cnt_int > 0 else 0.0,
            "ICOUNT": cnt_int,
            "NU_DELG": round(e_u_nint, 6),
            "NB_DELG": round(e_b_nint, 6),
            "NU_AVG_DELG": round(e_u_nint / cnt_nint, 6) if cnt_nint > 0 else 0.0,
            "NB_AVG_DELG": round(e_b_nint / cnt_nint, 6) if cnt_nint > 0 else 0.0,
            "NCOUNT": cnt_nint
        })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Calculate torsion potential energy (DeltaG) for bound and unbound protein structures.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=INPUT_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187.tsv",
        help="Path to backbone & sidechain torsion angle dataset file.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for energy files.")
    parser.add_argument("--force-field", type=str, default="amber", choices=["charmm", "amber"], help="Force field energy function formulation.")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_file.exists():
        logging.error(f"Input file not found: {args.input_file}")
        return

    df_torsion = pd.read_csv(args.input_file, sep="\t", keep_default_na=False)
    logging.info(f"Loaded dataset with {len(df_torsion)} residue entries from {args.input_file}")

    # 1. PDB Level Energies
    df_pdb_energy = calculate_dataset_pdb_energies(df_torsion, force_field=args.force_field)
    pdb_csv = args.output_dir / "torsion_potential_energy_summary_187.csv"
    pdb_json = args.output_dir / "torsion_potential_energy_summary_187.json"
    
    # Save CSV and JSON
    df_pdb_energy.to_csv(pdb_csv, index=False)
    with open(pdb_json, "w") as f:
        json.dump(df_pdb_energy.to_dict(orient="records"), f, indent=2)

    logging.info(f"Saved PDB-level energy summary: {pdb_csv} and {pdb_json}")

    # 2. Amino Acid Level Energies
    df_aa_energy = calculate_dataset_aa_energies(df_torsion, force_field=args.force_field)
    aa_csv = args.output_dir / "amino_acid_torsion_energy_summary_187.csv"
    aa_json = args.output_dir / "amino_acid_torsion_energy_summary_187.json"
    
    df_aa_energy.to_csv(aa_csv, index=False)
    with open(aa_json, "w") as f:
        json.dump(df_aa_energy.to_dict(orient="records"), f, indent=2)

    logging.info(f"Saved Amino Acid-level energy summary: {aa_csv} and {aa_json}")

    logging.info("DeltaG torsion energy calculation completed cleanly (JSON/CSV outputs updated, no .dat files created).")


if __name__ == "__main__":
    main()
