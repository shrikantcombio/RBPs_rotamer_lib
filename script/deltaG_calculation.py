#!/usr/bin/env python3
"""
deltaG_calculation.py

Calculates torsion potential energy (DeltaG) of side-chain and backbone
dihedral angles for bound and unbound protein structures using AMBER / CHARMM
force-field parameters.

Usage:
    python script/deltaG_calculation.py [options]
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "input_files"
RESULTS_DIR = ROOT_DIR / "output_files"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Gas constant * Temperature (RT in kcal/mol at ~298 K)
RT = 0.59225621

# Amino acid groups
AA_ONLY_CHI1 = ["CYS", "SER", "THR", "VAL"]
AA_AROMATIC = ["HIS", "PHE", "TRP", "TYR"]
EXCLUDED_RESIDUES = ["GLY", "ALA", "PRO"]

# Force field parameters (dihedral multiplicity, force constant Vn in kcal/mol, phase shift in deg, periodicity n)
DIHEDRAL_PARAMS = {
    "ARG": {"CHI1": [9, 1.400, 0.0, 3.0], "CHI2": [9, 1.400, 0.0, 3.0]},
    "ASN": {"CHI1": [1, 0.033, 0.0, 3.0], "CHI2": [1, 0.301, 180.0, 3.0]},
    "ASP": {"CHI1": [1, 0.058, 0.0, 3.0], "CHI2": [1, 0.000, 0.0, 3.0]},
    "CYS": {"CHI1": [1, 0.251, 0.0, 3.0]},
    "GLN": {"CHI1": [1, 0.033, 0.0, 3.0], "CHI2": [1, 0.412, 180.0, 3.0]},
    "GLU": {"CHI1": [1, 0.144, 0.0, 3.0], "CHI2": [1, 0.608, 180.0, 3.0]},
    "HIS": {"CHI1": [1, 0.219, 0.0, 3.0], "CHI2": [1, 0.122, 180.0, 3.0]},
    "ILE": {"CHI1": [1, 0.113, 0.0, 3.0], "CHI2": [1, 0.107, 0.0, 3.0]},
    "LEU": {"CHI1": [1, 0.144, 0.0, 3.0], "CHI2": [1, 0.164, 0.0, 3.0]},
    "LYS": {"CHI1": [9, 1.400, 0.0, 3.0], "CHI2": [1, 0.180, 0.0, 3.0]},
    "MET": {"CHI1": [1, 0.180, 0.0, 3.0], "CHI2": [1, 0.016, 0.0, 3.0]},
    "PHE": {"CHI1": [1, 5.728, 180.0, 3.0], "CHI2": [1, 0.126, -21.2, 4.0]},
    "SER": {"CHI1": [1, 0.401, 0.0, 3.0]},
    "THR": {"CHI1": [1, 0.315, 0.0, 3.0]},
    "TRP": {"CHI1": [1, 2.475, 0.0, 3.0], "CHI2": [1, 1.029, 0.0, 3.0]},
    "TYR": {"CHI1": [1, 5.728, 180.0, 3.0], "CHI2": [1, 0.126, -21.2, 4.0]},
    "VAL": {"CHI1": [1, 0.148, 0.0, 3.0]},
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


def calculate_single_pdb_energy(df_pdb, sasa_tag, mode="all", force_field="charmm"):
    """
    Calculate total and average torsion potential energy for bound and unbound states of a single PDB structure.

    Parameters:
    -----------
    df_pdb : pd.DataFrame
        DataFrame rows corresponding to a single PDB structure.
    sasa_tag : str
        'I' for Interface residues, 'N' for Non-interface residues.
    mode : str
        'all' for all valid residues, 'aromatic' for aromatic residues only.
    force_field : str
        'charmm' or 'amber'.

    Returns:
    --------
    dict with keys: PDB, E_unbound, E_bound, E_unbound_avg, E_bound_avg, count
    """
    K_phi = 0.8
    K_psi = 0.4
    m_phi = m_psi = 3.0
    delta_phi = 180.0
    delta_psi = 0.0

    pot_fn = torsion_potential_charmm if force_field.lower() == "charmm" else torsion_potential_amber

    pdb_id = str(df_pdb["PDB"].iloc[0])
    e_u = 0.0
    e_b = 0.0
    res_count = 0

    for _, row in df_pdb.iterrows():
        res_name = str(row["LABEL"]).split()[0]

        if res_name in EXCLUDED_RESIDUES:
            continue

        if str(row["SASA"]).strip() != sasa_tag:
            continue

        if mode == "aromatic" and res_name not in AA_AROMATIC:
            continue

        if res_name not in DIHEDRAL_PARAMS:
            continue

        res_count += 1
        params = DIHEDRAL_PARAMS[res_name]

        # Backbone contributions
        e_u_res = pot_fn(row["U_PHI"], K_phi, m_phi, delta_phi) + pot_fn(row["U_PSI"], K_psi, m_psi, delta_psi)
        e_b_res = pot_fn(row["B_PHI"], K_phi, m_phi, delta_phi) + pot_fn(row["B_PSI"], K_psi, m_psi, delta_psi)

        # Side-chain Chi1 contribution
        if "CHI1" in params:
            k_chi1 = params["CHI1"][1] / params["CHI1"][0]
            m_chi1 = params["CHI1"][3]
            d_chi1 = params["CHI1"][2]
            e_u_res += pot_fn(row["U_CHI1"], k_chi1, m_chi1, d_chi1)
            e_b_res += pot_fn(row["B_CHI1"], k_chi1, m_chi1, d_chi1)

        # Side-chain Chi2 contribution (for residues with >= 2 Chi angles)
        if res_name not in AA_ONLY_CHI1 and "CHI2" in params:
            k_chi2 = params["CHI2"][1] / params["CHI2"][0]
            m_chi2 = params["CHI2"][3]
            d_chi2 = params["CHI2"][2]
            e_u_res += pot_fn(row["U_CHI2"], k_chi2, m_chi2, d_chi2)
            e_b_res += pot_fn(row["B_CHI2"], k_chi2, m_chi2, d_chi2)

        e_u += e_u_res
        e_b += e_b_res

    e_u_avg = (e_u / res_count) if res_count > 0 else 0.0
    e_b_avg = (e_b / res_count) if res_count > 0 else 0.0

    return {
        "PDB": pdb_id,
        "E_unbound": round(e_u, 6),
        "E_bound": round(e_b, 6),
        "E_unbound_avg": round(e_u_avg, 6),
        "E_bound_avg": round(e_b_avg, 6),
        "count": res_count,
    }


def calculate_dataset_torsion_energy(df_torsion, mode="all", force_field="charmm"):
    """
    Calculate torsion potential energies across all PDBs in dataset for Interface and Non-interface states.
    """
    logging.info(f"Calculating dataset torsion energy (mode='{mode}', force_field='{force_field}')...")
    pdbs = df_torsion["PDB"].unique()
    energy_records = []

    for pdb in pdbs:
        df_pdb = df_torsion[df_torsion["PDB"] == pdb]

        res_int = calculate_single_pdb_energy(df_pdb, sasa_tag="I", mode=mode, force_field=force_field)
        res_nint = calculate_single_pdb_energy(df_pdb, sasa_tag="N", mode=mode, force_field=force_field)

        record = {
            "PDB": pdb,
            "IU_DELG": res_int["E_unbound"],
            "IB_DELG": res_int["E_bound"],
            "IU_AVG_DELG": res_int["E_unbound_avg"],
            "IB_AVG_DELG": res_int["E_bound_avg"],
            "ICOUNT": res_int["count"],
            "NU_DELG": res_nint["E_unbound"],
            "NB_DELG": res_nint["E_bound"],
            "NU_AVG_DELG": res_nint["E_unbound_avg"],
            "NB_AVG_DELG": res_nint["E_bound_avg"],
            "NCOUNT": res_nint["count"],
        }
        energy_records.append(record)

    df_energy = pd.DataFrame(energy_records)
    logging.info(f"Computed potential energy for {len(df_energy)} PDB entries.")
    return df_energy


def save_energy_dat_files(df_energy, output_dir=RESULTS_DIR, prefix="RBPs"):
    """
    Save individual energy .dat files matching classical output formats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files_map = {
        f"{prefix}_B_INT_187.dat": ("IB_DELG", "B"),
        f"{prefix}_B_NINT_187.dat": ("NB_DELG", "B"),
        f"{prefix}_avg_B_INT_187.dat": ("IB_AVG_DELG", "B"),
        f"{prefix}_avg_B_NINT_187.dat": ("NB_AVG_DELG", "B"),
        f"{prefix}_U_INT_187.dat": ("IU_DELG", "U"),
        f"{prefix}_U_NINT_187.dat": ("NU_DELG", "U"),
        f"{prefix}_avg_U_INT_187.dat": ("IU_AVG_DELG", "U"),
        f"{prefix}_avg_U_NINT_187.dat": ("NU_AVG_DELG", "U"),
    }

    for fname, (col_name, state) in files_map.items():
        file_path = output_dir / fname
        with open(file_path, "w") as f:
            for _, row in df_energy.iterrows():
                f.write(f"{row['PDB']}\t{row[col_name]}\t{state}\n")
        logging.info(f"Saved energy DAT file: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Calculate torsion potential energy (DeltaG) for bound and unbound protein structures.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DATASET_DIR / "PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv",
        help="Path to backbone & sidechain torsion angle dataset file.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for energy files.")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "aromatic"], help="Torsion calculation mode.")
    parser.add_argument("--force-field", type=str, default="charmm", choices=["charmm", "amber"], help="Force field energy function formulation.")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_file.exists():
        logging.error(f"Input file not found: {args.input_file}")
        return

    # Determine file separator
    try:
        df_torsion = pd.read_csv(args.input_file, sep=",")
        if "LABEL" not in df_torsion.columns:
            df_torsion = pd.read_csv(args.input_file, sep="\t")
    except Exception:
        df_torsion = pd.read_csv(args.input_file, sep="\t")

    logging.info(f"Loaded dataset with {len(df_torsion)} residue entries from {args.input_file}")

    # Calculate dataset energy
    df_energy = calculate_dataset_torsion_energy(df_torsion, mode=args.mode, force_field=args.force_field)

    # Save summary CSV
    output_csv = args.output_dir / "torsion_potential_energy_summary.csv"
    df_energy.to_csv(output_csv, index=False)
    logging.info(f"Saved unified energy summary table to: {output_csv}")

    # Save DAT files
    save_energy_dat_files(df_energy, output_dir=args.output_dir)

    logging.info("DeltaG torsion energy calculation completed successfully.")


if __name__ == "__main__":
    main()
