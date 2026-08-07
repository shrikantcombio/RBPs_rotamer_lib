#!/usr/bin/env python3
"""
cispep_pdb.py

Collects Cis-peptide information from PDB files (via CISPEP header records)
and/or from backbone omega torsion angle datasets.

Usage:
    python script/cispep_pdb.py [options]
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
PRDBV3_DIR = ROOT_DIR / "PRDBv3_dataset" / "PRDBv3.0"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_cispep_from_pdb(pdb_path, chain_ids=None):
    """
    Parse CISPEP records from a standard PDB file.

    PDB CISPEP format:
    COLUMNS      DATA TYPE       FIELD         DEFINITION
    ---------------------------------------------------------------------------------
    1 - 6        Record name     "CISPEP"
    8 - 10       Integer         serNum        Record serial number.
    12 - 14      Residue name    pep1          Name of the first residue.
    16           Character       chainID1      Chain identifier for first residue.
    18 - 21      Integer         seqNum1       Residue sequence number for first residue.
    22           AChar           iCode1        Insertion code for first residue.
    26 - 28      Residue name    pep2          Name of the second residue.
    30           Character       chainID2      Chain identifier for second residue.
    32 - 35      Integer         seqNum2       Residue sequence number for second residue.
    36           AChar           iCode2        Insertion code for second residue.
    44 - 46      Integer         modNum        Identifies the specific model.
    54 - 59      Real(6.2)       measure       Cis peptide angle in degrees.
    """
    cispep_records = []
    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        logging.warning(f"PDB file not found: {pdb_path}")
        return cispep_records

    pdb_id = pdb_path.stem

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("CISPEP"):
                try:
                    ser_num = line[7:10].strip()
                    pep1_resname = line[11:14].strip()
                    pep1_chnid = line[15].strip()
                    pep1_resnum = line[17:21].strip()
                    pep1_icode = line[21].strip()

                    pep2_resname = line[25:28].strip()
                    pep2_chnid = line[29].strip()
                    pep2_resnum = line[31:35].strip()
                    pep2_icode = line[36].strip()

                    mod_num = line[43:46].strip()
                    pep_angle_str = line[53:59].strip()
                    pep_angle = float(pep_angle_str) if pep_angle_str else np.nan

                    if chain_ids is None or pep1_chnid in chain_ids or pep2_chnid in chain_ids:
                        record = {
                            "PDB_ID": pdb_id,
                            "SER_NUM": ser_num,
                            "PEP1_RES": pep1_resname,
                            "PEP1_CHAIN": pep1_chnid,
                            "PEP1_RESNUM": pep1_resnum,
                            "PEP1_ICODE": pep1_icode,
                            "PEP2_RES": pep2_resname,
                            "PEP2_CHAIN": pep2_chnid,
                            "PEP2_RESNUM": pep2_resnum,
                            "PEP2_ICODE": pep2_icode,
                            "MODEL_NUM": mod_num,
                            "OMEGA_ANGLE": pep_angle,
                        }
                        cispep_records.append(record)
                except Exception as e:
                    logging.warning(f"Error parsing line in {pdb_path}: '{line.strip()}' -> {e}")

    return cispep_records


def collect_cispep_from_pdb_dataset(dataset_dir=PRDBV3_DIR, json_list_path=None, output_csv=None):
    """
    Collect CISPEP records across a PDB dataset using either a JSON list mapping or by scanning directory.
    """
    logging.info("Collecting CISPEP records from PDB files...")
    records = []

    if json_list_path and Path(json_list_path).exists():
        logging.info(f"Reading dataset list from JSON: {json_list_path}")
        prdb_df = pd.read_json(json_list_path)

        for _, row in prdb_df.iterrows():
            complex_pdb = row.get("C_PDB")
            unbound_pdb = row.get("U_pro_PDB")
            p_chain = str(row.get("C_pro_chain", ""))
            ub_pro_chain = str(row.get("U_pro_chain", ""))

            if pd.notna(complex_pdb) and complex_pdb:
                b_pdb_file = dataset_dir / complex_pdb / f"{complex_pdb}.pdb"
                c_records = parse_cispep_from_pdb(b_pdb_file, chain_ids=p_chain)
                for rec in c_records:
                    rec["TYPE"] = "Bound/Complex"
                records.extend(c_records)

            if pd.notna(unbound_pdb) and unbound_pdb:
                # Clean unbound PDB ID (remove asterisks if present)
                clean_ub_pdb = str(unbound_pdb).replace("*", "").strip()
                u_pdb_file = dataset_dir / complex_pdb / f"{clean_ub_pdb}.pdb"
                if not u_pdb_file.exists():
                    # Fallback check directly in dataset directory
                    u_pdb_file = dataset_dir / f"{clean_ub_pdb}.pdb"

                u_records = parse_cispep_from_pdb(u_pdb_file, chain_ids=ub_pro_chain)
                for rec in u_records:
                    rec["TYPE"] = "Unbound"
                records.extend(u_records)
    else:
        logging.info(f"Scanning directory for PDB files: {dataset_dir}")
        for pdb_file in dataset_dir.rglob("*.pdb"):
            recs = parse_cispep_from_pdb(pdb_file)
            for rec in recs:
                rec["TYPE"] = "Scanned"
            records.extend(recs)

    df_cispep = pd.DataFrame(records)
    logging.info(f"Total CISPEP records collected from PDB files: {len(df_cispep)}")

    if output_csv and not df_cispep.empty:
        df_cispep.to_csv(output_csv, index=False)
        logging.info(f"Saved PDB CISPEP records to: {output_csv}")

    return df_cispep


def classify_omega_angle(omega):
    """
    Classify backbone omega torsion angle into cis or trans conformation.
    Cis peptides: omega in 0° ± 30° ([-30, 30])
    Trans peptides: omega in 180° ± 30° ([150, 210] or [-180, -150])
    """
    if pd.isna(omega):
        return "Unknown"

    omega = float(omega)
    if -30.0 <= omega <= 30.0:
        return "cis"
    elif abs(omega) >= 150.0:
        return "trans"
    else:
        return "Other"


def collect_cispep_from_omega_torsions(torsion_file_path, residue_filter="PRO", output_csv=None):
    """
    Extract cis-peptide candidates from backbone omega torsion angle datasets.
    """
    logging.info(f"Reading omega torsion angles from: {torsion_file_path}")
    torsion_path = Path(torsion_file_path)

    if not torsion_path.exists():
        logging.error(f"Torsion file not found: {torsion_path}")
        return pd.DataFrame()

    # Determine file separator: try comma first if .csv or if comma in header, fallback to tab
    try:
        df_torsion = pd.read_csv(torsion_path, sep=",")
        if "LABEL" not in df_torsion.columns:
            df_torsion = pd.read_csv(torsion_path, sep="\t")
    except Exception:
        df_torsion = pd.read_csv(torsion_path, sep="\t")

    if "LABEL" not in df_torsion.columns or "B_OMEGA" not in df_torsion.columns:
        logging.error(f"Required columns ('LABEL', 'B_OMEGA') not found in {torsion_path}")
        return pd.DataFrame()

    # Add classification columns
    df_torsion["B_CIS_TRANS"] = df_torsion["B_OMEGA"].apply(classify_omega_angle)
    if "U_OMEGA" in df_torsion.columns:
        df_torsion["U_CIS_TRANS"] = df_torsion["U_OMEGA"].apply(classify_omega_angle)

    # Filter residues if requested
    if residue_filter and residue_filter.upper() != "ALL":
        target_res = residue_filter.upper()
        res_indices = df_torsion[df_torsion["LABEL"].str.contains(target_res, na=False)].index

        matched_rows = []
        for idx in res_indices:
            # Include previous row in the sequence for peptide context if available
            if idx > 0:
                prev_row = df_torsion.iloc[idx - 1].copy()
                prev_row["CONTEXT"] = "PREV_RESIDUE"
                matched_rows.append(prev_row)

            target_row = df_torsion.iloc[idx].copy()
            target_row["CONTEXT"] = "TARGET_RESIDUE"
            matched_rows.append(target_row)

        df_filtered = pd.DataFrame(matched_rows)
    else:
        df_filtered = df_torsion.copy()
        df_filtered["CONTEXT"] = "ALL_RESIDUES"

    logging.info(f"Collected {len(df_filtered)} residue rows matching filter '{residue_filter}'.")

    # Count cis vs trans
    cis_count_b = (df_filtered["B_CIS_TRANS"] == "cis").sum()
    trans_count_b = (df_filtered["B_CIS_TRANS"] == "trans").sum()
    logging.info(f"Bound state conformation counts - Cis: {cis_count_b}, Trans: {trans_count_b}")

    if output_csv and not df_filtered.empty:
        df_filtered.to_csv(output_csv, index=False)
        logging.info(f"Saved omega cis/trans classification to: {output_csv}")

    return df_filtered


def main():
    parser = argparse.ArgumentParser(description="Collect and analyze cis-peptide information from PDB files and Omega torsion angles.")
    parser.add_argument("--pdb-dir", type=Path, default=PRDBV3_DIR, help="Path to PDB dataset directory.")
    parser.add_argument("--json-list", type=Path, default=DATASET_DIR / "PRDBv3_pdbList_chains.json", help="Path to JSON file containing PDB chain mappings.")
    parser.add_argument("--torsion-file", type=Path, default=DATASET_DIR / "PROTEIN_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv", help="Path to backbone torsion angles dataset file.")
    parser.add_argument("--residue", type=str, default="PRO", help="Residue type to filter in torsion angle analysis (e.g. PRO, ALL).")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for CSV results.")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect CISPEP records from PDB files
    pdb_output_csv = args.output_dir / "CISPEP_from_PDB.csv"
    df_pdb_cispep = collect_cispep_from_pdb_dataset(
        dataset_dir=args.pdb_dir,
        json_list_path=args.json_list if args.json_list.exists() else None,
        output_csv=pdb_output_csv
    )

    # 2. Collect & classify cis/trans peptides from Omega torsion angles
    omega_output_csv = args.output_dir / f"{args.residue.upper()}_BACKBONE_OMEGA_CLASSIFICATION.csv"
    df_omega_cispep = collect_cispep_from_omega_torsions(
        torsion_file_path=args.torsion_file,
        residue_filter=args.residue,
        output_csv=omega_output_csv
    )

    logging.info("Cis-peptide information collection completed successfully.")


if __name__ == "__main__":
    main()
