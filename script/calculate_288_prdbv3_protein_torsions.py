#!/usr/bin/env python3
"""
calculate_288_prdbv3_protein_torsions.py

Executes PRince for 288 non-redundant protein-RNA complexes and calculates complete
bound protein backbone, side-chain torsion angles, DSSP secondary structure, and SASA
interface classifications.

Inputs:
  - Dataset Directory: /home/labuser/Projects/results/nr_288_PRDBv3.0
  - JSON List: /home/labuser/Projects/results/nr_288_PRDBv3.0/nr_PRDBv3.0_288_List.json
  - CIF Files: /home/labuser/Projects/results/nr_288_PRDBv3.0/cif_files

Outputs:
  - CSV: /home/labuser/Projects/results/nr_288_PRDBv3.0/ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.csv
  - TSV: /home/labuser/Projects/results/nr_288_PRDBv3.0/ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.tsv
  - JSON: /home/labuser/Projects/results/nr_288_PRDBv3.0/ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.json
"""

import os
import sys
import json
import logging
import subprocess
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = Path("/home/labuser/Projects/results/nr_288_PRDBv3.0")
JSON_LIST_PATH = BASE_DIR / "nr_PRDBv3.0_288_List.json"
CIF_DIR = BASE_DIR / "cif_files"
PRINCE_OUT_DIR = BASE_DIR / "prince_out"
PRINCE_BIN = Path("/home/labuser/Projects/PRince/bin/prince")

SCRIPT_DIR = Path(__file__).resolve().parent
ROTA_ASSIGN_DIR = SCRIPT_DIR.parent.parent / "rota_assign"
if str(ROTA_ASSIGN_DIR) not in sys.path:
    sys.path.insert(0, str(ROTA_ASSIGN_DIR))

from macromol_torsion.protein_backbone import ProteinBackboneCalculator
from macromol_torsion.protein_sidechain import ProteinSidechainCalculator
from macromol_torsion.dssp import DsspAnalyzer
from macromol_torsion.structure_reader import StructureReader

PRINCE_OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_prince_int_file(int_file_path):
    """Extracts interface residue keys (resname, chain, seq) from PRince .int file."""
    int_p = Path(int_file_path)
    if not int_p.exists():
        return set()

    interface_keys = set()
    with open(int_p, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_name = line[17:20].strip()
                chain = line[21:22].strip()
                seq = line[22:26].strip()
                interface_keys.add((res_name, chain, seq))
    return interface_keys


def run_prince(cif_path, prot_chains, rna_chains, out_dir):
    """Runs PRince CLI for a given structure."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PRINCE_BIN),
        "-i", str(cif_path),
        "-p", str(prot_chains),
        "-r", str(rna_chains),
        "-o", str(out_p)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logging.warning(f"PRince failed for {cif_path.name}: {res.stderr.strip()[:200]}")
    return out_p


def main():
    logging.info(f"Loading 288 complex list: {JSON_LIST_PATH}")
    with open(JSON_LIST_PATH, "r") as f:
        items = json.load(f)

    logging.info(f"Total complexes in list: {len(items)}")

    sc_calc = ProteinSidechainCalculator()
    bb_calc = ProteinBackboneCalculator()
    dssp = DsspAnalyzer(dssp_binary="/home/labuser/bin/dssp")

    all_records = []
    processed_count = 0

    for idx, item in enumerate(items):
        pdb_id = item["PDB ID"].strip()
        pr_chains = item["PR_chains"].strip()

        # Parse protein and RNA chains
        parts = pr_chains.split(":")
        prot_chains = parts[0].strip() if len(parts) > 0 else "A"
        rna_chains = parts[1].strip() if len(parts) > 1 else "R"

        cif_path = CIF_DIR / f"{pdb_id}.cif"
        if not cif_path.exists():
            logging.warning(f"[{idx+1}/{len(items)}] CIF file missing for {pdb_id}: {cif_path}. Skipping.")
            continue

        processed_count += 1
        logging.info(f"[{processed_count}/{len(items)}] Processing {pdb_id} (Prot: {prot_chains}, RNA: {rna_chains})...")

        # 1. Run PRince analysis to extract protein-RNA interface
        c_prince_out = PRINCE_OUT_DIR / pdb_id
        run_prince(cif_path, prot_chains, rna_chains, c_prince_out)

        int_file = c_prince_out / f"{pdb_id}P.int"
        if not int_file.exists():
            int_file = c_prince_out / f"{pdb_id}.int"

        sur_file = c_prince_out / f"{pdb_id}P.sur"
        if not sur_file.exists():
            sur_file = c_prince_out / f"{pdb_id}.sur"

        interface_keys = parse_prince_int_file(int_file)
        surface_keys = parse_prince_int_file(sur_file)

        # 2. Parse structure atoms using unified StructureReader
        atoms = StructureReader.parse_structure(cif_path)

        # Filter target protein chain atoms
        target_prot_chains = set([c for c in prot_chains.replace(',', '').replace(':', '') if c.strip()])

        # 3. Secondary Structure Assignment via DSSP
        ss_map = dssp.get_secondary_structure(cif_path)

        # 4. Protein Backbone Dihedrals
        phi, psi, omega, _, _, labels = bb_calc.calculate_dihedrals(atoms)

        # 5. Protein Sidechain Dihedrals
        chis = sc_calc.calculate_all_chis(atoms)

        # 6. Process residues
        for i, lbl in enumerate(labels):
            parts = lbl.split()
            if len(parts) >= 3:
                res_name, chain, seq = parts[0], parts[1], parts[2]
                if target_prot_chains and chain not in target_prot_chains:
                    continue

                # Sidechain Chi angles
                c1 = chis['CHI1'].get(lbl, 0.0)
                c2 = chis['CHI2'].get(lbl, 0.0)
                c3 = chis['CHI3'].get(lbl, 0.0)
                c4 = chis['CHI4'].get(lbl, 0.0)
                c5 = chis['CHI5'].get(lbl, 0.0)

                # Secondary structure
                ss_val = ss_map.get(lbl, "L")

                # SASA classification:
                # 'I': Interface (in .int)
                # 'N': Non-interface Surface (in .sur but not in .int)
                # 'C': Core buried (not in .sur)
                is_int = (res_name, chain, seq) in interface_keys or (chain, seq) in [(t[1], t[2]) for t in interface_keys]
                is_sur = (res_name, chain, seq) in surface_keys or (chain, seq) in [(t[1], t[2]) for t in surface_keys]

                if is_int:
                    sasa_val = "I"
                elif is_sur:
                    sasa_val = "N"
                else:
                    sasa_val = "C"

                record = {
                    "PDB": pdb_id,
                    "LABEL": lbl,
                    "PHI": phi[i],
                    "PSI": psi[i],
                    "OMEGA": omega[i],
                    "CHI1": c1,
                    "CHI2": c2,
                    "CHI3": c3,
                    "CHI4": c4,
                    "CHI5": c5,
                    "SS": ss_val,
                    "SASA": sasa_val,
                }
                all_records.append(record)

    df = pd.DataFrame(all_records)
    logging.info(f"288 Protein-RNA complex torsion calculation complete! Total residues: {len(df)}")

    # Export CSV, TSV, JSON outputs
    csv_file = BASE_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.csv"
    tsv_file = BASE_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.tsv"
    json_file = BASE_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.json"

    df.to_csv(csv_file, index=False)
    df.to_csv(tsv_file, sep="\t", index=False)

    records_json = df.to_dict(orient="records")
    with open(json_file, "w") as f_json:
        json.dump(records_json, f_json, indent=2)

    # Sync to output_files/
    out_dir_projects = SCRIPT_DIR / "RBPs_rotamer_lib" / "output_files"
    out_dir_projects.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir_projects / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.csv", index=False)
    df.to_csv(out_dir_projects / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.tsv", sep="\t", index=False)
    with open(out_dir_projects / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_288.json", "w") as f_j2:
        json.dump(records_json, f_j2, indent=2)

    logging.info("==================================================")
    logging.info("288 PROTEIN-RNA TORSION ANGLE PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info(f"  CSV Output:  {csv_file} ({os.path.getsize(csv_file)/1e6:.2f} MB)")
    logging.info(f"  TSV Output:  {tsv_file} ({os.path.getsize(tsv_file)/1e6:.2f} MB)")
    logging.info(f"  JSON Output: {json_file} ({os.path.getsize(json_file)/1e6:.2f} MB)")
    logging.info("==================================================")


if __name__ == "__main__":
    main()
