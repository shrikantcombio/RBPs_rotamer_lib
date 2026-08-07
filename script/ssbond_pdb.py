#!/usr/bin/env python3
"""
ssbond_pdb.py

Disulfide bond calculator from PDB (.pdb) and mmCIF (.cif) files.
Extracts SSBOND metadata from file headers and calculates 3D SG-SG distances,
disulfide chi3 torsion angles (CB1-SG1-SG2-CB2), and categorizes Cysteine
residues into:
  1. Cys_all: Total Cysteine count
  2. Cys_with_disulfide (CYD): Disulfide-bonded Cysteines
  3. Cys_without_disulfide (CYH): Free/non-disulfide-bonded Cysteines

Usage:
    python script/ssbond_pdb.py --input-path <path_to_pdb_or_cif_or_dir> [options]
"""

import argparse
import logging
import math
import sys
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

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from script.utils.mmcif_clean_reader import parse_file
except ImportError:
    from utils.mmcif_clean_reader import parse_file


class DisulfideBondCalculator:
    """
    Calculator to extract and analyze disulfide bonds from PDB and mmCIF structure files.
    """

    def __init__(self, min_dist=2.0, max_dist=2.5):
        """
        Initialize DisulfideBondCalculator.

        Parameters:
        -----------
        min_dist : float
            Minimum SG-SG distance threshold in Angstroms (default 2.0).
        max_dist : float
            Maximum SG-SG distance threshold in Angstroms (default 2.5).
        """
        self.min_dist = min_dist
        self.max_dist = max_dist

    @staticmethod
    def vector(p1, p2):
        """Returns the 3D vector from point p1 to p2."""
        return np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]], dtype=float)

    @classmethod
    def calculate_dihedral_angle(cls, p1, p2, p3, p4):
        """
        Calculate the dihedral angle in degrees between four 3D points.

        Parameters:
        -----------
        p1, p2, p3, p4 : array-like of shape (3,)
            3D Cartesian coordinates of the 4 atoms.

        Returns:
        --------
        float : Dihedral angle in degrees (-180.0 to 180.0).
        """
        b1 = cls.vector(p1, p2)
        b2 = cls.vector(p2, p3)
        b3 = cls.vector(p3, p4)

        norm_b2 = np.linalg.norm(b2)
        if norm_b2 == 0:
            return 0.0
        b2 /= norm_b2

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)
        if norm_n1 == 0 or norm_n2 == 0:
            return 0.0

        n1 /= norm_n1
        n2 /= norm_n2

        x = np.dot(n1, n2)
        y = np.dot(np.cross(n1, n2), b2)

        angle = np.degrees(np.arctan2(y, x))
        return float(angle)

    def parse_header_ssbonds_pdb(self, pdb_path, chain_ids=None):
        """
        Parse SSBOND record lines from a PDB header.

        Parameters:
        -----------
        pdb_path : Path or str
            Path to PDB file.
        chain_ids : list of str, optional
            Allowed chain IDs.

        Returns:
        --------
        list of dict : Disulfide bond metadata records.
        """
        ssbonds = []
        path = Path(pdb_path)
        if not path.exists():
            return ssbonds

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("SSBOND"):
                    try:
                        ser_num = line[7:10].strip()
                        cys1_name = line[11:14].strip()
                        cys1_chain = line[15].strip()
                        cys1_resnum = line[17:21].strip()
                        cys1_icode = line[22].strip()

                        cys2_name = line[25:28].strip()
                        cys2_chain = line[29].strip()
                        cys2_resnum = line[31:35].strip()
                        cys2_icode = line[36].strip()

                        sym1 = line[59:65].strip() if len(line) >= 65 else ""
                        sym2 = line[66:72].strip() if len(line) >= 72 else ""

                        length_str = line[73:78].strip() if len(line) >= 78 else "0.0"
                        length = float(length_str) if length_str else 0.0

                        if chain_ids is None or (cys1_chain in chain_ids or cys2_chain in chain_ids):
                            ssbonds.append(
                                {
                                    "file": path.name,
                                    "ser_num": ser_num,
                                    "cys1_chain": cys1_chain,
                                    "cys1_resnum": cys1_resnum,
                                    "cys1_icode": cys1_icode,
                                    "cys2_chain": cys2_chain,
                                    "cys2_resnum": cys2_resnum,
                                    "cys2_icode": cys2_icode,
                                    "header_length": length,
                                    "sym1": sym1,
                                    "sym2": sym2,
                                }
                            )
                    except Exception as e:
                        logging.debug(f"Could not parse SSBOND line in {path.name}: {line} ({e})")
        return ssbonds

    def parse_header_ssbonds_cif(self, cif_path, chain_ids=None):
        """
        Parse _struct_conn disulfide records from an mmCIF header.

        Parameters:
        -----------
        cif_path : Path or str
            Path to mmCIF file.
        chain_ids : list of str, optional
            Allowed chain IDs.

        Returns:
        --------
        list of dict : Disulfide bond metadata records from mmCIF.
        """
        ssbonds = []
        path = Path(cif_path)
        if not path.exists():
            return ssbonds

        # Parse CIF _struct_conn category manually or via MMCIFParser
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        in_struct_conn = False
        headers = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("_struct_conn."):
                in_struct_conn = True
                headers.append(line_str.split(".")[1])
            elif in_struct_conn and not line_str.startswith("_struct_conn."):
                if line_str.startswith("#") or line_str.startswith("loop_") or not line_str:
                    in_struct_conn = False
                    headers = []
                    continue

                parts = line_str.split()
                if len(parts) >= len(headers):
                    data = dict(zip(headers, parts))
                    conn_type = data.get("conn_type_id", "").lower()
                    if "disulf" in conn_type:
                        c1_chain = data.get("ptnr1_label_asym_id", data.get("ptnr1_auth_asym_id", ""))
                        c1_resnum = data.get("ptnr1_label_seq_id", data.get("ptnr1_auth_seq_id", ""))
                        c2_chain = data.get("ptnr2_label_asym_id", data.get("ptnr2_auth_asym_id", ""))
                        c2_resnum = data.get("ptnr2_label_seq_id", data.get("ptnr2_auth_seq_id", ""))
                        dist_val = float(data.get("pdbx_dist_value", "0.0")) if data.get("pdbx_dist_value", "?") != "?" else 0.0

                        if chain_ids is None or (c1_chain in chain_ids or c2_chain in chain_ids):
                            ssbonds.append(
                                {
                                    "file": path.name,
                                    "ser_num": data.get("id", ""),
                                    "cys1_chain": c1_chain,
                                    "cys1_resnum": c1_resnum,
                                    "cys1_icode": "",
                                    "cys2_chain": c2_chain,
                                    "cys2_resnum": c2_resnum,
                                    "cys2_icode": "",
                                    "header_length": dist_val,
                                    "sym1": data.get("ptnr1_symmetry", ""),
                                    "sym2": data.get("ptnr2_symmetry", ""),
                                }
                            )

        return ssbonds

    def _extract_cysteine_atoms(self, file_path, chain_ids=None):
        """
        Extract Cysteine residue atom information (SG, CB, coordinates, chain, resnum)
        using script/utils/mmcif_clean_reader.py.
        """
        path = Path(file_path)
        cysteines = []

        try:
            from script.utils.mmcif_clean_reader import parse_file
            cols, rows = parse_file(str(path))
            res_dict = {}
            for r in rows:
                ch_id = r.get("_atom_site.auth_asym_id") or r.get("_atom_site.label_asym_id")
                if chain_ids is None or ch_id in chain_ids:
                    res_name = r.get("_atom_site.auth_comp_id") or r.get("_atom_site.label_comp_id")
                    if res_name == "CYS":
                        res_num = r.get("_atom_site.auth_seq_id") or r.get("_atom_site.label_seq_id")
                        icode = r.get("_atom_site.pdbx_PDB_ins_code", ".").strip()
                        if icode == ".":
                            icode = ""
                        atom_name = r.get("_atom_site.auth_atom_id") or r.get("_atom_site.label_atom_id")
                        res_key = f"{ch_id}:{res_num}{icode}"

                        try:
                            x = float(r["_atom_site.Cartn_x"])
                            y = float(r["_atom_site.Cartn_y"])
                            z = float(r["_atom_site.Cartn_z"])
                            coord = np.array([x, y, z], dtype=float)
                        except (KeyError, ValueError):
                            continue

                        if res_key not in res_dict:
                            res_dict[res_key] = {
                                "chain": ch_id,
                                "resnum": res_num,
                                "icode": icode,
                                "res_key": res_key,
                                "sg_coord": None,
                                "cb_coord": None,
                            }

                        if atom_name == "SG":
                            res_dict[res_key]["sg_coord"] = coord
                        elif atom_name == "CB":
                            res_dict[res_key]["cb_coord"] = coord

            cysteines = [v for v in res_dict.values() if v["sg_coord"] is not None]
            return cysteines
        except Exception as e:
            logging.error(f"mmcif_clean_reader parsing failed for {path.name}: {e}")
            return cysteines

    def calculate_disulfide_bonds(self, file_path, chain_ids=None):
        """
        Calculate disulfide bond distances and chi3 torsion angles for all Cysteine pairs in a structure.

        Parameters:
        -----------
        file_path : Path or str
            Path to PDB or mmCIF structure file.
        chain_ids : list of str, optional
            Chain IDs to include.

        Returns:
        --------
        dict containing:
            'bonds': list of dicts (disulfide bond details)
            'cys_all': int (Total CYS count)
            'cys_cyd': int (Disulfide-bonded CYS count, CYD)
            'cys_cyh': int (Free CYS count, CYH)
            'cyd_keys': set of str (Residue keys involved in SSBOND)
            'cyh_keys': set of str (Free residue keys)
        """
        path = Path(file_path)
        cysteines = self._extract_cysteine_atoms(path, chain_ids=chain_ids)
        cys_all_count = len(cysteines)

        bonds = []
        cyd_keys = set()

        for i in range(len(cysteines)):
            for j in range(i + 1, len(cysteines)):
                c1 = cysteines[i]
                c2 = cysteines[j]

                dist = float(np.linalg.norm(c1["sg_coord"] - c2["sg_coord"]))

                if self.min_dist <= dist <= self.max_dist:
                    chi3_angle = None
                    if c1["cb_coord"] is not None and c2["cb_coord"] is not None:
                        chi3_angle = self.calculate_dihedral_angle(c1["cb_coord"], c1["sg_coord"], c2["sg_coord"], c2["cb_coord"])

                    cyd_keys.add(c1["res_key"])
                    cyd_keys.add(c2["res_key"])

                    bonds.append(
                        {
                            "file": path.name,
                            "cys1_chain": c1["chain"],
                            "cys1_resnum": c1["resnum"],
                            "cys1_icode": c1["icode"],
                            "cys2_chain": c2["chain"],
                            "cys2_resnum": c2["resnum"],
                            "cys2_icode": c2["icode"],
                            "distance_A": round(dist, 3),
                            "chi3_deg": round(chi3_angle, 2) if chi3_angle is not None else None,
                        }
                    )

        cyh_keys = set(c["res_key"] for c in cysteines) - cyd_keys
        cys_cyd_count = len(cyd_keys)
        cys_cyh_count = len(cyh_keys)

        return {
            "file": path.name,
            "bonds": bonds,
            "cys_all": cys_all_count,
            "cys_cyd": cys_cyd_count,
            "cys_cyh": cys_cyh_count,
            "cyd_keys": cyd_keys,
            "cyh_keys": cyh_keys,
        }

    def process_file(self, file_path, chain_ids=None):
        """
        Process a single structure file for both header metadata and 3D coordinate-calculated disulfide bonds.

        Parameters:
        -----------
        file_path : Path or str
            Path to PDB or CIF structure.
        chain_ids : list of str, optional
            Chain ID filter.

        Returns:
        --------
        dict containing header SSBONDs, 3D SSBONDs, and CYS counts.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # 1. Header parsing
        if suffix in [".cif", ".mmcif"]:
            header_bonds = self.parse_header_ssbonds_cif(path, chain_ids=chain_ids)
        else:
            header_bonds = self.parse_header_ssbonds_pdb(path, chain_ids=chain_ids)

        # 2. 3D coordinate calculation
        coord_result = self.calculate_disulfide_bonds(path, chain_ids=chain_ids)
        coord_result["header_bonds"] = header_bonds

        return coord_result


def main():
    parser = argparse.ArgumentParser(description="Disulfide Bond Calculator from PDB (.pdb) and mmCIF (.cif) files.")
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to input PDB/CIF file or directory containing structure files.",
    )
    parser.add_argument("--min-dist", type=float, default=2.0, help="Minimum SG-SG distance threshold in Angstroms.")
    parser.add_argument("--max-dist", type=float, default=2.5, help="Maximum SG-SG distance threshold in Angstroms.")
    parser.add_argument("--chain-id", type=str, default=None, help="Optional chain ID filter (e.g. 'A' or 'A:B').")
    parser.add_argument("--output-csv", type=Path, default=RESULTS_DIR / "disulfide_bonds_summary.csv", help="Path to save output CSV summary.")

    args = parser.parse_args()

    chain_ids = args.chain_id.split(":") if args.chain_id else None
    calculator = DisulfideBondCalculator(min_dist=args.min_dist, max_dist=args.max_dist)

    input_files = []
    if args.input_path.is_file():
        input_files.append(args.input_path)
    elif args.input_path.is_dir():
        for ext in ["*.pdb", "*.ent", "*.cif", "*.mmcif"]:
            input_files.extend(list(args.input_path.rglob(ext)))
    else:
        logging.error(f"Input path does not exist: {args.input_path}")
        return

    logging.info(f"Processing {len(input_files)} structure file(s)...")

    bonds_summary_list = []
    cys_counts_list = []

    for f_path in input_files:
        res = calculator.process_file(f_path, chain_ids=chain_ids)

        cys_counts_list.append(
            {
                "File": res["file"],
                "Cys_all": res["cys_all"],
                "Cys_with_disulfide_CYD": res["cys_cyd"],
                "Cys_without_disulfide_CYH": res["cys_cyh"],
                "Disulfide_Bonds_Found": len(res["bonds"]),
            }
        )

        for b in res["bonds"]:
            bonds_summary_list.append(b)

        # Log individual bond details to terminal
        for b in res["bonds"]:
            chi3_str = f"{b['chi3_deg']:7.2f}°" if b["chi3_deg"] is not None else " N/A"
            logging.info(
                f"{b['file']}: SSBOND between CYS {b['cys1_chain']}:{b['cys1_resnum']}{b['cys1_icode']} "
                f"and CYS {b['cys2_chain']}:{b['cys2_resnum']}{b['cys2_icode']} "
                f"| Distance: {b['distance_A']:.2f} Å | Chi3: {chi3_str}"
            )

    # Export CSV summary
    if bonds_summary_list:
        df_bonds = pd.DataFrame(bonds_summary_list)
        out_bonds_csv = args.output_csv.parent / f"{args.output_csv.stem}_bonds.csv"
        out_bonds_csv.parent.mkdir(parents=True, exist_ok=True)
        df_bonds.to_csv(out_bonds_csv, index=False)
        logging.info(f"Saved disulfide bonds detailed CSV: {out_bonds_csv}")

    if cys_counts_list:
        df_counts = pd.DataFrame(cys_counts_list)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_counts.to_csv(args.output_csv, index=False)
        logging.info(f"Saved Cysteine classification summary CSV: {args.output_csv}")

    logging.info("Disulfide bond calculation completed successfully.")


if __name__ == "__main__":
    main()
