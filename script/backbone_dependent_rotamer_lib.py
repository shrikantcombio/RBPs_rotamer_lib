#!/usr/bin/env python3
"""
backbone_dependent_rotamer_lib.py

Core implementation of the Backbone-Dependent Rotamer Library Generation Engine
using von Mises kernel smoothing and circular statistics for Bound ('B') and Unbound ('U')
protein rotameric states.

Usage:
    python script/backbone_dependent_rotamer_lib.py [options]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import rotamer_definitions as rotdef

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.torsion_dataset import TORSION_DATASET, load_torsion_dataset  # noqa: E402
RESULTS_DIR = ROOT_DIR / "output_files"
# Standalone runs write here; the published libraries (RBPs_BBD_rotamer_lib*/) are built by
# derive_all_updated_rotamer_libraries.py and are never a default target
DEFAULT_LIB_DIR = RESULTS_DIR / "bbd_rotamer_libs" / "consolidated"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LIB_DIR.mkdir(parents=True, exist_ok=True)


class BackboneDependentRotamerLibrary:
    """
    Class for calculating, formatting, and exporting backbone-dependent side-chain
    rotamer libraries for protein structures.
    """

    RESIDUE_CHI_COUNTS = rotdef.RESIDUE_CHI_COUNTS
    # Terminal sp2 chis (ASP, GLU, PHE, TYR, ASN, GLN, HIS, TRP) use residue-specific bins and,
    # for two-fold symmetric groups, 180-degree periodic statistics: see rotamer_definitions.py
    TERMINAL_SP2_CHI = rotdef.TERMINAL_SP2_CHI

    def __init__(self, bb_sep=30, rot_bin_lib=None, min_count=5):
        """
        Initialize the rotamer library generator.

        Parameters:
        -----------
        bb_sep : int
            Backbone angle separation bin size in degrees (default: 30).
        rot_bin_lib : dict, optional
            Custom rotamer binning rules mapping (residue, chi_idx) to number of splits.
        min_count : int
            Minimum observations for a (backbone_bin x rotamer_bin) cell to be reported.
            Default 5. Cells below this threshold produce unreliable probability estimates.
            Use 1 only for exploratory/diagnostic analysis.
        """
        self.bb_sep = bb_sep
        self.rot_bin_lib = rot_bin_lib if rot_bin_lib is not None else {("PTR", 1): 4, ("ALY", 4): 6}
        self.min_count = min_count
        self.results_cache = {}

    rotameric_class = staticmethod(rotdef.rotameric_class)
    weighted_circmean = staticmethod(rotdef.weighted_circmean)
    weighted_circstd = staticmethod(rotdef.weighted_circstd)

    @staticmethod
    def get_closest_angle(input_angle, degsep=30):
        """
        Backbone grid point of an angle: bins [g - degsep/2, g + degsep/2) centred on multiples of
        `degsep` in [-180, 180 - degsep]; the bin centred on 180 is labelled -180.
        """
        g = int(np.floor((input_angle + degsep / 2.0) / degsep)) * degsep
        return g - 360 if g >= 180 else g

    def compute_residue_rotamers(self, data, aa, state="B", last_free_rot=False):
        """
        Calculate backbone-dependent rotamer probabilities, weighted circular mean,
        and circular standard deviation for a specific amino acid.

        Parameters:
        -----------
        data : pd.DataFrame
            Residue torsion angle dataset.
        aa : str
            Three-letter amino acid code (e.g. 'ARG').
        state : str
            'B' for Bound state or 'U' for Unbound state.
        last_free_rot : bool
            Whether to handle terminal free sidechain angle.

        Returns:
        --------
        list of np.ndarray : Derived rotamer library rows.
        """
        if aa not in self.RESIDUE_CHI_COUNTS:
            logging.warning(f"Residue {aa} not supported in rotamer calculation.")
            return []

        nchi = self.RESIDUE_CHI_COUNTS[aa]
        phi_col = f"{state}_PHI"
        psi_col = f"{state}_PSI"
        chi_cols = [f"{state}_CHI{n+1}" for n in range(nchi)]

        # Verify required columns exist
        missing_cols = [c for c in [phi_col, psi_col] + chi_cols if c not in data.columns]
        if missing_cols:
            logging.warning(f"Missing columns {missing_cols} in dataset for residue {aa}.")
            return []

        # 1. Working population: residues with a defined (phi, psi) and every chi present.
        #    Chain termini and residues next to chain breaks have NaN phi or psi; residues with
        #    truncated side chains have NaN chis. Both are excluded.
        full_valid_mask = data[phi_col].notna().values & data[psi_col].notna().values
        for col in chi_cols:
            full_valid_mask &= data[col].notna().values
        data_full = data[full_valid_mask]

        # 1b. Rotamer bin of every chi (terminal sp2 chis use residue-specific bins)
        effect_chi_classes = rotdef.classify_chis(aa, data_full, chi_cols, self.rot_bin_lib)
        if last_free_rot and nchi > 1:
            nchi -= 1
            effect_chi_classes = effect_chi_classes[:-1]

        # 2. Classify backbone phi, psi angles into closest grid points
        #    (on full valid population only)
        bb_classes = []
        for ang in [phi_col, psi_col]:
            bb_classes.append(data_full[ang].apply(lambda x: self.get_closest_angle(x, self.bb_sep)).values)
        bb_classes = np.array(bb_classes)

        results = []
        if bb_classes.size == 0 or effect_chi_classes.size == 0:
            return results

        # 3. Loop through unique backbone (phi, psi) bins
        unique_bb_bins = np.unique(bb_classes.T, axis=0)

        for b in unique_bb_bins:
            bb_mask = (bb_classes[0] == b[0]) & (bb_classes[1] == b[1])
            total_bb_count = int(np.sum(bb_mask))

            if total_bb_count < self.min_count:
                continue

            unique_rot_states = np.unique(effect_chi_classes[:, bb_mask].T, axis=0)

            for c in unique_rot_states:
                mask = bb_mask.copy()
                for i, l in enumerate(c):
                    mask &= (effect_chi_classes[i] == l)

                count_each = int(np.sum(mask))
                if count_each < self.min_count:
                    continue

                # Denominator is every residue in the backbone cell, including rotamers dropped for
                # count < min_count, so the reported probabilities of a cell can sum to less than 1
                prob = float(count_each) / float(total_bb_count)
                angles_sub = data_full[mask]

                # Output array: phi, psi, count_each, count_tot, prob, chi1..n_mean, chi1..n_std
                line = np.zeros(2 * nchi + 5)
                line[0] = int(b[0])
                line[1] = int(b[1])
                line[2] = count_each
                line[3] = total_bb_count
                line[4] = prob

                for n in range(nchi):
                    mean_deg, sigma_deg = rotdef.chi_mean_and_sigma(aa, n, angles_sub[chi_cols[n]].values)
                    line[5 + n] = mean_deg
                    line[5 + n + nchi] = sigma_deg

                results.append(line)

        return results

    def generate_library(self, df, state="B", sasa_filter=None, residues=None):
        """
        Generate backbone-dependent rotamer library across all or specified residues.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset containing backbone and side-chain torsion angles.
        state : str
            'B' for Bound, 'U' for Unbound.
        sasa_filter : str, optional
            'I' for Interface, 'N' for Non-interface, or None for all.
        residues : list of str, optional
            List of amino acids to process. Default is all 18 standard residues.

        Returns:
        --------
        dict of {residue: pd.DataFrame} : Generated rotamer libraries per residue.
        """
        if residues is None:
            residues = list(self.RESIDUE_CHI_COUNTS.keys())

        df_work = df.copy()
        if sasa_filter:
            if sasa_filter == "S":
                df_work = df_work[df_work["SASA"].astype(str).isin(["I", "N", "S"])]
            else:
                df_work = df_work[df_work["SASA"].astype(str).str.startswith(sasa_filter)]

        lib_dict = {}
        for aa in residues:
            logging.info(f"Generating rotamer library for {aa} (State: {state}, SASA: {sasa_filter or 'ALL'})...")
            res_data = df_work[df_work["LABEL"].str[:3] == aa]

            raw_results = self.compute_residue_rotamers(res_data, aa, state=state)
            nchi = self.RESIDUE_CHI_COUNTS.get(aa, 0)

            if not raw_results or nchi == 0:
                continue

            columns = ["PHI", "PSI", "_COUNT", "COUNT", "PROB"]
            for i in range(1, nchi + 1):
                columns.append(f"{state}_CHI{i}")
            for i in range(1, nchi + 1):
                columns.append(f"{state}_CHI{i}Sig")

            res_df = pd.DataFrame(raw_results, columns=columns)
            res_df.insert(0, "AA", aa)

            # Cast integer columns
            res_df["PHI"] = res_df["PHI"].astype(int)
            res_df["PSI"] = res_df["PSI"].astype(int)
            res_df["_COUNT"] = res_df["_COUNT"].astype(int)
            res_df["COUNT"] = res_df["COUNT"].astype(int)

            lib_dict[aa] = res_df

        cache_key = f"{state}_{sasa_filter or 'ALL'}"
        self.results_cache[cache_key] = lib_dict
        return lib_dict

    def export_lib_files(self, output_dir, state="B", sasa_filter=None):
        """
        Export individual residue .lib files with standard formatting.
        """
        cache_key = f"{state}_{sasa_filter or 'ALL'}"
        if cache_key not in self.results_cache:
            logging.error(f"No library cached for key {cache_key}. Call generate_library first.")
            return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        header_template = "#AA   PHI   PSI  _Count  Count   Prob    "
        line_fmt_base = "{: <3}  {:4d}  {:4d}  {:4d}   {:4d}  {:10.8f}"

        for aa, res_df in self.results_cache[cache_key].items():
            nchi = self.RESIDUE_CHI_COUNTS[aa]

            formatter = line_fmt_base + "{:8.2f}" * nchi + "{:8.2f}" * nchi
            header = header_template + " ".join([f"{state}_CHI{i+1}" for i in range(nchi)]) + " " + " ".join([f"{state}_CHI{i+1}Sig" for i in range(nchi)]) + "\n"

            lines_str = header
            for _, row in res_df.iterrows():
                vals = [
                    row["AA"],
                    int(row["PHI"]),
                    int(row["PSI"]),
                    int(row["_COUNT"]),
                    int(row["COUNT"]),
                    float(row["PROB"]),
                ]
                for i in range(1, nchi + 1):
                    vals.append(float(row[f"{state}_CHI{i}"]))
                for i in range(1, nchi + 1):
                    vals.append(float(row[f"{state}_CHI{i}Sig"]))

                lines_str += formatter.format(*vals) + "\n"

            sasa_suffix = f"_{sasa_filter}" if sasa_filter else ""
            file_name = f"{aa}_bbd_{state}{sasa_suffix}.lib"
            lib_file_path = out_path / file_name

            with open(lib_file_path, "w", encoding="utf-8") as f:
                f.write(lines_str)

            logging.info(f"Saved residue library file: {lib_file_path}")

    def export_consolidated_csv(self, output_csv_path, state="B", sasa_filter=None):
        """
        Export consolidated rotamer library as a single CSV file.
        """
        cache_key = f"{state}_{sasa_filter or 'ALL'}"
        if cache_key not in self.results_cache:
            logging.error(f"No library cached for key {cache_key}. Call generate_library first.")
            return

        all_dfs = list(self.results_cache[cache_key].values())
        if not all_dfs:
            logging.warning("No rotamer library entries to export.")
            return

        combined_df = pd.concat(all_dfs, ignore_index=True)
        csv_file_path = Path(output_csv_path)
        csv_file_path.parent.mkdir(parents=True, exist_ok=True)

        combined_df.to_csv(csv_file_path, index=False)
        logging.info(f"Saved consolidated rotamer library CSV: {csv_file_path}")

    def export_master_lib(self, output_master_lib_path, state="B", sasa_filter=None):
        """
        Export a single merged master .lib file containing all amino acids standardized with 4 Chi angles.
        """
        cache_key = f"{state}_{sasa_filter or 'ALL'}"
        if cache_key not in self.results_cache:
            logging.error(f"No library cached for key {cache_key}. Call generate_library first.")
            return

        all_dfs = list(self.results_cache[cache_key].values())
        if not all_dfs:
            logging.warning("No rotamer library entries to export.")
            return

        out_path = Path(output_master_lib_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        expected_chi_cols = [f"{state}_CHI{i}" for i in range(1, 5)]
        expected_sig_cols = [f"{state}_CHI{i}Sig" for i in range(1, 5)]
        header = "#AA   PHI   PSI  _Count  Count   Prob    " + " ".join(expected_chi_cols) + " " + " ".join(expected_sig_cols) + "\n"
        formatter = "{: <3}  {:4d}  {:4d}  {:4d}   {:4d}  {:10.8f}" + "{:8.2f}" * 4 + "{:8.2f}" * 4

        lines_str = header
        for res_df in all_dfs:
            for _, row in res_df.iterrows():
                vals = [
                    str(row["AA"]),
                    int(row["PHI"]),
                    int(row["PSI"]),
                    int(row["_COUNT"]),
                    int(row["COUNT"]),
                    float(row["PROB"]),
                ]
                # Chi 1 to 4
                for i in range(1, 5):
                    chi_col = f"{state}_CHI{i}"
                    vals.append(float(row[chi_col]) if chi_col in row and pd.notna(row[chi_col]) else 0.0)
                # ChiSig 1 to 4
                for i in range(1, 5):
                    sig_col = f"{state}_CHI{i}Sig"
                    vals.append(float(row[sig_col]) if sig_col in row and pd.notna(row[sig_col]) else 0.0)

                lines_str += formatter.format(*vals) + "\n"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(lines_str)

        logging.info(f"Saved master merged rotamer library .lib file: {out_path}")

    @classmethod
    def merge_individual_lib_files(cls, input_dir, state="B", sasa_filter=None, output_master_lib=None, output_master_csv=None):
        """
        Merge individual amino acid .lib files from a directory into a single master .lib and/or .csv file.
        (Implements logic from merge_rotalib.ipynb).

        Parameters:
        -----------
        input_dir : Path or str
            Directory containing individual {AA}_bbd_{state}.lib files.
        state : str
            'B' for Bound or 'U' for Unbound state.
        sasa_filter : str, optional
            'I', 'N', or None.
        output_master_lib : Path or str, optional
            Output file path for merged master .lib file.
        output_master_csv : Path or str, optional
            Output file path for merged master .csv file.
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            logging.error(f"Input directory does not exist: {input_dir}")
            return None

        residues = list(cls.RESIDUE_CHI_COUNTS.keys())
        expected_cols = ["AA", "PHI", "PSI", "_Count", "Count", "Prob"] + \
                        [f"{state}_CHI{i}" for i in range(1, 5)] + \
                        [f"{state}_CHI{i}Sig" for i in range(1, 5)]

        sasa_suffix = f"_{sasa_filter}" if sasa_filter else ""
        df_list = []

        for aa in residues:
            file_name = f"{aa}_bbd_{state}{sasa_suffix}.lib"
            file_path = input_dir / file_name

            if not file_path.exists():
                # Fallback check without sasa_suffix
                file_path = input_dir / f"{aa}_bbd_{state}.lib"

            if not file_path.exists():
                logging.warning(f"Residue library file not found: {file_path}")
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    header_line = f.readline().strip().lstrip("#").strip()
                    columns = [col.strip() for col in header_line.split()]

                    if "AA" not in columns:
                        columns.insert(0, "AA")

                    df_res = pd.read_csv(
                        file_path,
                        sep=r"\s+",
                        comment="#",
                        names=columns,
                        skiprows=1,
                        dtype={col: float for col in columns if col != "AA"},
                        na_values=["", "NA", "NaN"],
                        keep_default_na=False,
                    )
                    df_res["AA"] = aa

                    for col in expected_cols:
                        if col not in df_res.columns:
                            df_res[col] = 0.0

                    df_res = df_res[expected_cols]
                    df_list.append(df_res)
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")

        if not df_list:
            logging.error("No valid residue library files were read.")
            return None

        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df = combined_df.convert_dtypes()
        num_cols = combined_df.select_dtypes(include="number").columns
        combined_df[num_cols] = combined_df[num_cols].fillna(0.0)

        # 1. Export master CSV if requested
        if output_master_csv:
            csv_path = Path(output_master_csv)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_csv(csv_path, index=False)
            logging.info(f"Successfully merged individual .lib files into master CSV: {csv_path}")

        # 2. Export master LIB if requested
        if output_master_lib:
            lib_path = Path(output_master_lib)
            lib_path.parent.mkdir(parents=True, exist_ok=True)
            header = "#AA   PHI   PSI  _Count  Count   Prob    " + " ".join([f"{state}_CHI{i}" for i in range(1, 5)]) + " " + " ".join([f"{state}_CHI{i}Sig" for i in range(1, 5)]) + "\n"
            formatter = "{: <3}  {:4d}  {:4d}  {:4d}   {:4d}  {:10.8f}" + "{:8.2f}" * 4 + "{:8.2f}" * 4

            lines_str = header
            for _, row in combined_df.iterrows():
                vals = [
                    str(row["AA"]),
                    int(row["PHI"]),
                    int(row["PSI"]),
                    int(row["_Count"]),
                    int(row["Count"]),
                    float(row["Prob"]),
                ]
                for i in range(1, 5):
                    vals.append(float(row[f"{state}_CHI{i}"]))
                for i in range(1, 5):
                    vals.append(float(row[f"{state}_CHI{i}Sig"]))

                lines_str += formatter.format(*vals) + "\n"

            with open(lib_path, "w", encoding="utf-8") as f:
                f.write(lines_str)

            logging.info(f"Successfully merged individual .lib files into master LIB: {lib_path}")

        return combined_df


def main():
    parser = argparse.ArgumentParser(description="Backbone-Dependent Rotamer Library Generation Engine.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=TORSION_DATASET,
        help="Path to dataset containing backbone & sidechain torsion angles.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "bbd_rotamer_libs", help="Output directory for .lib files.")
    parser.add_argument("--lib-csv-dir", type=Path, default=DEFAULT_LIB_DIR, help="Output directory for consolidated CSV libraries.")
    parser.add_argument("--state", type=str, default="both", choices=["B", "U", "both"], help="Protein state: 'B' for Bound, 'U' for Unbound, 'both' for both.")
    parser.add_argument("--sasa", type=str, default="all", choices=["all", "I", "N", "both_interface"], help="SASA interface filter.")
    parser.add_argument("--bb-sep", type=int, default=30, help="Backbone phi/psi bin separation step in degrees.")
    parser.add_argument("--merge-dir", type=Path, default=None, help="Directory containing individual .lib files to merge into master files.")

    args = parser.parse_args()

    # Handle standalone merging mode if --merge-dir is specified
    if args.merge_dir is not None:
        logging.info(f"Standalone merging mode: Merging .lib files from {args.merge_dir}")
        states_to_process = ["B", "U"] if args.state == "both" else [args.state]
        sasa_list = [None] if args.sasa == "all" else (["I", "N"] if args.sasa == "both_interface" else [args.sasa])

        for state in states_to_process:
            for sasa in sasa_list:
                sasa_prefix = f"_{sasa}" if sasa else ""
                master_lib = args.lib_csv_dir / f"RBPs{sasa_prefix}_bbd_{state}.lib"
                master_csv = args.lib_csv_dir / f"RBPs{sasa_prefix}_bbd_{state}_rotamer_lib.csv"

                BackboneDependentRotamerLibrary.merge_individual_lib_files(
                    input_dir=args.merge_dir,
                    state=state,
                    sasa_filter=sasa,
                    output_master_lib=master_lib,
                    output_master_csv=master_csv,
                )
        return

    if not args.input_file.exists():
        logging.error(f"Input file not found: {args.input_file}")
        return

    df = load_torsion_dataset(args.input_file)

    logging.info(f"Loaded dataset with {len(df)} residue entries.")

    rot_engine = BackboneDependentRotamerLibrary(bb_sep=args.bb_sep)

    states_to_process = ["B", "U"] if args.state == "both" else [args.state]
    sasa_list = [None] if args.sasa == "all" else (["I", "N"] if args.sasa == "both_interface" else [args.sasa])

    for state in states_to_process:
        for sasa in sasa_list:
            sasa_tag = sasa if sasa else "ALL"
            logging.info(f"--- Processing Rotamer Library Generation [State: {state}, SASA: {sasa_tag}] ---")

            rot_engine.generate_library(df, state=state, sasa_filter=sasa)

            # 1. Export individual .lib files
            out_lib_dir = args.output_dir / f"State_{state}_{sasa_tag}"
            rot_engine.export_lib_files(out_lib_dir, state=state, sasa_filter=sasa)

            # 2. Export consolidated master CSV file matching repository conventions
            sasa_prefix = f"_{sasa}" if sasa else ""
            csv_filename = f"RBPs{sasa_prefix}_bbd_{state}_rotamer_lib.csv"
            out_csv_path = args.lib_csv_dir / csv_filename
            rot_engine.export_consolidated_csv(out_csv_path, state=state, sasa_filter=sasa)

            # 3. Export merged master .lib file matching repository conventions
            master_lib_filename = f"RBPs{sasa_prefix}_bbd_{state}.lib"
            out_master_lib_path = args.lib_csv_dir / master_lib_filename
            rot_engine.export_master_lib(out_master_lib_path, state=state, sasa_filter=sasa)

    logging.info("Backbone-Dependent Rotamer Library Generation completed successfully.")


if __name__ == "__main__":
    main()

