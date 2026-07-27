#!/usr/bin/env python3
"""
rotamer_transition_matrix.py

Class and engine to derive side-chain rotamer transition matrices from Unbound ('U')
to Bound ('B') states across Chi angles (Chi1, Chi2, Chi3, Chi4) and generate
publication-quality transition matrix plots.

Usage:
    python script/rotamer_transition_matrix.py [options]
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "input_files"
RESULTS_DIR = ROOT_DIR / "output_files" / "transition_matrices"
FIGURES_DIR = ROOT_DIR / "figure" / "transition_matrices"
ALT_FIGURES_DIR = ROOT_DIR / "figures" / "transition_matrices"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ALT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set Matplotlib style
plt.style.use("tableau-colorblind10")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


class RotamerTransitionMatrix:
    """
    Class to calculate and plot side-chain rotamer transition matrices between
    Unbound ('U') and Bound ('B') states for protein residues across Chi angles (1..4).
    """

    # Rotamer state symbols: T (trans/gg), p (g+), P (gt), t (t), M (tg), m (g-)
    CHI_STATES = ["T", "p", "P", "t", "M", "m"]

    # Target residues for each Chi angle depth
    CHI_DEPTH_RESIDUES = {
        1: ["CYS", "SER", "THR", "VAL"],
        2: ["ARG", "ASN", "ASP", "GLU", "GLN", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "TRP", "TYR"],
        3: ["ARG", "LYS", "GLU", "GLN", "MET"],
        4: ["ARG", "LYS"],
    }

    def __init__(self, df=None, chi_depth=1, sasa_filter=None, threshold=0.0):
        """
        Initialize the RotamerTransitionMatrix calculator.

        Parameters:
        -----------
        df : pd.DataFrame, optional
            Dataset containing backbone and side-chain torsion angles.
        chi_depth : int
            Depth of Chi angles (1, 2, 3, or 4).
        sasa_filter : str, optional
            'I' for Interface, 'N' for Non-interface, or None for all.
        threshold : float
            Minimum row transition probability threshold to filter empty states.
        """
        self.df = df
        self.chi_depth = chi_depth
        self.sasa_filter = sasa_filter
        self.threshold = threshold

    @staticmethod
    def classify_chi_state(val):
        """
        Classify a side-chain torsion angle value (in degrees) into a rotamer state symbol.

        Parameters:
        -----------
        val : float
            Dihedral angle in degrees (-180 to 180).

        Returns:
        --------
        str : Rotamer state symbol ('T', 'p', 'P', 't', 'M', 'm').
        """
        if pd.isna(val):
            return "T"
        val = float(val)
        if -30.0 < val <= 30.0:
            return "T"  # Trans / gg (around 0°)
        elif 30.0 < val <= 90.0:
            return "p"  # g+ (around 60°)
        elif 90.0 < val <= 150.0:
            return "P"  # gt (around 120°)
        elif val > 150.0 or val <= -150.0:
            return "t"  # trans (around 180°)
        elif -150.0 < val <= -90.0:
            return "M"  # tg (around -120°)
        elif -90.0 < val <= -30.0:
            return "m"  # g- (around -60°)
        return "T"

    def _generate_state_combinations(self, depth):
        """Generate all possible rotamer state combination strings for a given Chi depth."""
        import itertools

        return ["".join(p) for p in itertools.product(self.CHI_STATES, repeat=depth)]

    def compute_transition_matrix(self, df, residue, chi_depth=None, sasa_filter=None, threshold=None):
        """
        Compute the transition count and probability matrices from Unbound ('U') to Bound ('B') states.

        Parameters:
        -----------
        df : pd.DataFrame
            Torsion angle dataset.
        residue : str
            Three-letter amino acid code (e.g. 'CYS', 'ARG').
        chi_depth : int, optional
            Chi angle depth (1, 2, 3, or 4). Defaults to self.chi_depth.
        sasa_filter : str, optional
            'I', 'N', or None. Defaults to self.sasa_filter.
        threshold : float, optional
            Filtering threshold for transition probability.

        Returns:
        --------
        dict containing:
            'counts_df': pd.DataFrame (Unbound states as rows, Bound states as columns)
            'prob_df': pd.DataFrame (Row-normalized transition probabilities)
            'unbound_states': list of str
            'bound_states': list of str
            'total_count': int
        """
        depth = chi_depth if chi_depth is not None else self.chi_depth
        sasa = sasa_filter if sasa_filter is not None else self.sasa_filter
        thresh = threshold if threshold is not None else self.threshold

        # Filter dataset for residue and SASA condition
        df_res = df[df["LABEL"].str[:3] == residue].copy()
        if sasa:
            df_res = df_res[df_res["SASA"].astype(str).str.contains(sasa)]

        if len(df_res) == 0:
            logging.warning(f"No data entries for residue {residue} with SASA filter {sasa}.")
            return None

        # Build list of Chi angle column names
        u_cols = [f"U_CHI{i}" for i in range(1, depth + 1)]
        b_cols = [f"B_CHI{i}" for i in range(1, depth + 1)]

        # Check required columns
        for c in u_cols + b_cols:
            if c not in df_res.columns:
                logging.error(f"Required column '{c}' not found in dataset for residue {residue}.")
                return None

        # Classify states row by row
        u_states = []
        b_states = []
        for _, row in df_res.iterrows():
            u_st = "".join([self.classify_chi_state(row[col]) for col in u_cols])
            b_st = "".join([self.classify_chi_state(row[col]) for col in b_cols])
            u_states.append(u_st)
            b_states.append(b_st)

        # Generate full possible state list or active state list
        all_possible_states = self._generate_state_combinations(depth)

        # Build raw count matrix
        counts_df = pd.DataFrame(0, index=all_possible_states, columns=all_possible_states)
        for u_st, b_st in zip(u_states, b_states):
            counts_df.loc[u_st, b_st] += 1

        # Remove rows and columns that are completely zero in the data
        active_u_states = counts_df.index[counts_df.sum(axis=1) > 0].tolist()
        active_b_states = counts_df.columns[counts_df.sum(axis=0) > 0].tolist()

        counts_filtered = counts_df.loc[active_u_states, active_b_states].copy()

        # Apply probability threshold filtering if requested (useful for Chi3 & Chi4)
        if thresh > 0.0 and len(counts_filtered) > 0:
            row_sums = counts_filtered.sum(axis=1)
            row_prob = counts_filtered.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

            keep_u = row_prob.index[(row_prob > thresh).any(axis=1)].tolist()
            keep_b = row_prob.columns[(row_prob > thresh).any(axis=0)].tolist()

            if keep_u and keep_b:
                counts_filtered = counts_filtered.loc[keep_u, keep_b]

        # Calculate row-normalized transition probabilities P(U -> B)
        row_totals = counts_filtered.sum(axis=1)
        prob_df = counts_filtered.div(row_totals.replace(0, np.nan), axis=0).fillna(0.0)

        return {
            "counts_df": counts_filtered,
            "prob_df": prob_df,
            "unbound_states": counts_filtered.index.tolist(),
            "bound_states": counts_filtered.columns.tolist(),
            "total_count": len(df_res),
        }

    def plot_transition_matrix(
        self,
        result_dict,
        residue,
        chi_depth=1,
        sasa_filter=None,
        save_path=None,
        cmap="Blues",
        annot=False,
    ):
        """
        Plot publication-quality heatmap of the Unbound -> Bound rotamer transition matrix.

        Parameters:
        -----------
        result_dict : dict
            Result dictionary returned by compute_transition_matrix.
        residue : str
            Residue three-letter code.
        chi_depth : int
            Chi angle depth (1, 2, 3, or 4).
        sasa_filter : str, optional
            'I', 'N', or 'ALL'.
        save_path : Path or str, optional
            Output image file path.
        cmap : str
            Matplotlib/Seaborn colormap name.
        annot : bool
            Whether to annotate matrix cells with numeric values.
        """
        if not result_dict or result_dict["prob_df"].empty:
            logging.warning(f"Empty transition matrix for {residue}. Skipping plot.")
            return

        prob_df = result_dict["prob_df"]
        unbound_states = result_dict["unbound_states"]
        bound_states = result_dict["bound_states"]

        # Determine dynamic plot size based on matrix dimensions
        n_rows = len(unbound_states)
        n_cols = len(bound_states)

        if chi_depth == 1:
            figsize = (7, 6)
        elif chi_depth == 2:
            figsize = (10, 8)
        elif chi_depth == 3:
            figsize = (14, 12)
        else:
            figsize = (16, 14)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(
            prob_df,
            cmap=cmap,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            annot=annot,
            fmt=".2f" if annot else "",
            linewidths=0.5 if n_rows <= 36 else 0.1,
            xticklabels=bound_states,
            yticklabels=unbound_states,
            cbar_kws={"label": r"Transition Probability P(Unbound $\rightarrow$ Bound)"},
            square=True,
        )

        # Format tick labels
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8 if n_cols > 36 else 10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8 if n_rows > 36 else 10)

        sasa_str = f" ({'Interface' if sasa_filter == 'I' else ('Non-Interface' if sasa_filter == 'N' else 'All')})"
        ax.set_title(f"{residue} $\chi_{chi_depth}$ Rotamer Transition Matrix{sasa_str}", fontsize=14, fontweight="bold", pad=12)

        ax.set_xlabel("Bound Rotamer States", fontsize=12, fontweight="bold")
        ax.set_ylabel("Unbound Rotamer States", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logging.info(f"Saved transition matrix plot to: {save_path}")

        plt.close(fig)

    def compute_and_save_all(
        self,
        df,
        chi_depth=1,
        sasa_filter="N",
        output_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
    ):
        """
        Compute transition matrices and save CSVs and plots for all valid residues at a given Chi depth.

        Parameters:
        -----------
        df : pd.DataFrame
            Torsion angle dataset.
        chi_depth : int
            Chi angle depth (1, 2, 3, or 4).
        sasa_filter : str
            'I', 'N', or None.
        output_dir : Path or str
            Output directory for CSV data.
        figures_dir : Path or str
            Output directory for plot images.
        """
        output_dir = Path(output_dir)
        figures_dir = Path(figures_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        residues = self.CHI_DEPTH_RESIDUES.get(chi_depth, [])
        sasa_tag = sasa_filter if sasa_filter else "ALL"

        logging.info(f"Computing Chi{chi_depth} transition matrices for residues {residues} (SASA: {sasa_tag})...")

        for res in residues:
            res_dict = self.compute_transition_matrix(df, residue=res, chi_depth=chi_depth, sasa_filter=sasa_filter)

            if not res_dict:
                continue

            # 1. Save CSV of probabilities
            prob_csv = output_dir / f"{res}_chi{chi_depth}_{sasa_tag}_tm_prob.csv"
            res_dict["prob_df"].to_csv(prob_csv)
            logging.info(f"Saved transition probability CSV: {prob_csv}")

            # 2. Save CSV of raw counts
            counts_csv = output_dir / f"{res}_chi{chi_depth}_{sasa_tag}_tm_counts.csv"
            res_dict["counts_df"].to_csv(counts_csv)

            # 3. Generate and save plot
            plot_png = figures_dir / f"{res}_chi{chi_depth}_{sasa_tag}_tm.png"
            self.plot_transition_matrix(
                res_dict,
                residue=res,
                chi_depth=chi_depth,
                sasa_filter=sasa_filter,
                save_path=plot_png,
                cmap="gray_r" if chi_depth >= 3 else "Blues",
            )
            # Also save to secondary figures directory if available
            alt_png = ALT_FIGURES_DIR / f"{res}_chi{chi_depth}_{sasa_tag}_tm.png"
            if alt_png.parent != figures_dir:
                self.plot_transition_matrix(
                    res_dict,
                    residue=res,
                    chi_depth=chi_depth,
                    sasa_filter=sasa_filter,
                    save_path=alt_png,
                    cmap="gray_r" if chi_depth >= 3 else "Blues",
                )


def main():
    parser = argparse.ArgumentParser(description="Side-Chain Rotamer Transition Matrix Engine (Unbound -> Bound).")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DATASET_DIR / "PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv",
        help="Path to dataset containing backbone & sidechain torsion angles.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for transition matrix CSV files.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Output directory for figure plots.")
    parser.add_argument("--chi-depth", type=str, default="all", choices=["1", "2", "3", "4", "all"], help="Chi angle depth (1..4 or 'all').")
    parser.add_argument("--sasa", type=str, default="both", choices=["I", "N", "all", "both"], help="SASA interface filter.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Probability threshold for filtering rare rotamer states.")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_file.exists():
        logging.error(f"Input file not found: {args.input_file}")
        return

    # Load dataset
    try:
        df = pd.read_csv(args.input_file, sep=",")
        if "LABEL" not in df.columns:
            df = pd.read_csv(args.input_file, sep="\t")
    except Exception:
        df = pd.read_csv(args.input_file, sep="\t")

    logging.info(f"Loaded dataset with {len(df)} residue entries.")

    tm_engine = RotamerTransitionMatrix(df=df, threshold=args.threshold)

    depths_to_process = [1, 2, 3, 4] if args.chi_depth == "all" else [int(args.chi_depth)]
    sasa_list = [None] if args.sasa == "all" else (["I", "N"] if args.sasa == "both" else [args.sasa])

    for depth in depths_to_process:
        for sasa in sasa_list:
            tm_engine.compute_and_save_all(
                df,
                chi_depth=depth,
                sasa_filter=sasa,
                output_dir=args.output_dir,
                figures_dir=args.figures_dir,
            )

    logging.info("Rotamer Transition Matrix computation and plotting completed successfully.")


if __name__ == "__main__":
    main()
