#!/usr/bin/env python3
"""
rotamer_plot.py

Unified Master Rotamer Visualization Engine (RotamerPlot).
Consolidates all rotamer plotting methods into a single class:
  1. plot_polar_histogram: Polar projection circular histograms of side-chain Chi angles (Chi1..Chi5)
     with rotamer region shading and percentage annotations.
  2. plot_stacked_distribution: 100% stacked bar plots comparing Bound ('B') vs Unbound ('U') rotamer distributions.
  3. plot_transition_matrix: Unbound -> Bound rotamer transition matrix heatmaps.
  4. plot_sidechain_kde: Linear side-chain Chi angle distributions with KDE overlays.

Usage:
    python script/rotamer_plot.py --type <polar|stacked|transition|kde|all> [options]
"""

import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "input_files"
RESULTS_DIR = ROOT_DIR / "output_files"
FIGURES_DIR = ROOT_DIR / "figure"
ALT_FIGURES_DIR = ROOT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ALT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib styling
plt.style.use("tableau-colorblind10")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


class RotamerPlot:
    """
    Unified visualization engine for protein side-chain rotamer analysis.
    """

    # Rotamer state colors, the same as the stacked distribution plots
    ROTAMER_COLORS = {
        "T": "#004225", "p": "#B22222", "P": "#FFBF00",
        "t": "#30BFBF", "M": "#FF4500", "m": "#663399",
    }
    # Centre angle (degrees) of each 60-degree state sector
    STATE_CENTERS = {"T": 0.0, "p": 60.0, "P": 120.0, "t": 180.0, "M": -120.0, "m": -60.0}

    # Maximum Chi angle depth per residue
    RESIDUE_CHI_MAP = {
        "ARG": 5, "LYS": 4, "GLU": 3, "GLN": 3, "MET": 3,
        "ASP": 2, "ASN": 2, "HIS": 2, "LEU": 2, "ILE": 2,
        "PHE": 2, "PRO": 2, "TRP": 2, "TYR": 2,
        "CYS": 1, "SER": 1, "THR": 1, "VAL": 1,
    }

    def __init__(self, dataset_file=None, excel_file=None):
        """
        Initialize RotamerPlot visualizer.

        Parameters:
        -----------
        dataset_file : Path or str, optional
            Path to main torsion angle dataset TSV/CSV.
        excel_file : Path or str, optional
            Path to Figure_2_Tabulated_187.xlsx dataset.
        """
        # Cleaned 187-complex dataset (comma-separated; missing angles empty, not 0.0)
        self.dataset_file = Path(dataset_file) if dataset_file else DATASET_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"

        # Stacked plots compute their counts from the torsion dataset unless an Excel file is given
        self.excel_file = Path(excel_file) if excel_file else None
        self._stacked_plotter = None
        self._df_cache = None

    def load_dataset(self):
        """Load and cache the main torsion angle dataset."""
        if self._df_cache is not None:
            return self._df_cache

        if not self.dataset_file.exists():
            logging.error(f"Dataset file not found: {self.dataset_file}")
            return None

        try:
            df = pd.read_csv(self.dataset_file, sep=",")
            if "LABEL" not in df.columns:
                df = pd.read_csv(self.dataset_file, sep="\t")
        except Exception:
            df = pd.read_csv(self.dataset_file, sep="\t")

        self._df_cache = df
        return df

    # =========================================================================
    # 1. POLAR HISTOGRAM PLOTS
    # =========================================================================
    @staticmethod
    def calculate_polar_histogram(data_radians, nbins=36):
        """Calculate polar histogram counts and bin centers."""
        if len(data_radians) == 0:
            bins = np.linspace(-np.pi, np.pi, nbins + 1)
            centers = (bins[:-1] + bins[1:]) / 2.0
            return np.zeros(nbins), np.zeros(nbins), centers

        bins = np.linspace(-np.pi, np.pi, nbins + 1)
        hist, _ = np.histogram(data_radians, bins=bins)
        max_h = np.max(np.sqrt(hist)) if np.max(hist) > 0 else 1.0
        sqrt_hist = np.sqrt(hist) / max_h
        pct_hist = (hist / np.sum(hist)) * 100.0 if np.sum(hist) > 0 else np.zeros(nbins)
        centers = (bins[:-1] + bins[1:]) / 2.0
        return sqrt_hist, pct_hist, centers

    @staticmethod
    def calculate_rotamer_percentages(chi_degrees):
        """Percentage of angles in each of the six states (boundaries as in classify_chi_state)."""
        from rotamer_distribution_plotter import RBPRotamerDistributionPlotter

        states = pd.Series([RBPRotamerDistributionPlotter.classify_chi_state(a) for a in chi_degrees]).dropna()
        if states.empty:
            return {k: 0.0 for k in "ptmPTM"}
        pct = states.value_counts(normalize=True) * 100.0
        return {k: round(float(pct.get(k, 0.0)), 1) for k in "ptmPTM"}

    def plot_polar_histogram(
        self,
        residue_list,
        chi_levels=None,
        state_prefix="B",
        save_path=None,
        title=None,
    ):
        """
        Plot multi-panel polar projection histograms for side-chain torsion angles.

        Parameters:
        -----------
        residue_list : list of str
            Amino acid three-letter codes (e.g. ['ASP', 'ASN'] or ['PHE', 'TYR', 'TRP']).
        chi_levels : list of int, optional
            List of Chi angle levels to plot (e.g. [1, 2]). Defaults to max for residues.
        state_prefix : str
            'B' for Bound or 'U' for Unbound state angles.
        save_path : Path or str, optional
            Output image file path.
        title : str, optional
            Overall figure title.
        """
        df = self.load_dataset()
        if df is None:
            return

        ncols = len(residue_list)
        if chi_levels is None:
            max_c = max([self.RESIDUE_CHI_MAP.get(r, 1) for r in residue_list])
            chi_levels = list(range(1, min(max_c + 1, 4)))

        nrows = len(chi_levels)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(max(4, ncols * 3.6), max(4, nrows * 3.6)),
            subplot_kw={"projection": "polar"},
            gridspec_kw=dict(wspace=0.55, hspace=0.6),
        )

        if nrows == 1 and ncols == 1:
            axes_grid = np.array([[ax]])
        elif nrows == 1:
            axes_grid = np.array([ax])
        elif ncols == 1:
            axes_grid = np.array([[a] for a in ax])
        else:
            axes_grid = ax

        for j, resn in enumerate(residue_list):
            df_res = df[df["LABEL"].str[:3] == resn]

            for i, chi_i in enumerate(chi_levels):
                cur_ax = axes_grid[i, j]
                cur_ax.set_theta_zero_location("N")
                cur_ax.set_theta_direction(-1)
                cur_ax.yaxis.grid(False)
                cur_ax.set_yticks([])
                cur_ax.set_ylim(0, 1.05)
                cur_ax.spines["polar"].set_linewidth(0.8)

                if i == 0:
                    cur_ax.set_title(resn, pad=46, fontsize=12, fontweight="bold")
                if j == 0:
                    cur_ax.text(-0.42, 0.5, r"$\chi_{" + str(chi_i) + r"}$", transform=cur_ax.transAxes,
                                fontsize=14, ha="center", va="center")

                col_name = f"{state_prefix}_CHI{chi_i}"
                if col_name in df_res.columns and not df_res[col_name].dropna().empty:
                    chi_vals = df_res[col_name].dropna().values
                    chi_rad = np.radians(chi_vals)

                    sqrth, _, b = self.calculate_polar_histogram(chi_rad, nbins=36)

                    # One colour per state; the state of a 10-degree bin is that of its centre
                    from rotamer_distribution_plotter import RBPRotamerDistributionPlotter
                    bin_states = np.array([RBPRotamerDistributionPlotter.classify_chi_state(np.degrees(c)) for c in b])
                    for rot, color in self.ROTAMER_COLORS.items():
                        m = bin_states == rot
                        cur_ax.bar(b[m], sqrth[m], width=np.radians(10), color=color, edgecolor="white",
                                   linewidth=0.3, zorder=3)

                    # State boundaries
                    for b_ang in np.radians([30, 90, 150, -30, -90, -150]):
                        cur_ax.axvline(b_ang, color="#9a9a9a", linestyle="--", linewidth=0.7, zorder=2)

                    # Degree labels just outside the circle, state percentages further out
                    cur_ax.set_xticks(np.radians([0, 90, 180, 270]))  # in [0, 360): ticks outside it shrink the circle
                    cur_ax.set_xticklabels([r"$0^\circ$", r"$90^\circ$", r"$180^\circ$", r"$-90^\circ$"], fontsize=8, color="#555555")
                    cur_ax.tick_params(axis="x", pad=1)
                    cur_ax.xaxis.grid(False)

                    pcts = self.calculate_rotamer_percentages(chi_vals)
                    for rot, ang in self.STATE_CENTERS.items():
                        p = pcts.get(rot, 0.0)
                        if round(p) < 1:
                            continue
                        r = 1.45 if rot in ("T", "t") else 1.28
                        cur_ax.text(np.radians(ang), r, f"{rot} {p:.0f}%", fontsize=8.5, fontweight="bold",
                                    color=self.ROTAMER_COLORS[rot], ha="center", va="center", clip_on=False)

        fig.subplots_adjust(top=0.9, bottom=0.06, left=0.1, right=0.95, wspace=0.55, hspace=0.6)

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=600, bbox_inches="tight", facecolor="white")
            logging.info(f"Saved polar histogram plot: {out_p}")

        plt.close(fig)

    # =========================================================================
    # 2. STACKED ROTAMER DISTRIBUTION PLOTS
    # =========================================================================
    def plot_stacked_distribution(self, chi_level=1, sasa_type="int", save_path=None):
        """100% stacked B/U rotamer state bars; same figure as rotamer_distribution_plotter.py."""
        from rotamer_distribution_plotter import RBPRotamerDistributionPlotter

        if self._stacked_plotter is None:
            excel = self.excel_file if self.excel_file is not None and self.excel_file.exists() else None
            self._stacked_plotter = RBPRotamerDistributionPlotter(excel_file=excel)
        self._stacked_plotter.plot_stacked_distribution(chi_level=chi_level, sasa_type=sasa_type, save_path=save_path)

    # =========================================================================
    # 3. TRANSITION MATRIX HEATMAP PLOTS
    # =========================================================================
    def plot_transition_matrix(self, result_dict, residue, chi_depth=1, sasa_filter=None, save_path=None):
        """Plot heatmap for rotamer transition matrix."""
        if not HAS_SEABORN:
            logging.error("Seaborn is required for transition matrix plots.")
            return

        if not result_dict or result_dict["prob_df"].empty:
            return

        prob_df = result_dict["prob_df"]
        unbound_states = result_dict["unbound_states"]
        bound_states = result_dict["bound_states"]

        n_rows = len(unbound_states)
        n_cols = len(bound_states)

        figsize = (7, 6) if chi_depth == 1 else ((10, 8) if chi_depth == 2 else (14, 12))
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            prob_df,
            cmap="gray_r" if chi_depth >= 3 else "Blues",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            linewidths=0.5 if n_rows <= 36 else 0.1,
            xticklabels=bound_states,
            yticklabels=unbound_states,
            cbar_kws={"label": r"Transition Probability P(Unbound $\rightarrow$ Bound)"},
            square=True,
        )

        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8 if n_cols > 36 else 10)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8 if n_rows > 36 else 10)

        sasa_str = f" ({'Interface' if sasa_filter == 'I' else ('Non-Interface' if sasa_filter == 'N' else 'All')})"
        ax.set_title(f"{residue} $\chi_{chi_depth}$ Rotamer Transition Matrix{sasa_str}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Bound Rotamer States", fontsize=12, fontweight="bold")
        ax.set_ylabel("Unbound Rotamer States", fontsize=12, fontweight="bold")

        plt.tight_layout()

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")
            logging.info(f"Saved transition matrix plot: {out_p}")

        plt.close(fig)

    # =========================================================================
    # 4. MASTER PLOT GENERATOR
    # =========================================================================
    def generate_all_plots(self, figures_dir=FIGURES_DIR):
        """Generate all polar, stacked, and transition matrix plots."""
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        logging.info("--- Generating Polar Histogram Plots ---")
        polar_groups = [
            (["ASP", "ASN"], [1, 2], "ASP_ASN_polar.png"),
            (["PHE", "TYR", "TRP"], [1, 2], "Aromatic_polar.png"),
            (["GLU", "GLN"], [1, 2, 3], "GLU_GLN_polar.png"),
            (["PRO"], [1, 2], "PRO_polar.png"),
            (["CYS", "SER", "THR", "VAL"], [1], "Chi1_polar.png"),
        ]

        out_polar_dir = figures_dir / "polar_histograms"
        for res_list, chi_list, fn in polar_groups:
            self.plot_polar_histogram(res_list, chi_levels=chi_list, save_path=out_polar_dir / fn)
            # Mirror to secondary figures dir
            self.plot_polar_histogram(res_list, chi_levels=chi_list, save_path=ALT_FIGURES_DIR / "polar_histograms" / fn)

        logging.info("--- Generating Stacked Distribution Plots ---")
        out_stacked_dir = figures_dir / "rotamer_distribution"
        for chi in range(1, 6):
            for sasa in ["int", "nonint"]:
                fn = f"side_chain_chi{chi}_{sasa}.png"
                self.plot_stacked_distribution(chi_level=chi, sasa_type=sasa, save_path=out_stacked_dir / fn)

        logging.info("Master RotamerPlot generation completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Master Rotamer Visualization Engine (RotamerPlot).")
    parser.add_argument("--type", type=str, default="all", choices=["polar", "stacked", "transition", "all"], help="Plot type to generate.")
    parser.add_argument("--dataset-file", type=Path, default=None, help="Path to main torsion angle dataset.")
    parser.add_argument("--excel-file", type=Path, default=None, help="Path to Figure_2_Tabulated_187.xlsx dataset.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Output directory for plots.")

    args = parser.parse_args()

    plotter = RotamerPlot(dataset_file=args.dataset_file, excel_file=args.excel_file)

    if args.type == "polar":
        out_dir = args.figures_dir / "polar_histograms"
        plotter.plot_polar_histogram(["ASP", "ASN"], [1, 2], save_path=out_dir / "ASP_ASN_polar.png")
        plotter.plot_polar_histogram(["PHE", "TYR", "TRP"], [1, 2], save_path=out_dir / "Aromatic_polar.png")
        plotter.plot_polar_histogram(["GLU", "GLN"], [1, 2, 3], save_path=out_dir / "GLU_GLN_polar.png")
        plotter.plot_polar_histogram(["PRO"], [1, 2], save_path=out_dir / "PRO_polar.png")
    elif args.type == "stacked":
        out_dir = args.figures_dir / "rotamer_distribution"
        for chi in range(1, 6):
            for sasa in ["int", "nonint"]:
                plotter.plot_stacked_distribution(chi_level=chi, sasa_type=sasa, save_path=out_dir / f"side_chain_chi{chi}_{sasa}.png")
    elif args.type == "all":
        plotter.generate_all_plots(figures_dir=args.figures_dir)

    logging.info("RotamerPlot execution finished successfully.")


if __name__ == "__main__":
    main()
