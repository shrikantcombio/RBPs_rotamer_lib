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
from matplotlib.ticker import FuncFormatter, MaxNLocator
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

    # Rotamer colors for polar histograms & stacked distributions
    ROTAMER_COLORS = {
        "p": "#440154",  # Deep purple (g+)
        "t": "#21908c",  # Teal (trans)
        "m": "#de77ae",  # Magenta (g-)
        "P": "#9ecae1",  # Light blue-purple (off g+)
        "T": "#b8de6f",  # Light green (off trans)
        "M": "#f1a340",  # Light orange (off g-)
    }

    # Custom palette for 6 stacked rotamer states
    STACKED_PALETTE = ["#004225", "#B22222", "#FFBF00", "#30BFBF", "#FF4500", "#663399"]

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
        self.dataset_file = Path(dataset_file) if dataset_file else DATASET_DIR / "PROTEIN_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"
        if not self.dataset_file.exists():
            self.dataset_file = DATASET_DIR / "PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"

        self.excel_file = Path(excel_file) if excel_file else DATASET_DIR / "Figure_2_Tabulated_187.xlsx"
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
        """Calculate percentages for on-rotamers (p, t, m) and off-rotamers (P, T, M)."""
        if len(chi_degrees) == 0:
            return {"p": 0.0, "t": 0.0, "m": 0.0, "P": 0.0, "T": 0.0, "M": 0.0}

        angles = ((np.array(chi_degrees) + 180.0) % 360.0) - 180.0
        total = len(angles)

        mask_p = (angles > 30.0) & (angles <= 90.0)
        mask_t = (angles >= 150.0) | (angles <= -150.0)
        mask_m = (angles >= -90.0) & (angles < -30.0)

        mask_P = (angles > 90.0) & (angles <= 150.0)
        mask_T = (angles > -30.0) & (angles < 30.0)
        mask_M = (angles > -150.0) & (angles <= -90.0)

        return {
            "p": round(np.sum(mask_p) / total * 100.0, 1),
            "t": round(np.sum(mask_t) / total * 100.0, 1),
            "m": round(np.sum(mask_m) / total * 100.0, 1),
            "P": round(np.sum(mask_P) / total * 100.0, 1),
            "T": round(np.sum(mask_T) / total * 100.0, 1),
            "M": round(np.sum(mask_M) / total * 100.0, 1),
        }

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
            figsize=(max(4, ncols * 3.5), max(4, nrows * 3.0)),
            subplot_kw={"projection": "polar"},
            gridspec_kw=dict(wspace=0.35, hspace=0.35),
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
                cur_ax.set_ylim(0, None)

                if i == 0:
                    cur_ax.set_xlabel(resn, labelpad=20, fontsize=12, fontweight="bold")
                    cur_ax.xaxis.set_label_position("top")
                if j == 0:
                    cur_ax.set_ylabel(r"$\chi_{" + str(chi_i) + r"}$", fontsize=12, fontweight="bold", labelpad=25, style="italic")

                col_name = f"{state_prefix}_CHI{chi_i}"
                if col_name in df_res.columns and not df_res[col_name].dropna().empty:
                    chi_vals = df_res[col_name].dropna().values
                    chi_rad = np.radians(chi_vals)

                    sqrth, _, b = self.calculate_polar_histogram(chi_rad, nbins=36)

                    # Plot light gray base histogram
                    cur_ax.bar(b, sqrth, width=np.radians(10), alpha=0.3, color="lightgray", zorder=1)

                    # Rotamer masks
                    mask_p = (b > np.pi / 6) & (b <= np.pi / 2)
                    mask_t = (b >= 5 * np.pi / 6) | (b <= -5 * np.pi / 6)
                    mask_m = (b >= -np.pi / 2) & (b < -np.pi / 6)

                    mask_P_off = (b > np.pi / 2) & (b <= 5 * np.pi / 6)
                    mask_T_off = (b > -np.pi / 6) & (b < np.pi / 6)
                    mask_M_off = (b > -5 * np.pi / 6) & (b <= -np.pi / 2)

                    # Bars for on-rotamers
                    cur_ax.bar(b[mask_p], sqrth[mask_p], width=np.radians(10), alpha=0.9, color=self.ROTAMER_COLORS["p"], zorder=3)
                    cur_ax.bar(b[mask_t], sqrth[mask_t], width=np.radians(10), alpha=0.9, color=self.ROTAMER_COLORS["t"], zorder=3)
                    cur_ax.bar(b[mask_m], sqrth[mask_m], width=np.radians(10), alpha=0.9, color=self.ROTAMER_COLORS["m"], zorder=3)

                    # Bars for off-rotamers
                    cur_ax.bar(b[mask_P_off], sqrth[mask_P_off], width=np.radians(10), alpha=1.0, color=self.ROTAMER_COLORS["P"], zorder=4)
                    cur_ax.bar(b[mask_T_off], sqrth[mask_T_off], width=np.radians(10), alpha=1.0, color=self.ROTAMER_COLORS["T"], zorder=4)
                    cur_ax.bar(b[mask_M_off], sqrth[mask_M_off], width=np.radians(10), alpha=1.0, color=self.ROTAMER_COLORS["M"], zorder=4)

                    max_r = np.max(sqrth) if len(sqrth) > 0 else 1.0
                    pcts = self.calculate_rotamer_percentages(chi_vals)

                    # Annotations
                    annot_on = {"p": (np.radians(60), max_r * 0.7), "t": (np.pi, max_r * 0.7), "m": (np.radians(-60), max_r * 0.7)}
                    annot_off = {"P": (np.radians(120), max_r * 0.6), "T": (np.radians(-5), max_r * 0.6), "M": (np.radians(-120), max_r * 0.6)}

                    for rot, (ang, rad) in annot_on.items():
                        p = pcts.get(rot, 0)
                        if p > 1.0:
                            cur_ax.annotate(f"{rot} {p:.0f}%", fontsize=8, fontweight="bold", xy=(ang, rad), ha="center", va="center")

                    for rot, (ang, rad) in annot_off.items():
                        p = pcts.get(rot, 0)
                        if p > 0.5:
                            cur_ax.annotate(f"{rot} {p:.0f}%", fontsize=8, fontweight="bold", xy=(ang, rad), ha="center", va="center")

                    # Angle ticks
                    tick_locs = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
                    tick_labs = [r"$0^\circ$", r"$90^\circ$", r"$180^\circ$", r"$-90^\circ$"]
                    cur_ax.set_xticks(tick_locs)
                    cur_ax.set_xticklabels(tick_labs, fontsize=8)

                    # Boundary lines
                    for b_ang in [np.pi / 6, np.pi / 2, 5 * np.pi / 6, -np.pi / 6, -np.pi / 2, -5 * np.pi / 6]:
                        cur_ax.axvline(b_ang, color="darkgray", linestyle="--", alpha=0.7)

        fig.subplots_adjust(top=0.92, bottom=0.08, wspace=0.35, hspace=0.35)

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight", facecolor="white")
            logging.info(f"Saved polar histogram plot: {out_p}")

        plt.close(fig)

    # =========================================================================
    # 2. STACKED ROTAMER DISTRIBUTION PLOTS
    # =========================================================================
    def plot_stacked_distribution(self, chi_level=1, sasa_type="int", save_path=None):
        """Plot 100% stacked bar chart for rotamer state distributions."""
        excel_path = self.excel_file
        if not excel_path.exists():
            logging.error(f"Excel file not found: {excel_path}")
            return

        sheet_name = f"chi{chi_level}_{sasa_type}"
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
        except Exception as e:
            logging.error(f"Could not load sheet {sheet_name}: {e}")
            return

        rot_cols = ["T", "p", "P", "t", "M", "m"]
        need_cols = ["Amino_acid", "forms"] + rot_cols
        df_sub = df[need_cols].copy()
        df_sub["Amino_acid"] = df_sub["Amino_acid"].astype(str).str.replace(r"_[IN]$", "", regex=True)

        df_grouped = df_sub.groupby(["Amino_acid", "forms"]).sum(numeric_only=True)
        row_totals = df_grouped.sum(axis=1)
        df_percent = df_grouped.div(row_totals.replace(0, np.nan), axis=0) * 100.0

        group_labels = list(df_percent.index.get_level_values(0).unique())
        bound_unbound_labels = ["B", "U"]
        aa_count = len(group_labels)

        xticks_major = [i * 2 + 0.5 for i in range(aa_count)]
        xticks_minor = [i for i in range(aa_count * 2)]

        fig, ax = plt.subplots(figsize=(max(8, aa_count * 0.75), 6))
        bar_width = 0.8
        bottoms = [0.0] * len(df_percent)

        for i, col in enumerate(rot_cols):
            vals = df_percent[col].values
            ax.bar(range(len(df_percent)), vals, bar_width, label=col, bottom=bottoms, color=self.STACKED_PALETTE[i], zorder=2)
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax.set_xticks(xticks_major)
        ax.set_xticklabels(group_labels, fontsize=12, fontweight="bold", rotation=0, ha="center")

        for i, tick in enumerate(xticks_minor):
            label = bound_unbound_labels[i % 2]
            ax.text(tick, -7.0, label, ha="center", va="top", fontsize=11, color="black", rotation=0)

        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}%"))
        ax.set_ylabel("Rotamer Percentage (%)", fontsize=13, fontweight="bold")

        sasa_title = "Interface" if sasa_type == "int" else "Non-Interface"
        ax.set_title(f"$\chi_{chi_level}$ Rotamer Distribution ({sasa_title} Residues)", fontsize=14, fontweight="bold", pad=15)
        ax.legend(ncols=6, loc="upper right", bbox_to_anchor=(1.0, 1.14), fancybox=True, shadow=True, fontsize=11)
        ax.grid(axis="y", linestyle="-", alpha=0.7, zorder=0)

        plt.tight_layout()

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")
            logging.info(f"Saved stacked distribution plot: {out_p}")

        plt.close(fig)

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
