#!/usr/bin/env python3
"""
rotamer_distribution_plotter.py

Refactored script to calculate and plot publication-quality side-chain rotamer angle
distributions (Chi1, Chi2, Chi3, Chi4, Chi5) comparing Bound ('B') and Unbound ('U')
protein states for Interface ('int') and Non-Interface ('nonint') regions.

Usage:
    python script/rotamer_distribution_plotter.py [options]
"""

import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "input_files"
RESULTS_DIR = ROOT_DIR / "output_files" / "rotamer_distribution"
FIGURES_DIR = ROOT_DIR / "figure" / "rotamer_distribution"
ALT_FIGURES_DIR = ROOT_DIR / "figures" / "rotamer_distribution"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ALT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib styling
plt.style.use("tableau-colorblind10")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


class RBPRotamerDistributionPlotter:
    """
    Plotter and analyzer for side-chain rotamer angle distributions across Chi angles (1..5)
    and SASA partitions (Interface vs Non-Interface).
    """

    # Custom color palette for 6 rotamer states: T, p, P, t, M, m
    # T (trans/gg), p (g+), P (gt), t (trans), M (tg), m (g-)
    ROTAMER_COLUMNS = ["T", "p", "P", "t", "M", "m"]
    COLOR_PALETTE = ["#004225", "#B22222", "#FFBF00", "#30BFBF", "#FF4500", "#663399"]

    def __init__(self, excel_file=None):
        """
        Initialize RBPRotamerDistributionPlotter.

        Parameters:
        -----------
        excel_file : Path or str, optional
            Path to Figure_2_Tabulated_187.xlsx dataset.
        """
        self.excel_file = Path(excel_file) if excel_file else DATASET_DIR / "Figure_2_Tabulated_187.xlsx"

    @staticmethod
    def classify_chi_state(angle_deg):
        """
        Classify side-chain torsion angle (degrees) into discrete 6-rotamer states:
          - 'T': trans/gg (-30, 30]
          - 'p': g+ (30, 90]
          - 'P': gt (90, 150]
          - 't': trans (>150 or <= -150)
          - 'M': tg (-150, -90]
          - 'm': g- (-90, -30]
        """
        if pd.isna(angle_deg):
            return None
        val = float(angle_deg)
        val = ((val + 180.0) % 360.0) - 180.0
        if -30.0 < val <= 30.0:
            return "T"
        elif 30.0 < val <= 90.0:
            return "p"
        elif 90.0 < val <= 150.0:
            return "P"
        elif val > 150.0 or val <= -150.0:
            return "t"
        elif -150.0 < val <= -90.0:
            return "M"
        elif -90.0 < val <= -30.0:
            return "m"
        return None

    def calculate_rotamer_tables_from_tsv(self, tsv_file=None, output_excel=None):
        """
        Calculate rotamer state counts and percentage distributions directly from
        raw torsion angle TSV file (PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv).
        Recreates all sheets for Figure_2_Tabulated_187.xlsx.
        """
        tsv_path = Path(tsv_file) if tsv_file else DATASET_DIR / "PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"
        if not tsv_path.exists():
            tsv_path = DATASET_DIR / "PROTEIN_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"

        if not tsv_path.exists():
            logging.error(f"Raw dataset TSV file not found: {tsv_path}")
            return None

        try:
            df = pd.read_csv(tsv_path, sep=",")
            if "LABEL" not in df.columns:
                df = pd.read_csv(tsv_path, sep="\t")
        except Exception:
            df = pd.read_csv(tsv_path, sep="\t")

        df["Amino_acid"] = df["LABEL"].astype(str).str[:3]

        chi_map = {
            1: ["ARG", "ASN", "ASP", "GLU", "GLN", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "TRP", "TYR", "CYS", "SER", "THR", "VAL"],
            2: ["ARG", "ASN", "ASP", "GLU", "GLN", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "TRP", "TYR"],
            3: ["ARG", "LYS", "GLU", "GLN", "MET"],
            4: ["ARG", "LYS"],
            5: ["ARG"],
        }

        sheets_dict = {}

        for chi in range(1, 6):
            for sasa_code, sasa_label in [("I", "int"), ("N", "nonint")]:
                sheet_name = f"chi{chi}_{sasa_label}"
                sub_df = df[df["SASA"] == sasa_code]

                rows = []
                for aa in chi_map[chi]:
                    aa_df = sub_df[sub_df["Amino_acid"] == aa]

                    for form, prefix in [("B", "B_"), ("U", "U_")]:
                        col = f"{prefix}CHI{chi}"
                        if col in aa_df.columns:
                            states = aa_df[col].apply(self.classify_chi_state).dropna()
                            counts = states.value_counts().to_dict()
                            total = len(states)
                        else:
                            counts = {}
                            total = 0

                        aa_disp = aa
                        if chi in [4, 5]:
                            aa_disp = f"{aa}_I" if sasa_code == "I" else f"{aa}_N"

                        row_dict = {
                            "Amino_acid": aa_disp,
                            "forms": form,
                            "T": counts.get("T", 0),
                            "p": counts.get("p", 0),
                            "P": counts.get("P", 0),
                            "t": counts.get("t", 0),
                            "M": counts.get("M", 0),
                            "m": counts.get("m", 0),
                            "Total": total,
                        }
                        rows.append(row_dict)

                sheet_df = pd.DataFrame(rows)
                sheets_dict[sheet_name] = sheet_df

        if output_excel:
            out_excel = Path(output_excel)
            out_excel.parent.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(out_excel) as writer:
                for sheet_name, sheet_df in sheets_dict.items():
                    sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            logging.info(f"Saved tabulated rotamer distributions to Excel: {out_excel}")

        return sheets_dict

    def load_sheet(self, sheet_name):
        """
        Load a specific sheet from the Excel dataset.

        Parameters:
        -----------
        sheet_name : str
            Sheet name (e.g. 'chi1_int', 'chi1_nonint').

        Returns:
        --------
        pd.DataFrame or None
        """
        if not self.excel_file.exists():
            logging.error(f"Excel dataset file not found at {self.excel_file}")
            return None

        try:
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            return df
        except Exception as e:
            logging.error(f"Failed to read sheet '{sheet_name}' from {self.excel_file}: {e}")
            return None

    def plot_stacked_distribution(
        self,
        chi_level=1,
        sasa_type="int",
        save_path=None,
        figures_dir=FIGURES_DIR,
    ):
        """
        Generate 100% stacked bar plot comparing Bound ('B') and Unbound ('U') rotamer distributions.

        Parameters:
        -----------
        chi_level : int
            Chi angle level (1, 2, 3, 4, 5).
        sasa_type : str
            'int' (Interface) or 'nonint' (Non-Interface).
        save_path : Path or str, optional
            Specific file path to save plot image.
        figures_dir : Path or str
            Output figure directory.
        """
        sheet_name = f"chi{chi_level}_{sasa_type}"
        df = self.load_sheet(sheet_name)

        if df is None or df.empty:
            logging.warning(f"No data available for {sheet_name}. Skipping plot.")
            return

        need_cols = ["Amino_acid", "forms"] + self.ROTAMER_COLUMNS
        for c in need_cols:
            if c not in df.columns:
                logging.error(f"Missing column '{c}' in sheet {sheet_name}.")
                return

        df_sub = df[need_cols].copy()

        # Clean amino acid names (remove _I or _N suffix if present)
        df_sub["Amino_acid"] = df_sub["Amino_acid"].astype(str).str.replace(r"_[IN]$", "", regex=True)

        # Group by Amino_acid and forms ('B', 'U'), sum counts, and normalize to 100%
        df_grouped = df_sub.groupby(["Amino_acid", "forms"]).sum(numeric_only=True)
        row_totals = df_grouped.sum(axis=1)
        df_percent = df_grouped.div(row_totals.replace(0, np.nan), axis=0) * 100.0

        # Unique amino acids and form labels
        group_labels = list(df_percent.index.get_level_values(0).unique())
        bound_unbound_labels = ["B", "U"]
        amino_acid_count = len(group_labels)

        # Calculate tick positions for double-bars (B and U)
        xticks_major = [i * 2 + 0.5 for i in range(amino_acid_count)]
        xticks_minor = [i for i in range(amino_acid_count * 2)]

        fig, ax = plt.subplots(figsize=(max(8, amino_acid_count * 0.75), 6))
        bar_width = 0.8
        bottoms = [0.0] * len(df_percent)

        # Plot stacked bars for each rotamer state
        for i, col in enumerate(self.ROTAMER_COLUMNS):
            vals = df_percent[col].values
            ax.bar(
                range(len(df_percent)),
                vals,
                bar_width,
                label=col,
                bottom=bottoms,
                color=self.COLOR_PALETTE[i],
                zorder=2,
            )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        # Set major ticks (amino acid labels)
        ax.set_xticks(xticks_major)
        ax.set_xticklabels(group_labels, fontsize=12, fontweight="bold", rotation=0, ha="center")

        # Add minor text labels for 'B' and 'U' below each bar
        for i, tick in enumerate(xticks_minor):
            label = bound_unbound_labels[i % 2]
            ax.text(tick, -7.0, label, ha="center", va="top", fontsize=11, color="black", rotation=0)

        # Y-axis formatting
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}%"))
        plt.setp(ax.get_yticklabels(), fontsize=12)
        ax.set_ylabel("Rotamer Percentage (%)", fontsize=13, fontweight="bold")

        sasa_title = "Interface" if sasa_type == "int" else "Non-Interface"
        ax.set_title(f"$\chi_{chi_level}$ Rotamer Distribution ({sasa_title} Residues)", fontsize=14, fontweight="bold", pad=15)

        # Legend
        ax.legend(
            ncols=len(self.ROTAMER_COLUMNS),
            loc="upper right",
            bbox_to_anchor=(1.0, 1.14),
            fancybox=True,
            shadow=True,
            fontsize=11,
        )

        # Y-grid lines
        ax.grid(axis="y", linestyle="-", alpha=0.7, which="major", zorder=0)

        plt.tight_layout()

        # Determine file output path
        if save_path:
            out_path = Path(save_path)
        else:
            out_path = Path(figures_dir) / f"side_chain_chi{chi_level}_{sasa_type}.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        logging.info(f"Saved rotamer distribution plot: {out_path}")

        # Also save to secondary figures directory
        alt_path = ALT_FIGURES_DIR / f"side_chain_chi{chi_level}_{sasa_type}.png"
        if alt_path.parent != out_path.parent:
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(alt_path, bbox_inches="tight", dpi=300)

        plt.close(fig)

    def compute_and_export_summary(self, output_dir=RESULTS_DIR):
        """
        Export summary CSV tables of rotamer percentage distributions for all Chi levels.

        Parameters:
        -----------
        output_dir : Path or str
            Output CSV directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_summaries = []

        for chi_level in range(1, 6):
            for sasa_type in ["int", "nonint"]:
                sheet_name = f"chi{chi_level}_{sasa_type}"
                df = self.load_sheet(sheet_name)

                if df is None or df.empty:
                    continue

                need_cols = ["Amino_acid", "forms"] + self.ROTAMER_COLUMNS
                if not all(c in df.columns for c in need_cols):
                    continue

                df_sub = df[need_cols].copy()
                df_sub["Amino_acid"] = df_sub["Amino_acid"].astype(str).str.replace(r"_[IN]$", "", regex=True)

                df_grouped = df_sub.groupby(["Amino_acid", "forms"]).sum(numeric_only=True)
                row_totals = df_grouped.sum(axis=1)
                df_percent = df_grouped.div(row_totals.replace(0, np.nan), axis=0) * 100.0

                df_percent = df_percent.reset_index()
                df_percent.insert(0, "Chi_Level", f"Chi{chi_level}")
                df_percent.insert(1, "SASA", "Interface" if sasa_type == "int" else "Non-Interface")

                # Calculate On-rotamers (p, t, m) vs Off-rotamers (T, P, M)
                df_percent["On_Rotamer_Percent"] = df_percent[["p", "t", "m"]].sum(axis=1)
                df_percent["Off_Rotamer_Percent"] = df_percent[["T", "P", "M"]].sum(axis=1)

                all_summaries.append(df_percent)

        if all_summaries:
            combined_summary = pd.concat(all_summaries, ignore_index=True)
            summary_csv_path = output_dir / "rotamer_distribution_summary.csv"
            combined_summary.to_csv(summary_csv_path, index=False)
            logging.info(f"Saved rotamer distribution summary CSV: {summary_csv_path}")

    def plot_chi1_chi2_heatmap(
        self,
        tsv_file=None,
        state="B",
        save_path=None,
        figures_dir=FIGURES_DIR,
    ):
        """
        Generate publication-quality Chi1-Chi2 rotamer state heatmap with top percentage bar plot
        reproducing notebook/chi1_chi2_distribution.png.

        Parameters:
        -----------
        tsv_file : Path or str, optional
            Path to raw torsion angle dataset.
        state : str
            'B' for Bound state or 'U' for Unbound state.
        save_path : Path or str, optional
        figures_dir : Path or str
        """
        import matplotlib.gridspec as gridspec

        tsv_path = Path(tsv_file) if tsv_file else DATASET_DIR / "PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"
        if not tsv_path.exists():
            tsv_path = DATASET_DIR / "PROTEIN_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"

        if not tsv_path.exists():
            logging.error(f"Raw dataset TSV file not found: {tsv_path}")
            return

        try:
            df = pd.read_csv(tsv_path, sep=",")
            if "LABEL" not in df.columns:
                df = pd.read_csv(tsv_path, sep="\t")
        except Exception:
            df = pd.read_csv(tsv_path, sep="\t")

        df["Amino_acid"] = df["LABEL"].astype(str).str[:3]

        aa_chi2 = ["ARG", "ASN", "ASP", "GLU", "GLN", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "TRP", "TYR"]
        df_sub = df[df["Amino_acid"].isin(aa_chi2)].copy()

        prefix = f"{state}_"
        col_chi1 = f"{prefix}CHI1"
        col_chi2 = f"{prefix}CHI2"

        if col_chi1 not in df_sub.columns or col_chi2 not in df_sub.columns:
            logging.error(f"Columns {col_chi1} or {col_chi2} not found in dataset.")
            return

        df_sub["chi1_st"] = df_sub[col_chi1].apply(self.classify_chi_state)
        df_sub["chi2_st"] = df_sub[col_chi2].apply(self.classify_chi_state)
        df_sub["chi12"] = df_sub["chi1_st"] + df_sub["chi2_st"]

        df_valid = df_sub.dropna(subset=["chi12"]).copy()

        # Calculate overall state percentages
        overall_pct = (df_valid["chi12"].value_counts(normalize=True) * 100.0).round(1)

        # Filter states with percentage >= 0.1%
        sorted_states = overall_pct[overall_pct >= 0.1].index.tolist()

        # Compute Matrix (Amino Acids x Chi1Chi2 States) in percentage
        matrix_rows = []
        for aa in aa_chi2:
            aa_df = df_valid[df_valid["Amino_acid"] == aa]
            total_aa = len(aa_df)
            if total_aa > 0:
                counts = aa_df["chi12"].value_counts().to_dict()
                row = [(counts.get(st, 0) / total_aa) * 100.0 for st in sorted_states]
            else:
                row = [0.0] * len(sorted_states)
            matrix_rows.append(row)

        matrix = np.array(matrix_rows)

        fig = plt.figure(figsize=(14, 9))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 4, 0.4], hspace=0.15)

        ax_bar = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])
        ax_cbar = fig.add_subplot(gs[2])

        # 1. Top Bar Chart
        bar_vals = [overall_pct[st] for st in sorted_states]
        bars = ax_bar.bar(range(len(sorted_states)), bar_vals, color="#87ceeb", width=0.8, edgecolor="none")

        for idx, val in enumerate(bar_vals):
            ax_bar.text(idx, val + 0.8, f"{val:.1f}", ha="center", va="bottom", fontsize=11)

        ax_bar.set_xlim(-0.6, len(sorted_states) - 0.4)
        ax_bar.set_ylim(0, max(bar_vals) * 1.25)
        ax_bar.set_ylabel("Percentage (%)", fontsize=14)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # 2. Bottom Heatmap
        im = ax_heat.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=max(55.0, matrix.max()))

        ax_heat.set_xticks(range(len(sorted_states)))
        ax_heat.set_xticklabels(sorted_states, fontsize=13, rotation=90, ha="center")

        ax_heat.set_yticks(range(len(aa_chi2)))
        ax_heat.set_yticklabels(aa_chi2, fontsize=13)
        ax_heat.set_ylabel("Amino acids", fontsize=14, labelpad=10)
        ax_heat.set_xlabel(r"$\chi_1\chi_2$ states", fontsize=15, labelpad=10)

        # 3. Colorbar at bottom
        cbar = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
        cbar.set_label("Percentage", fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        # Determine file output path
        if save_path:
            out_path = Path(save_path)
        else:
            out_path = Path(figures_dir) / "chi1_chi2_distribution.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=600, bbox_inches="tight")
        logging.info(f"Saved Chi1-Chi2 heatmap plot: {out_path}")

        alt_path = ALT_FIGURES_DIR / "chi1_chi2_distribution.png"
        if alt_path.parent != out_path.parent:
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(alt_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

    def plot_all(self, figures_dir=FIGURES_DIR, output_dir=RESULTS_DIR):
        """
        Generate all rotamer distribution plots and summary tables for Chi1 to Chi5,
        including the Chi1-Chi2 state heatmap.
        """
        for chi in range(1, 6):
            for sasa in ["int", "nonint"]:
                self.plot_stacked_distribution(chi_level=chi, sasa_type=sasa, figures_dir=figures_dir)

        self.plot_chi1_chi2_heatmap(figures_dir=figures_dir)
        self.compute_and_export_summary(output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Side-Chain Rotamer Angle Distribution Plotter.")
    parser.add_argument(
        "--excel-file",
        type=Path,
        default=DATASET_DIR / "Figure_2_Tabulated_187.xlsx",
        help="Path to Figure_2_Tabulated_187.xlsx dataset file.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for summary CSV tables.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Output directory for figure plots.")
    parser.add_argument("--chi-level", type=str, default="all", choices=["1", "2", "3", "4", "5", "all"], help="Chi angle level (1..5 or 'all').")
    parser.add_argument("--sasa", type=str, default="both", choices=["int", "nonint", "both"], help="SASA partition filter ('int', 'nonint', 'both').")
    parser.add_argument("--plot-heatmap", action="store_true", help="Plot Chi1-Chi2 rotamer state heatmap.")

    args = parser.parse_args()

    plotter = RBPRotamerDistributionPlotter(excel_file=args.excel_file)

    if args.plot_heatmap:
        plotter.plot_chi1_chi2_heatmap(figures_dir=args.figures_dir)

    if args.chi_level == "all" and args.sasa == "both":
        plotter.plot_all(figures_dir=args.figures_dir, output_dir=args.output_dir)
    else:
        chi_list = [1, 2, 3, 4, 5] if args.chi_level == "all" else [int(args.chi_level)]
        sasa_list = ["int", "nonint"] if args.sasa == "both" else [args.sasa]

        for chi in chi_list:
            for sasa in sasa_list:
                plotter.plot_stacked_distribution(chi_level=chi, sasa_type=sasa, figures_dir=args.figures_dir)

        plotter.compute_and_export_summary(output_dir=args.output_dir)

    logging.info("Rotamer distribution plotting completed successfully.")


if __name__ == "__main__":
    main()

