#!/usr/bin/env python3
"""
energy_pvalue_plotter.py

Calculates statistical significance (independent two-sample t-test and paired t-test)
for side-chain energy distributions (Bound vs Unbound states) across Interface and Non-Interface
regions, and generates publication-quality boxplots and paired line plots.

Saves figures under figure/energy_pvalues/ and figures/energy_pvalues/.

Usage:
    python script/energy_pvalue_plotter.py [options]
"""

import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
RESULTS_DIR = ROOT_DIR / "output_files" / "energy_pvalues"
FIGURES_DIR = ROOT_DIR / "figure" / "energy_pvalues"
ALT_FIGURES_DIR = ROOT_DIR / "figures" / "energy_pvalues"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ALT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set global Matplotlib style
plt.style.use("tableau-colorblind10")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]


class EnergyPValuePlotter:
    """
    Statistical calculator and visualizer for side-chain energy distributions.
    """

    def __init__(self, figures_dir=FIGURES_DIR, alt_figures_dir=ALT_FIGURES_DIR, results_dir=RESULTS_DIR):
        self.figures_dir = Path(figures_dir)
        self.alt_figures_dir = Path(alt_figures_dir)
        self.results_dir = Path(results_dir)

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.alt_figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, data_path):
        """Load tab-separated energy data file."""
        p = Path(data_path)
        if not p.exists():
            logging.error(f"Data file not found: {p}")
            return None
        df = pd.read_csv(p, sep="\t")
        return df

    def compute_statistics(self, df):
        """
        Compute independent and paired t-test p-values.
        """
        if "Form" not in df.columns or "Avg.E" not in df.columns:
            logging.error("DataFrame missing required columns 'Form' or 'Avg.E'.")
            return None

        b_vals = df[df["Form"] == "B"]["Avg.E"].values
        u_vals = df[df["Form"] == "U"]["Avg.E"].values

        # Independent t-test
        ind_stat, ind_p = stats.ttest_ind(b_vals, u_vals, equal_var=False)

        # Paired t-test (if equal number of paired samples)
        paired_stat, paired_p = np.nan, np.nan
        if len(b_vals) == len(u_vals) and len(b_vals) > 0:
            paired_stat, paired_p = stats.ttest_rel(b_vals, u_vals)

        summary = {
            "N_Bound": len(b_vals),
            "Mean_Bound": float(np.mean(b_vals)) if len(b_vals) > 0 else np.nan,
            "Std_Bound": float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else np.nan,
            "N_Unbound": len(u_vals),
            "Mean_Unbound": float(np.mean(u_vals)) if len(u_vals) > 0 else np.nan,
            "Std_Unbound": float(np.std(u_vals, ddof=1)) if len(u_vals) > 1 else np.nan,
            "Ind_t_stat": float(ind_stat),
            "Ind_p_value": float(ind_p),
            "Paired_t_stat": float(paired_stat),
            "Paired_p_value": float(paired_p),
        }
        return summary

    def plot_boxplot(self, df, label="Interface", filename_stem="energy_distribution"):
        """
        Generate boxplot with jitter points and independent t-test p-value annotation.
        """
        summary = self.compute_statistics(df)
        if summary is None:
            return

        fig, ax = plt.subplots(figsize=(5, 5))

        colors = {"B": "#004225", "U": "#B22222"}

        if HAS_SEABORN:
            sns.boxplot(data=df, x="Form", y="Avg.E", hue="Form", palette=colors, ax=ax, width=0.4, boxprops=dict(alpha=0.6), legend=False)
            sns.stripplot(data=df, x="Form", y="Avg.E", hue="Form", palette=colors, alpha=0.6, jitter=0.2, size=5, ax=ax, legend=False)
        else:
            forms = df["Form"].unique()
            data_by_form = [df[df["Form"] == f]["Avg.E"].values for f in forms]
            ax.boxplot(data_by_form, labels=forms, patch_artist=True)

        ax.set_title(f"Energy Distribution ({label})", fontsize=14, color="red", fontweight="bold", style="italic")
        ax.set_xlabel(label, fontsize=16, fontweight="bold")
        ax.set_ylabel("E (Kcal/mol)", fontsize=16, fontweight="bold")

        # Annotate p-value using axes transAxes coordinates
        p_val = summary["Ind_p_value"]
        p_str = f"t-test, p = {p_val:.4f}" if p_val >= 0.0001 else f"t-test, p = {p_val:.2e}"
        ax.text(0.5, 0.92, p_str, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()

        out_path1 = self.figures_dir / f"{filename_stem}_boxplot.png"
        out_path2 = self.alt_figures_dir / f"{filename_stem}_boxplot.png"

        fig.savefig(out_path1, dpi=300, bbox_inches="tight")
        fig.savefig(out_path2, dpi=300, bbox_inches="tight")
        logging.info(f"Saved boxplot: {out_path1}")

        plt.close(fig)

    def plot_paired(self, df, label="Interface", filename_stem="energy_distribution"):
        """
        Generate paired line plot with paired t-test p-value annotation.
        """
        summary = self.compute_statistics(df)
        if summary is None:
            return

        fig, ax = plt.subplots(figsize=(5, 5))

        if "Res" in df.columns:
            pivot_df = df.pivot(index="Res", columns="Form", values="Avg.E").dropna()
            for res, row in pivot_df.iterrows():
                ax.plot(["B", "U"], [row["B"], row["U"]], color="gray", alpha=0.6, linewidth=1, zorder=1)
                ax.scatter(["B", "U"], [row["B"], row["U"]], c=["#004225", "#B22222"], s=30, zorder=2)
        else:
            b_vals = df[df["Form"] == "B"]["Avg.E"].values
            u_vals = df[df["Form"] == "U"]["Avg.E"].values
            for b, u in zip(b_vals, u_vals):
                ax.plot(["B", "U"], [b, u], color="gray", alpha=0.6, linewidth=1, zorder=1)
                ax.scatter(["B", "U"], [b, u], c=["#004225", "#B22222"], s=30, zorder=2)

        ax.set_title(f"Paired Energy Comparison ({label})", fontsize=14, color="red", fontweight="bold", style="italic")
        ax.set_xlabel(label, fontsize=16, fontweight="bold")
        ax.set_ylabel("E (Kcal/mol)", fontsize=16, fontweight="bold")

        # Annotate paired p-value
        p_val = summary["Paired_p_value"]
        if not np.isnan(p_val):
            p_str = f"paired t-test, p = {p_val:.4f}" if p_val >= 0.0001 else f"paired t-test, p = {p_val:.2e}"
            ax.text(0.5, 0.92, p_str, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()

        out_path1 = self.figures_dir / f"{filename_stem}_paired.png"
        out_path2 = self.alt_figures_dir / f"{filename_stem}_paired.png"

        fig.savefig(out_path1, dpi=300, bbox_inches="tight")
        fig.savefig(out_path2, dpi=300, bbox_inches="tight")
        logging.info(f"Saved paired plot: {out_path1}")

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Energy P-Value Calculator and Visualizer.")
    parser.add_argument("--data-file", type=Path, default=DATASET_DIR / "RBPs_BU_INT_187.dat", help="Path to input .dat energy file.")
    parser.add_argument("--label", type=str, default=None, help="Region label (e.g. Interface or Non-interface).")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Output figure directory.")

    args = parser.parse_args()

    plotter = EnergyPValuePlotter(figures_dir=args.figures_dir)

    df = plotter.load_data(args.data_file)
    if df is not None:
        label = args.label
        if label is None:
            label = "Non-interface" if "NINT" in str(args.data_file).upper() else "Interface"

        stem = args.data_file.stem
        plotter.plot_boxplot(df, label=label, filename_stem=stem)
        plotter.plot_paired(df, label=label, filename_stem=stem)

        summary = plotter.compute_statistics(df)
        summary_df = pd.DataFrame([summary])
        out_csv = RESULTS_DIR / f"{stem}_stats_summary.csv"
        summary_df.to_csv(out_csv, index=False)
        logging.info(f"Exported statistical summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
