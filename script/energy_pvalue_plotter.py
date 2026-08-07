#!/usr/bin/env python3
"""
energy_pvalue_plotter.py

Master statistical calculator and visualizer for side-chain energy distributions (Bound vs Unbound states).
Reads directly from JSON summary files to avoid CSV string/number conversion artifacts (e.g. '4E78').

Generates:
1. PDB-level total and average energy boxplots and paired line plots.
2. Amino Acid-wise paired two-tailed t-test energy comparison plots at Interface (I) and Non-Interface (N).
3. Dedicated energy comparison plots for Aromatic Amino Acids (HIS, PHE, TRP, TYR).

Saves 600 DPI publication plots under figure/energy_pvalues/ and figures/energy_pvalues/.

Usage:
    python script/energy_pvalue_plotter.py
"""

import json
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
OUTPUT_DIR = ROOT_DIR / "output_files"
RESULTS_DIR = OUTPUT_DIR / "energy_pvalues"
FIGURES_DIR = ROOT_DIR / "figure" / "energy_pvalues"
ALT_FIGURES_DIR = ROOT_DIR / "figures" / "energy_pvalues"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ALT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set global Matplotlib style
plt.style.use("tableau-colorblind10")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0


class EnergyPValuePlotter:
    def __init__(self, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR, alt_figures_dir=ALT_FIGURES_DIR):
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.alt_figures_dir = Path(alt_figures_dir)

    def load_json(self, filename):
        path = self.output_dir / filename
        if not path.exists():
            logging.error(f"JSON summary file not found: {path}")
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def plot_pdb_energy(self, df_pdb, region="INT", metric="AVG"):
        """Plot PDB-level energy distributions (Bound vs Unbound) for Interface or Non-interface."""
        if region == "INT":
            u_col = "IU_AVG_DELG" if metric == "AVG" else "IU_DELG"
            b_col = "IB_AVG_DELG" if metric == "AVG" else "IB_DELG"
        else:
            u_col = "NU_AVG_DELG" if metric == "AVG" else "NU_DELG"
            b_col = "NB_AVG_DELG" if metric == "AVG" else "NB_DELG"
        label = "Interface" if region == "INT" else "Non-interface"
        metric_str = "Average" if metric == "AVG" else "Total"

        u_vals = df_pdb[u_col].values
        b_vals = df_pdb[b_col].values
        res_ids = df_pdb["PDB_ID"].values

        # Two-tailed t-tests
        ind_stat, ind_p = stats.ttest_ind(b_vals, u_vals, equal_var=False)
        pair_stat, pair_p = stats.ttest_rel(b_vals, u_vals)

        # Build DataFrame for plot
        df_plot = pd.DataFrame({
            "PDB_ID": np.repeat(res_ids, 2),
            "Energy": np.concatenate([u_vals, b_vals]),
            "Form": np.repeat(["U", "B"], len(res_ids))
        })

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        colors = {"B": "#004225", "U": "#B22222"}

        if HAS_SEABORN:
            sns.boxplot(data=df_plot, x="Form", y="Energy", hue="Form", palette=colors, ax=ax, width=0.4, boxprops=dict(alpha=0.6), legend=False)
            sns.stripplot(data=df_plot, x="Form", y="Energy", hue="Form", palette=colors, alpha=0.6, jitter=0.2, size=4.5, ax=ax, legend=False)

        ax.set_title(f"{metric_str} Energy ({label})", fontsize=14, color="red", fontweight="bold", style="italic")
        ax.set_xlabel(label, fontsize=15, fontweight="bold")
        ax.set_ylabel("E (kcal/mol)", fontsize=15, fontweight="bold")

        p_str = f"paired t-test, p = {pair_p:.4f}" if pair_p >= 0.0001 else f"paired t-test, p = {pair_p:.2e}"
        ax.text(0.5, 0.92, p_str, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        plt.tight_layout()

        stem = f"pdb_{metric.lower()}_energy_{region.lower()}"
        out1 = self.figures_dir / f"{stem}_boxplot.png"
        out2 = self.alt_figures_dir / f"{stem}_boxplot.png"
        fig.savefig(out1, dpi=600, bbox_inches="tight")
        fig.savefig(out2, dpi=600, bbox_inches="tight")
        plt.close(fig)

        # Paired Line Plot
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        for u, b in zip(u_vals, b_vals):
            ax.plot(["U", "B"], [u, b], color="gray", alpha=0.5, linewidth=0.8, zorder=1)
            ax.scatter(["U", "B"], [u, b], c=["#B22222", "#004225"], s=25, zorder=2)

        ax.set_title(f"Paired {metric_str} Energy ({label})", fontsize=14, color="red", fontweight="bold", style="italic")
        ax.set_xlabel(label, fontsize=15, fontweight="bold")
        ax.set_ylabel("E (kcal/mol)", fontsize=15, fontweight="bold")
        ax.text(0.5, 0.92, p_str, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        plt.tight_layout()
        out1_p = self.figures_dir / f"{stem}_paired.png"
        out2_p = self.alt_figures_dir / f"{stem}_paired.png"
        fig.savefig(out1_p, dpi=600, bbox_inches="tight")
        fig.savefig(out2_p, dpi=600, bbox_inches="tight")
        plt.close(fig)

        logging.info(f"Generated PDB plots for {metric_str} {label}: paired p = {pair_p:.4e}")

    def plot_amino_acid_wise_energy(self, df_aa, aromatic_only=False, region="INT"):
        """Generate Amino Acid-wise paired energy comparison plots (Bound vs Unbound)."""
        df_work = df_aa.copy()
        if aromatic_only:
            df_work = df_work[df_work["AA"].isin(["HIS", "PHE", "TRP", "TYR"])]
            title_prefix = "Aromatic Amino Acid"
            stem_prefix = "aromatic_aa"
        else:
            title_prefix = "Amino Acid-Wise"
            stem_prefix = "amino_acid"

        u_col = "IU_AVG_DELG" if region == "INT" else "NU_AVG_DELG"
        b_col = "IB_AVG_DELG" if region == "INT" else "NB_AVG_DELG"
        region_label = "Interface" if region == "INT" else "Non-interface"

        aas = df_work["AA"].values
        u_vals = df_work[u_col].values
        b_vals = df_work[b_col].values

        # Perform paired t-test across amino acids
        pair_stat, pair_p = stats.ttest_rel(b_vals, u_vals)

        # Reshape into long DataFrame for grouped bar plot
        df_long = pd.DataFrame({
            "AA": np.repeat(aas, 2),
            "Energy": np.concatenate([u_vals, b_vals]),
            "State": np.repeat(["Unbound (U)", "Bound (B)"], len(aas))
        })

        fig, ax = plt.subplots(figsize=(10 if not aromatic_only else 6.5, 6))
        palette = {"Unbound (U)": "#B22222", "Bound (B)": "#004225"}

        if HAS_SEABORN:
            sns.barplot(data=df_long, x="AA", y="Energy", hue="State", palette=palette, ax=ax, edgecolor="black", linewidth=1.0)

        ax.set_title(f"{title_prefix} Energy Comparison ({region_label})\n(Paired t-test p = {pair_p:.4f})", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Amino Acid Type", fontsize=13, fontweight="bold")
        ax.set_ylabel("Average Torsional Energy (kcal/mol)", fontsize=13, fontweight="bold")
        ax.legend(title="State", frameon=True, facecolor="white")

        plt.xticks(fontweight="bold")
        plt.tight_layout()

        stem = f"{stem_prefix}_energy_{region.lower()}"
        out1 = self.figures_dir / f"{stem}_comparison.png"
        out2 = self.alt_figures_dir / f"{stem}_comparison.png"

        fig.savefig(out1, dpi=600, bbox_inches="tight")
        fig.savefig(out2, dpi=600, bbox_inches="tight")
        plt.close(fig)

        logging.info(f"Generated {title_prefix} plot for {region_label}: paired p = {pair_p:.4e}")


def main():
    plotter = EnergyPValuePlotter()

    # Load JSON files
    df_pdb = plotter.load_json("torsion_potential_energy_summary_187.json")
    df_aa = plotter.load_json("amino_acid_torsion_energy_summary_187.json")

    if df_pdb is not None:
        # 1. PDB Level Plots
        plotter.plot_pdb_energy(df_pdb, region="INT", metric="AVG")
        plotter.plot_pdb_energy(df_pdb, region="INT", metric="TOT")
        plotter.plot_pdb_energy(df_pdb, region="NINT", metric="AVG")
        plotter.plot_pdb_energy(df_pdb, region="NINT", metric="TOT")

    if df_aa is not None:
        # 2. Amino Acid Level Plots (All AAs)
        plotter.plot_amino_acid_wise_energy(df_aa, aromatic_only=False, region="INT")
        plotter.plot_amino_acid_wise_energy(df_aa, aromatic_only=False, region="NINT")

        # 3. Aromatic Amino Acid Plots
        plotter.plot_amino_acid_wise_energy(df_aa, aromatic_only=True, region="INT")
        plotter.plot_amino_acid_wise_energy(df_aa, aromatic_only=True, region="NINT")

    logging.info("All two-tailed t-test energy plots generated successfully from JSON summaries.")


if __name__ == "__main__":
    main()
