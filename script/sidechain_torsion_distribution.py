#!/usr/bin/env python3
"""
sidechain_torsion_distribution.py

Calculates the continuous distribution of side-chain torsion (Chi) angles
for all 20 standard amino acids in protein structures (Bound vs Unbound states).
Generates distribution summary statistics and publication-quality figures.

Usage:
    python script/sidechain_torsion_distribution.py [options]
"""

import argparse
import logging
import sys
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
sys.path.insert(0, str(ROOT_DIR))
from utils.torsion_dataset import TORSION_DATASET, load_torsion_dataset  # noqa: E402
RESULTS_DIR = ROOT_DIR / "output_files"
FIGURES_DIR = ROOT_DIR / "figures" / "sidechain_distributions"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set global Matplotlib / Seaborn style
plt.style.use("tableau-colorblind10")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0

# Number of Chi angles per amino acid
RESIDUE_CHI_MAP = {
    "ARG": ["CHI1", "CHI2", "CHI3", "CHI4", "CHI5"],
    "LYS": ["CHI1", "CHI2", "CHI3", "CHI4"],
    "GLN": ["CHI1", "CHI2", "CHI3"],
    "GLU": ["CHI1", "CHI2", "CHI3"],
    "MET": ["CHI1", "CHI2", "CHI3"],
    "ASN": ["CHI1", "CHI2"],
    "ASP": ["CHI1", "CHI2"],
    "HIS": ["CHI1", "CHI2"],
    "ILE": ["CHI1", "CHI2"],
    "LEU": ["CHI1", "CHI2"],
    "PHE": ["CHI1", "CHI2"],
    "PRO": ["CHI1", "CHI2"],
    "TRP": ["CHI1", "CHI2"],
    "TYR": ["CHI1", "CHI2"],
    "CYS": ["CHI1"],
    "SER": ["CHI1"],
    "THR": ["CHI1"],
    "VAL": ["CHI1"],
    "ALA": [],
    "GLY": [],
}


def _filter_res_df(df, res):
    if "LABEL" in df.columns:
        return df[df["LABEL"].astype(str).str[:3] == res]
    elif "AA" in df.columns:
        return df[df["AA"].astype(str).str[:3] == res]
    else:
        return pd.DataFrame()


def get_max_bin_info(data_series, num_bins=36, angle_range=(-180, 180)):
    """
    Find the bin with maximum frequency count and compute member statistics.
    """
    valid_data = pd.Series(data_series, dtype=float).dropna()
    valid_data = valid_data[(valid_data >= angle_range[0]) & (valid_data <= angle_range[1])]

    if len(valid_data) == 0:
        return {
            "count": 0,
            "max_bin_count": 0,
            "max_bin_start": np.nan,
            "max_bin_end": np.nan,
            "max_bin_center": np.nan,
        }

    counts, bin_edges = np.histogram(valid_data, bins=num_bins, range=angle_range)
    max_idx = np.argmax(counts)
    bin_start = bin_edges[max_idx]
    bin_end = bin_edges[max_idx + 1]
    bin_center = (bin_start + bin_end) / 2.0

    return {
        "count": len(valid_data),
        "max_bin_count": int(counts[max_idx]),
        "max_bin_start": float(bin_start),
        "max_bin_end": float(bin_end),
        "max_bin_center": float(bin_center),
    }


def compute_distribution_summary(df, state_prefix="B"):
    """
    Compute distribution summary statistics for all residues and Chi angles using circular statistics.
    """
    from scipy.stats import circmean, circstd
    logging.info(f"Computing distribution statistics for state '{state_prefix}'...")
    summary_rows = []

    for res, chi_list in RESIDUE_CHI_MAP.items():
        if not chi_list:
            continue

        if "LABEL" in df.columns:
            res_df = df[df["LABEL"].astype(str).str[:3] == res]
        elif "AA" in df.columns:
            res_df = df[df["AA"].astype(str).str[:3] == res]
        else:
            continue
        for chi in chi_list:
            col_name = f"{state_prefix}_{chi}"
            if col_name not in res_df.columns:
                continue

            series = res_df[col_name]
            # Exclude NaN or null values while preserving valid 0.0 deg dihedrals
            valid_vals = []
            for v in series.dropna():
                try:
                    vf = float(v)
                    if not np.isnan(vf):
                        valid_vals.append(vf)
                except (ValueError, TypeError):
                    continue

            valid_arr = np.array(valid_vals, dtype=float)

            if len(valid_arr) > 0:
                mean_val = float(circmean(valid_arr, high=180.0, low=-180.0))
                std_val = float(circstd(valid_arr, high=180.0, low=-180.0))
            else:
                mean_val = np.nan
                std_val = np.nan

            max_info = get_max_bin_info(valid_arr, num_bins=36)

            summary_rows.append(
                {
                    "RESIDUE": res,
                    "CHI_ANGLE": chi,
                    "STATE": state_prefix,
                    "TOTAL_COUNT": max_info["count"],
                    "MEAN_ANGLE": round(mean_val, 2) if not np.isnan(mean_val) else np.nan,
                    "STD_ANGLE": round(std_val, 2) if not np.isnan(std_val) else np.nan,
                    "PEAK_BIN_CENTER": round(max_info["max_bin_center"], 2) if not np.isnan(max_info["max_bin_center"]) else np.nan,
                    "PEAK_BIN_RANGE": f"[{max_info['max_bin_start']:.1f}, {max_info['max_bin_end']:.1f}]" if not np.isnan(max_info["max_bin_start"]) else "N/A",
                    "PEAK_BIN_COUNT": max_info["max_bin_count"],
                }
            )

    return pd.DataFrame(summary_rows)


def format_axis(ax, title, xlabel, ylabel, xlim=(-180, 180), xtick_step=60, y_limit=None):
    """
    Helper function to apply clean, consistent publication styling to plot axes.
    """
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    else:
        ax.set_xlabel("")

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel("")

    ax.set_xlim(xlim)
    xticks = np.arange(xlim[0], xlim[1] + 1, xtick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x}°" for x in xticks], rotation=0, fontsize=10)

    if y_limit:
        ax.set_ylim(0, y_limit)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved figure: {save_path}")
    plt.close(fig)


def plot_arg_distribution(df, state="B", kde=True, save_path=None):
    """Plot Chi1 to Chi4 distributions for Arginine (ARG) in a 2x2 grid."""
    arg_df = _filter_res_df(df, "ARG")
    if len(arg_df) == 0:
        logging.warning("No data found for ARG.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Arginine (ARG) Side-Chain Torsion Distributions (State: {state})", fontsize=14, fontweight="bold")

    chi_cols = [f"{state}_CHI{i}" for i in range(1, 5)]
    titles = [r"$\chi_1$ Dihedral", r"$\chi_2$ Dihedral", r"$\chi_3$ Dihedral", r"$\chi_4$ Dihedral"]

    for idx, (ax, col, title) in enumerate(zip(axes.flat, chi_cols, titles)):
        if col in arg_df.columns:
            data = arg_df[col].dropna()
            if len(data) > 0:
                sns.histplot(data, bins=36, kde=kde, ax=ax, color="#00468B", edgecolor="black", alpha=0.7)
            format_axis(ax, title, f"{title} (deg)", "Density / Count")

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_lys_distribution(df, state="B", kde=True, save_path=None):
    """Plot Chi1 to Chi4 distributions for Lysine (LYS) in a 2x2 grid."""
    lys_df = _filter_res_df(df, "LYS")
    if len(lys_df) == 0:
        logging.warning("No data found for LYS.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Lysine (LYS) Side-Chain Torsion Distributions (State: {state})", fontsize=14, fontweight="bold")

    chi_cols = [f"{state}_CHI{i}" for i in range(1, 5)]
    titles = [r"$\chi_1$ Dihedral", r"$\chi_2$ Dihedral", r"$\chi_3$ Dihedral", r"$\chi_4$ Dihedral"]

    for idx, (ax, col, title) in enumerate(zip(axes.flat, chi_cols, titles)):
        if col in lys_df.columns:
            data = lys_df[col].dropna()
            if len(data) > 0:
                sns.histplot(data, bins=36, kde=kde, ax=ax, color="#ED0000", edgecolor="black", alpha=0.7)
            format_axis(ax, title, f"{title} (deg)", "Density / Count")

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_glu_gln_met_distribution(df, state="B", kde=True, save_path=None):
    """Plot Chi1 to Chi3 distributions for GLU, GLN, and MET in a 3x3 grid."""
    fig, axs = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    residues = ["GLU", "GLN", "MET"]
    chi_angles = ["CHI1", "CHI2", "CHI3"]
    colors = ["#7570b3", "#e7298a", "#66a61e"]

    for r, res in enumerate(residues):
        res_df = _filter_res_df(df, res)
        for c, chi in enumerate(chi_angles):
            ax = axs[r, c]
            col = f"{state}_{chi}"
            if col in res_df.columns:
                data = res_df[col].dropna()
                if len(data) > 0:
                    sns.histplot(
                        data,
                        bins=36,
                        binrange=(-180, 180),
                        kde=kde,
                        color=colors[r],
                        edgecolor="white",
                        ax=ax,
                        alpha=0.7,
                    )
            xlabel = r"$\chi_{" + chi[-1] + r"}$ (degrees)" if r == 2 else None
            ylabel = "Count" if c == 0 else None
            format_axis(ax, f"{res} {chi}", xlabel, ylabel)

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_two_chi_pairs_distribution(df, res1, res2, title_prefix, color="#1b9e77", state="B", kde=True, save_path=None):
    """Plot Chi1 and Chi2 distributions for a pair of amino acids (e.g. ASP/ASN, LEU/ILE, PHE/HIS, TRP/TYR) in a 2x2 grid."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    res1_df = _filter_res_df(df, res1)
    res2_df = _filter_res_df(df, res2)

    c1 = f"{state}_CHI1"
    c2 = f"{state}_CHI2"

    if c1 in res1_df.columns:
        sns.histplot(res1_df[c1].dropna(), bins=36, binrange=(-180, 180), kde=kde, color=color, ax=axs[0, 0], alpha=0.7)
    if c1 in res2_df.columns:
        sns.histplot(res2_df[c1].dropna(), bins=36, binrange=(-180, 180), kde=kde, color=color, ax=axs[0, 1], alpha=0.7)
    if c2 in res1_df.columns:
        sns.histplot(res1_df[c2].dropna(), bins=36, binrange=(-180, 180), kde=kde, color=color, ax=axs[1, 0], alpha=0.7)
    if c2 in res2_df.columns:
        sns.histplot(res2_df[c2].dropna(), bins=36, binrange=(-180, 180), kde=kde, color=color, ax=axs[1, 1], alpha=0.7)

    format_axis(axs[0, 0], f"{res1} $\chi_1$", None, "Count")
    format_axis(axs[0, 1], f"{res2} $\chi_1$", None, None)
    format_axis(axs[1, 0], f"{res1} $\chi_2$", r"$\chi_2$ (degrees)", "Count")
    format_axis(axs[1, 1], f"{res2} $\chi_2$", r"$\chi_2$ (degrees)", None)

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_single_chi_group_distribution(df, residues=["CYS", "SER", "THR", "VAL"], state="B", kde=True, save_path=None):
    """Plot Chi1 distribution for single-Chi residues (CYS, SER, THR, VAL) in a 2x2 grid."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    colors = ["#e6ab02", "#a6761d", "#666666", "#e7298a"]

    for idx, (res, (r, c)) in enumerate(zip(residues, coords)):
        ax = axs[r, c]
        res_df = _filter_res_df(df, res)
        col = f"{state}_CHI1"
        if col in res_df.columns:
            data = res_df[col].dropna()
            if len(data) > 0:
                sns.histplot(
                    data,
                    bins=36,
                    binrange=(-180, 180),
                    kde=kde,
                    color=colors[idx],
                    edgecolor="white",
                    ax=ax,
                    alpha=0.7,
                )
        xlabel = r"$\chi_1$ (degrees)" if r == 1 else None
        ylabel = "Count" if c == 0 else None
        format_axis(ax, f"{res} $\chi_1$", xlabel, ylabel)

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_proline_distribution(df, state="B", kde=True, save_path=None):
    """Plot Chi1 and Chi2 distribution for Proline (PRO) in a 1x2 grid (-60° to 60° range)."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    pro_df = _filter_res_df(df, "PRO")

    c1 = f"{state}_CHI1"
    c2 = f"{state}_CHI2"

    if c1 in pro_df.columns:
        sns.histplot(pro_df[c1].dropna(), bins=36, binrange=(-60, 60), kde=kde, color="#333333", ax=axs[0], alpha=0.7)
    if c2 in pro_df.columns:
        sns.histplot(pro_df[c2].dropna(), bins=36, binrange=(-60, 60), kde=kde, color="#333333", ax=axs[1], alpha=0.7)

    format_axis(axs[0], r"PRO $\chi_1$", r"$\chi_1$ (degrees)", "Count", xlim=(-60, 60), xtick_step=20)
    format_axis(axs[1], r"PRO $\chi_2$", r"$\chi_2$ (degrees)", None, xlim=(-60, 60), xtick_step=20)

    plt.tight_layout()
    _save_figure(fig, save_path)


def generate_all_distribution_plots(df, state="B", kde=True, figures_dir=FIGURES_DIR):
    """Generate and save all distribution figure subplots."""
    logging.info(f"Generating distribution plots for state '{state}'...")

    plot_arg_distribution(df, state=state, kde=kde, save_path=figures_dir / f"ARG_{state}_CHI_distribution.png")
    plot_lys_distribution(df, state=state, kde=kde, save_path=figures_dir / f"LYS_{state}_CHI_distribution.png")
    plot_glu_gln_met_distribution(df, state=state, kde=kde, save_path=figures_dir / f"GLU_GLN_MET_{state}_CHI_distribution.png")
    plot_two_chi_pairs_distribution(df, "ASP", "ASN", "ASP_ASN", color="#1b9e77", state=state, kde=kde, save_path=figures_dir / f"ASP_ASN_{state}_CHI_distribution.png")
    plot_two_chi_pairs_distribution(df, "LEU", "ILE", "LEU_ILE", color="#d95f02", state=state, kde=kde, save_path=figures_dir / f"LEU_ILE_{state}_CHI_distribution.png")
    plot_two_chi_pairs_distribution(df, "PHE", "HIS", "PHE_HIS", color="#7570b3", state=state, kde=kde, save_path=figures_dir / f"PHE_HIS_{state}_CHI_distribution.png")
    plot_two_chi_pairs_distribution(df, "TRP", "TYR", "TRP_TYR", color="#e7298a", state=state, kde=kde, save_path=figures_dir / f"TRP_TYR_{state}_CHI_distribution.png")
    plot_single_chi_group_distribution(df, residues=["CYS", "SER", "THR", "VAL"], state=state, kde=kde, save_path=figures_dir / f"CYS_SER_THR_VAL_{state}_CHI_distribution.png")
    plot_proline_distribution(df, state=state, kde=kde, save_path=figures_dir / f"PRO_{state}_CHI_distribution.png")


def main():
    parser = argparse.ArgumentParser(description="Calculate side-chain torsion angle distributions and generate publication figures.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=TORSION_DATASET,
        help="Path to dataset containing backbone & sidechain torsion angles.",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for CSV summary files.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Output directory for figures.")
    parser.add_argument("--state", type=str, default="B", choices=["B", "U"], help="Protein state: 'B' for Bound, 'U' for Unbound.")
    parser.add_argument("--no-kde", action="store_true", help="Disable Kernel Density Estimation (KDE) curve overlay.")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    if not args.input_file.exists():
        logging.error(f"Input file not found: {args.input_file}")
        return

    df = load_torsion_dataset(args.input_file)

    logging.info(f"Loaded dataset with {len(df)} residue entries.")

    # 1. Compute summary statistics
    df_summary = compute_distribution_summary(df, state_prefix=args.state)
    summary_csv = args.output_dir / f"sidechain_torsion_distribution_summary_{args.state}.csv"
    df_summary.to_csv(summary_csv, index=False)
    logging.info(f"Saved distribution summary statistics to: {summary_csv}")

    # 2. Generate plots
    generate_all_distribution_plots(
        df,
        state=args.state,
        kde=not args.no_kde,
        figures_dir=args.figures_dir,
    )

    logging.info("Side-chain torsion distribution analysis completed successfully.")


if __name__ == "__main__":
    main()
