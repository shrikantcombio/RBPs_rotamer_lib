#!/usr/bin/env python3
"""
backbone_scatter_plot.py

Descriptive scatter plots of backbone and side-chain torsions, bound vs unbound; the statistics are
in backbone_deviation.py.

  1. 3D projections: delta phi, delta psi and delta chi_n (bound - unbound, shortest arc) for
     every residue type, chi and region (I interface, N non-interface surface, C core).
  2. Ramachandran plots coloured by chi_n: bound only, unbound only, and bound vs unbound side by
     side, per residue group.

Two-fold symmetric terminal chis (ASP chi2, GLU chi3, PHE chi2, TYR chi2) are shown in their
canonical 180-degree range (rotamer_definitions.TERMINAL_SP2_CHI), and their delta chi is taken
modulo 180, so an OD1/OD2 or ring-flip relabelling is not shown as a 180-degree change. Missing
angles are empty in the dataset and are left out of each plot.

Outputs: figure(s)/backbone_scatter/3d_projections/ and figure(s)/backbone_scatter/ramachandran/

Usage:
    python script/backbone_scatter_plot.py [--plots all|3d|ramachandran]
"""

import argparse
import logging
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rotamer_definitions as rotdef

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "input_files"
TORSION_DATASET = DATASET_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"
FIGURES_DIRS = [ROOT_DIR / "figure" / "backbone_scatter", ROOT_DIR / "figures" / "backbone_scatter"]

REGIONS = {"I": "Interface", "N": "Non-interface surface", "C": "Core"}
MAX_CHI_3D = 4   # ARG chi5 is planar (NH1/NH2 equivalent) and has no rotamers to project

# Residue groups for the Ramachandran figures: (residues, file name)
RAMACHANDRAN_GROUPS = [
    (["ARG", "LYS"], "chi4_amino_acids"),
    (["GLU", "GLN", "MET"], "chi3_amino_acids"),
    (["ASN", "ASP", "LEU", "ILE"], "chi2_amino_acids"),
    (["CYS", "SER", "THR", "VAL"], "chi1_amino_acids"),
    (["PHE", "TRP", "TYR", "HIS"], "aromatic_amino_acids"),
    (["PRO"], "proline_amino_acid"),
]

CMAP = cm.twilight_shifted
NORM = colors.Normalize(-180, 180)
TEXT_COLOR, MUTED_COLOR, GRID_COLOR = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": MUTED_COLOR, "axes.labelcolor": TEXT_COLOR, "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR, "text.color": TEXT_COLOR,
})


def signed_diff(a, b, period=360.0):
    """Signed shortest difference a - b (degrees) in [-period/2, period/2)."""
    return (np.asarray(a, float) - np.asarray(b, float) + period / 2) % period - period / 2


def chi_period(aa, chi_idx):
    """180 for a two-fold symmetric terminal chi, else 360."""
    sp2 = rotdef.terminal_sp2(aa, chi_idx)
    return sp2[0] if sp2 is not None else 360.0


def display_chi(aa, chi_idx, angles):
    """Chi angles as plotted: symmetric terminal chis mapped into their canonical range."""
    sp2 = rotdef.terminal_sp2(aa, chi_idx)
    angles = np.asarray(angles, float)
    if sp2 is None or sp2[0] != 180:
        return angles
    return rotdef.canonical_angle(angles, sp2[0], sp2[1])


def chi_label(aa, k):
    return rf"$\chi_{k}$" + (" (mod 180°)" if chi_period(aa, k - 1) == 180 else "")


def save(fig, subdir, name):
    for out_dir in FIGURES_DIRS:
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / subdir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_dataset(path=TORSION_DATASET):
    df = pd.read_csv(path)
    df["AA"] = df["LABEL"].str[:3]
    return df[df["AA"].isin(rotdef.RESIDUE_CHI_COUNTS)].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 3D delta projections
# ----------------------------------------------------------------------------------------------
def plot_3d(aa, region, k, d_phi, d_psi, d_chi):
    """Scatter of delta phi, delta psi, delta chi_k for one residue type and region."""
    half = chi_period(aa, k - 1) / 2
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter3D(d_phi, d_psi, d_chi, c=TEXT_COLOR, marker=".", s=10, alpha=0.6, linewidths=0)
    ax.set_title(f"{aa}, {REGIONS[region].lower()} (n = {len(d_chi):,})", fontsize=11)
    ticks = np.arange(-180, 181, 60)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(np.arange(-half, half + 1, half / 3))
    ax.set_xlim3d(-180, 180)
    ax.set_ylim3d(-180, 180)
    ax.set_zlim3d(-half, half)
    ax.tick_params(labelsize=7)
    ax.set_xlabel(r"$\Delta\phi$ (°)", fontsize=10)
    ax.set_ylabel(r"$\Delta\psi$ (°)", fontsize=10)
    ax.set_zlabel(rf"$\Delta\chi_{k}$ (°)" + (" mod 180" if half == 90 else ""), fontsize=10)
    ax.set_box_aspect(None, zoom=0.85)   # keeps the z label inside the saved figure
    save(fig, "3d_projections", f"{aa}_{region}_chi{k}_3D_projection.png")


def generate_3d_plots(df):
    logging.info("Generating 3D delta projections...")
    d_phi = signed_diff(df["B_PHI"], df["U_PHI"])
    d_psi = signed_diff(df["B_PSI"], df["U_PSI"])
    n_plots = 0
    for aa, sub_idx in df.groupby("AA").indices.items():
        for k in range(1, min(rotdef.RESIDUE_CHI_COUNTS[aa], MAX_CHI_3D) + 1):
            d_chi = signed_diff(df[f"B_CHI{k}"].values[sub_idx], df[f"U_CHI{k}"].values[sub_idx],
                                chi_period(aa, k - 1))
            for region in REGIONS:
                keep = ((df["SASA"].values[sub_idx] == region) & np.isfinite(d_chi)
                        & np.isfinite(d_phi[sub_idx]) & np.isfinite(d_psi[sub_idx]))
                if keep.any():
                    plot_3d(aa, region, k, d_phi[sub_idx][keep], d_psi[sub_idx][keep], d_chi[keep])
                    n_plots += 1
    logging.info(f"Saved {n_plots} 3D projections")


# ----------------------------------------------------------------------------------------------
# Ramachandran plots coloured by chi
# ----------------------------------------------------------------------------------------------
def ramachandran_panel(ax, res_data, state, aa, k, title):
    cols = [f"{state}_PHI", f"{state}_PSI", f"{state}_CHI{k}"]
    sub = res_data.dropna(subset=cols)
    ax.scatter(sub[cols[0]], sub[cols[1]], c=display_chi(aa, k - 1, sub[cols[2]]), cmap=CMAP, norm=NORM,
               s=6, alpha=0.7, linewidths=0, rasterized=True)
    ax.set_title(f"{title}, {chi_label(aa, k)} (n = {len(sub):,})", fontsize=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.tick_params(labelsize=8)
    ax.grid(color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_aspect("equal")


def add_chi_colorbar(fig, axes):
    cbar = fig.colorbar(cm.ScalarMappable(cmap=CMAP, norm=NORM), ax=axes, fraction=0.025, shrink=0.6, pad=0.02)
    cbar.set_ticks(np.arange(-180, 181, 60))
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel(r"$\chi_n$ (°)", fontsize=11)


def plot_ramachandran_group(df, residues, state, name):
    """One row per residue, one column per chi, for the bound (B) or unbound (U) state."""
    title = {"B": "Bound", "U": "Unbound"}[state]
    max_chi = max(min(rotdef.RESIDUE_CHI_COUNTS[aa], MAX_CHI_3D) for aa in residues)
    fig, axes = plt.subplots(len(residues), max_chi, figsize=(3.6 * max_chi + 1, 3.6 * len(residues)),
                             squeeze=False, constrained_layout=True)
    for row, aa in enumerate(residues):
        res_data = df[df["AA"] == aa]
        nchi = min(rotdef.RESIDUE_CHI_COUNTS[aa], MAX_CHI_3D)
        for col in range(max_chi):
            ax = axes[row, col]
            if col >= nchi:
                ax.axis("off")
                continue
            ramachandran_panel(ax, res_data, state, aa, col + 1, f"{aa} {title.lower()}")
            if row == len(residues) - 1 or col >= min(rotdef.RESIDUE_CHI_COUNTS[residues[row + 1]], MAX_CHI_3D):
                ax.set_xlabel(r"$\phi$ (°)", fontsize=10)
            if col == 0:
                ax.set_ylabel(r"$\psi$ (°)", fontsize=10)
    add_chi_colorbar(fig, axes)
    save(fig, "ramachandran", f"Only_{state}_{name}_scatter_plot.png")


def plot_side_by_side(df, aa):
    """Bound (left) and unbound (right) Ramachandran plots per chi of one residue type."""
    nchi = min(rotdef.RESIDUE_CHI_COUNTS[aa], MAX_CHI_3D)
    fig, axes = plt.subplots(nchi, 2, figsize=(8.2, 3.7 * nchi), squeeze=False, constrained_layout=True)
    res_data = df[df["AA"] == aa]
    for k in range(1, nchi + 1):
        for col, state in enumerate("BU"):
            ax = axes[k - 1, col]
            ramachandran_panel(ax, res_data, state, aa, k, f"{aa} {'bound' if state == 'B' else 'unbound'}")
            if col == 0:
                ax.set_ylabel(r"$\psi$ (°)", fontsize=10)
            if k == nchi:
                ax.set_xlabel(r"$\phi$ (°)", fontsize=10)
    add_chi_colorbar(fig, axes)
    save(fig, "ramachandran", f"{aa}_side_by_side_bound_unbound_scatter.png")


def generate_ramachandran_plots(df):
    logging.info("Generating Ramachandran plots coloured by chi...")
    for residues, name in RAMACHANDRAN_GROUPS:
        for state in "BU":
            plot_ramachandran_group(df, residues, state, name)
        for aa in residues:
            plot_side_by_side(df, aa)
    logging.info("Saved Ramachandran plots")


def main():
    parser = argparse.ArgumentParser(description="Bound/unbound backbone and side-chain torsion scatter plots.")
    parser.add_argument("--dataset-file", type=Path, default=TORSION_DATASET, help="Torsion angle dataset.")
    parser.add_argument("--plots", choices=["all", "3d", "ramachandran"], default="all", help="Plots to generate.")
    args = parser.parse_args()

    logging.info(f"Reading dataset from: {args.dataset_file}")
    df = load_dataset(args.dataset_file)
    if args.plots in ("all", "3d"):
        generate_3d_plots(df)
    if args.plots in ("all", "ramachandran"):
        generate_ramachandran_plots(df)
    logging.info(f"Figures in {FIGURES_DIRS[-1]}")


if __name__ == "__main__":
    main()
