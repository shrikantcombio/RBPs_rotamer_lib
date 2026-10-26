#!/usr/bin/env python3
"""
ramachandran_plot.py

Ramachandran validation and plots of the bound and unbound proteins of the 187 PRDBv3 complexes,
against the MolProbity Top8000 reference contours (Richardson lab reference_data, CC-BY-4.0;
Williams et al. 2018 Protein Sci 27:293; Hintze et al. 2016 Proteins 84:1177).

Categories (as in MolProbity ramalyze), in order of precedence:
  Glycine    GLY                                   rama8000-gly-sym
  trans-Pro  PRO, preceding peptide bond trans     rama8000-transpro
  cis-Pro    PRO, preceding omega within 30 of 0   rama8000-cispro
  pre-Pro    residue before PRO (not GLY / PRO)    rama8000-prepro-noGP
  Ile/Val    ILE, VAL                              rama8000-ileval-nopreP
  General    all others                            rama8000-general-noGPIVpreP
The preceding omega is the OMEGA of the previous residue in the dataset (CA(i-1)-C(i-1)-N(i)-CA(i));
a residue is pre-Pro when the next residue in the same chain, peptide-bonded to it (psi defined),
is PRO.

Region: the grid value at (phi, psi) (linear interpolation between bin centres, periodic) is
Favored when >= 0.02 (2%), Allowed when >= 0.0005 (General), 0.002 (cis-Pro) or 0.001 (others),
else Outlier (MolProbity thresholds).

Outputs:
  output_files/ramachandran/ramachandran_residues.csv   per residue: category, score, region (B and U)
  output_files/ramachandran/ramachandran_summary.csv    per state, category and residue type:
                                                        counts and % favored / allowed / outlier,
                                                        circular mean and SD of phi and psi
  figure(s)/ramachandran/ramachandran_categories_{B,U}.png   six categories over Top8000 contours
  figure(s)/ramachandran/per_residue/<AA>_{B,U}_ramachandran.png
                                                        one residue type (pre-Pro and cis-Pro
                                                        residues excluded) over its contour

Usage:
    python script/ramachandran_plot.py [--states B U] [--no-per-residue]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.top8000 import TOP8000_DIR, Top8000Grid  # noqa: E402
from utils.torsion_dataset import TORSION_DATASET, load_torsion_dataset  # noqa: E402

RESULTS_DIR = ROOT_DIR / "output_files" / "ramachandran"
FIGURES_DIRS = [ROOT_DIR / "figure" / "ramachandran", ROOT_DIR / "figures" / "ramachandran"]
RAMA_GRID_DIR = TOP8000_DIR / "Top8000_ramachandran_pct_contour_grids"

# Category -> (Top8000 grid file, allowed threshold)
CATEGORIES = {
    "General": ("rama8000-general-noGPIVpreP.data", 0.0005),
    "Glycine": ("rama8000-gly-sym.data", 0.001),
    "trans-Pro": ("rama8000-transpro.data", 0.001),
    "cis-Pro": ("rama8000-cispro.data", 0.002),
    "pre-Pro": ("rama8000-prepro-noGP.data", 0.001),
    "Ile/Val": ("rama8000-ileval-nopreP.data", 0.001),
}
FAVORED = 0.02
CIS_OMEGA = 30.0
REGIONS = ["Favored", "Allowed", "Outlier"]

# Manuscript scatter style (final_clustering/phi_psi_scatter_plot.ipynb)
POINT_COLOR = "#3F51B5"
OUTLIER_COLOR = "#eb6834"
FAVORED_FILL, ALLOWED_FILL, CONTOUR_LINE = "#d6d5cf", "#efeeea", "#9a9993"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})


# ----------------------------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------------------------
def load_grids():
    return {cat: Top8000Grid(RAMA_GRID_DIR / fname) for cat, (fname, _) in CATEGORIES.items()}


def neighbour_columns(df):
    """Previous residue's OMEGA and next residue's name, within the same complex and chain."""
    chain = df["LABEL"].str.split().str[1]
    resname = df["LABEL"].str[:3]
    same_prev = (df["PDB"].shift(1) == df["PDB"]) & (chain.shift(1) == chain)
    same_next = (df["PDB"].shift(-1) == df["PDB"]) & (chain.shift(-1) == chain)
    prev_omega = {s: df[f"{s}_OMEGA"].shift(1).where(same_prev) for s in "BU"}
    # psi of residue i is defined only when i is peptide-bonded to i + 1
    next_res = {s: resname.shift(-1).where(same_next & df[f"{s}_PSI"].notna()) for s in "BU"}
    return resname, prev_omega, next_res


def rama_category(resname, prev_omega, next_res):
    """MolProbity Ramachandran category of each residue (arrays of equal length)."""
    cat = np.full(len(resname), "General", dtype=object)
    cat[np.isin(resname, ["ILE", "VAL"])] = "Ile/Val"
    cat[(next_res == "PRO") & ~np.isin(resname, ["GLY", "PRO"])] = "pre-Pro"
    is_pro = resname == "PRO"
    cis = np.abs(np.nan_to_num(prev_omega.astype(float), nan=180.0)) < CIS_OMEGA
    cat[is_pro & cis] = "cis-Pro"
    cat[is_pro & ~cis] = "trans-Pro"
    cat[resname == "GLY"] = "Glycine"
    return cat


def classify(df, grids):
    """Per-residue category, Top8000 score and region for the bound and unbound states."""
    resname, prev_omega, next_res = neighbour_columns(df)
    out = pd.DataFrame({"PDB": df["PDB"], "LABEL": df["LABEL"], "RESNAME": resname, "SASA": df["SASA"]})
    for s in "BU":
        phi, psi = df[f"{s}_PHI"].to_numpy(float), df[f"{s}_PSI"].to_numpy(float)
        cat = rama_category(resname.to_numpy(), prev_omega[s].to_numpy(), next_res[s].to_numpy())
        score = np.full(len(df), np.nan)
        region = np.full(len(df), None, dtype=object)
        for name, (_, allowed) in CATEGORIES.items():
            m = cat == name
            score[m] = grids[name](phi[m], psi[m])
            region[m] = np.where(score[m] >= FAVORED, "Favored",
                                 np.where(score[m] >= allowed, "Allowed", "Outlier"))
        undefined = np.isnan(phi) | np.isnan(psi)
        region[undefined] = None
        out[f"{s}_PHI"], out[f"{s}_PSI"] = phi, psi
        out[f"{s}_RAMA_TYPE"] = np.where(undefined, None, cat)
        out[f"{s}_RAMA_SCORE"], out[f"{s}_RAMA_REGION"] = score, region
    return out


def summary_table(res):
    rows = []
    for s in "BU":
        d = res.dropna(subset=[f"{s}_RAMA_REGION"])
        groups = [("All", "All", d)] + [(cat, "All", g) for cat, g in d.groupby(f"{s}_RAMA_TYPE")] + \
                 [(cat, aa, g) for (cat, aa), g in d.groupby([f"{s}_RAMA_TYPE", "RESNAME"])]
        for cat, aa, g in groups:
            counts = g[f"{s}_RAMA_REGION"].value_counts()
            n = len(g)
            rec = {"STATE": s, "RAMA_TYPE": cat, "RESNAME": aa, "TOTAL_COUNT": n}
            for r in REGIONS:
                rec[f"{r.upper()}_COUNT"] = int(counts.get(r, 0))
                rec[f"{r.upper()}_PCT"] = round(counts.get(r, 0) / n * 100, 2)
            for a in ("PHI", "PSI"):
                rec[f"MEAN_{a}"] = round(float(circmean(g[f"{s}_{a}"], high=180, low=-180)), 2)
                rec[f"SD_{a}"] = round(float(circstd(g[f"{s}_{a}"], high=180, low=-180)), 2)
            rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def draw_contours(ax, grid, allowed):
    # Pad one wrapped bin on every side so the contours reach the +-180 edges
    x = np.concatenate([[grid.centres(0)[0] - grid.width[0]], grid.centres(0), [grid.centres(0)[-1] + grid.width[0]]])
    y = np.concatenate([[grid.centres(1)[0] - grid.width[1]], grid.centres(1), [grid.centres(1)[-1] + grid.width[1]]])
    z = np.pad(grid.values, 1, mode="wrap").T          # rows = psi, columns = phi
    ax.contourf(x, y, z, levels=[allowed, FAVORED, np.inf], colors=[ALLOWED_FILL, FAVORED_FILL], zorder=0)
    ax.contour(x, y, z, levels=[allowed, FAVORED], colors=CONTOUR_LINE, linewidths=0.6, zorder=1)


def style_axes(ax, title, subtitle=None):
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ticks = np.arange(-180, 181, 60)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=10)
    ax.axhline(0, linestyle="--", linewidth=0.8, color="#0072B2", zorder=1)
    ax.axvline(0, linestyle="--", linewidth=0.8, color="#0072B2", zorder=1)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=13, pad=16 if subtitle else 6)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color="#4a4a46")


def scatter_panel(ax, d, s, grid, allowed, title):
    draw_contours(ax, grid, allowed)
    out = d[f"{s}_RAMA_REGION"] == "Outlier"
    ax.scatter(d.loc[~out, f"{s}_PHI"], d.loc[~out, f"{s}_PSI"], s=8, alpha=0.6, color=POINT_COLOR,
               marker="P", linewidths=0, rasterized=True, zorder=2)
    ax.scatter(d.loc[out, f"{s}_PHI"], d.loc[out, f"{s}_PSI"], s=14, color=OUTLIER_COLOR,
               marker="X", linewidths=0, zorder=3)
    counts = d[f"{s}_RAMA_REGION"].value_counts()
    n = len(d)
    style_axes(ax, title, f"n = {n:,}  ·  favored {counts.get('Favored', 0) / n:.1%}  ·  "
                          f"outliers {counts.get('Outlier', 0)} ({counts.get('Outlier', 0) / n:.2%})")


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(facecolor=FAVORED_FILL, edgecolor=CONTOUR_LINE, label="Top8000 favored (98%)"),
            Patch(facecolor=ALLOWED_FILL, edgecolor=CONTOUR_LINE, label="Top8000 allowed"),
            plt.Line2D([], [], marker="P", linestyle="", color=POINT_COLOR, label="Favored / allowed"),
            plt.Line2D([], [], marker="X", linestyle="", color=OUTLIER_COLOR, label="Outlier")]


def save(fig, name):
    for out_dir in FIGURES_DIRS:
        (out_dir / Path(name).parent).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_categories(res, grids, s):
    state = {"B": "bound", "U": "unbound"}[s]
    fig, axes = plt.subplots(2, 3, figsize=(12, 9.2))
    d_all = res.dropna(subset=[f"{s}_RAMA_REGION"])
    for ax, (cat, (_, allowed)) in zip(axes.flat, CATEGORIES.items()):
        scatter_panel(ax, d_all[d_all[f"{s}_RAMA_TYPE"] == cat], s, grids[cat], allowed, cat)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\phi$ (°)", fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\psi$ (°)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.5)
    fig.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=4,
               frameon=False, fontsize=10)
    fig.suptitle(f"Ramachandran plot, {state} proteins (187 complexes)", fontsize=14, y=0.995)
    save(fig, f"ramachandran_categories_{s}.png")


def plot_per_residue(res, grids, s):
    """One panel per residue type over its own category contour; pre-Pro and cis-Pro excluded."""
    d_all = res.dropna(subset=[f"{s}_RAMA_REGION"])
    for aa, d in d_all.groupby("RESNAME"):
        cat = {"GLY": "Glycine", "PRO": "trans-Pro", "ILE": "Ile/Val", "VAL": "Ile/Val"}.get(aa, "General")
        d = d[d[f"{s}_RAMA_TYPE"] == cat]
        fig, ax = plt.subplots(figsize=(4, 4))
        scatter_panel(ax, d, s, grids[cat], CATEGORIES[cat][1], aa)
        ax.set_xlabel(r"$\phi$ (°)", fontsize=14)
        ax.set_ylabel(r"$\psi$ (°)", fontsize=14)
        save(fig, f"per_residue/{aa}_{s}_ramachandran.png")


def main():
    parser = argparse.ArgumentParser(description="Ramachandran validation against MolProbity Top8000 contours.")
    parser.add_argument("--dataset-file", type=Path, default=TORSION_DATASET, help="Torsion angle dataset.")
    parser.add_argument("--states", nargs="+", default=["B", "U"], choices=["B", "U"], help="States to plot.")
    parser.add_argument("--no-per-residue", action="store_true", help="Skip the per-residue-type figures.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for tables.")
    args = parser.parse_args()

    grids = load_grids()
    res = classify(load_torsion_dataset(args.dataset_file), grids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.output_dir / "ramachandran_residues.csv", index=False, float_format="%.5g")
    summary = summary_table(res)
    summary.to_csv(args.output_dir / "ramachandran_summary.csv", index=False)
    for s in "BU":
        r = summary[(summary.STATE == s) & (summary.RAMA_TYPE == "All")].iloc[0]
        logging.info(f"{s}: {r.TOTAL_COUNT} residues, favored {r.FAVORED_PCT}%, allowed {r.ALLOWED_PCT}%, "
                     f"outliers {r.OUTLIER_COUNT} ({r.OUTLIER_PCT}%)")

    for s in args.states:
        plot_categories(res, grids, s)
        if not args.no_per_residue:
            plot_per_residue(res, grids, s)
    logging.info(f"Tables in {args.output_dir}; figures in {FIGURES_DIRS[-1]}")


if __name__ == "__main__":
    main()
