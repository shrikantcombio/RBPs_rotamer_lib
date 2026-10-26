#!/usr/bin/env python3
"""
sidechain_torsion_distribution.py

Side-chain torsion (chi) distributions of the bound and unbound proteins of the 187 PRDBv3
complexes, with MolProbity Top8000 rotamer validation (Richardson lab reference_data, CC-BY-4.0;
Hintze et al. 2016 Proteins 84:1177; Williams et al. 2018 Protein Sci 27:293).

Angle conventions:
  - Two-fold symmetric terminal chis (ASP chi2, GLU chi3, PHE chi2, TYR chi2) are folded into
    their 180-degree canonical range (rotamer_definitions.TERMINAL_SP2_CHI), as in the rotamer
    libraries and in MolProbity: an OD1/OD2, OE1/OE2 or ring-flip relabelling is the same
    conformation.
  - ARG chi5 (CD-NE-CZ-NH1) is planar and NH1/NH2 are equivalent, so chi5 near 0 and near 180
    are the same; it is shown as the deviation from planarity, folded into [-90, 90).
  - Rotamer bins are the library bins (rotamer_definitions.chi_class): p / t / m for sp3 chis,
    residue-specific modes for terminal sp2 chis. A single mean of a multi-modal chi is not
    meaningful, so statistics are given per bin.

Top8000 rotamer validation (as in MolProbity rotalyze): the residue's chis (all of them; PRO chi1
only) are looked up in the Top8000 rotamer grid of its type (linear interpolation, periodic);
value >= 0.02 Favored, >= 0.003 Allowed, else Outlier. Residues with a missing chi are not scored.

Outputs (output_files/sidechain_distributions/):
  chi_bin_summary.csv        per residue type, chi, state and bin: count, %, circular mean / SD
  rotamer_validation.csv     per residue: Top8000 score and region, bound and unbound
  rotamer_validation_summary.csv  per state, region (I / N / C / all) and residue type
Figures (figure(s)/sidechain_distributions/):
  <group>_CHI_distribution.png     chi histograms, bound (filled) vs unbound (outline), with the
                                   Top8000 rotamer centres (frequency >= 1%) marked
  chi1_chi2/<AA>_{B,U}_chi1_chi2.png  chi1 vs chi2 scatter in the manuscript style over the Top8000
                                   favored / allowed regions (for residues with more chis, the
                                   region where some value of the other chis is favored / allowed)

Usage:
    python script/sidechain_torsion_distribution.py [--region all|I|N|C]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import circmean, circstd

import rotamer_definitions as rotdef

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
from utils.top8000 import TOP8000_DIR, Top8000Grid  # noqa: E402
from utils.torsion_dataset import TORSION_DATASET, load_torsion_dataset  # noqa: E402

RESULTS_DIR = ROOT_DIR / "output_files" / "sidechain_distributions"
FIGURES_DIRS = [ROOT_DIR / "figure" / "sidechain_distributions", ROOT_DIR / "figures" / "sidechain_distributions"]
ROTA_GRID_DIR = TOP8000_DIR / "Top8000_rotamer_pct_contour_grids"
ROTA_CENTRES_DIR = TOP8000_DIR / "Top8000_rotamer_central_values"

FAVORED, ALLOWED = 0.02, 0.003
REGION_NAMES = {"all": "all residues", "I": "interface", "N": "non-interface surface", "C": "core"}
CHI_COUNTS = {aa: n + (1 if aa == "ARG" else 0) for aa, n in rotdef.RESIDUE_CHI_COUNTS.items()}  # + ARG chi5
# Top8000 grid per residue type and how many chis it uses
ROTA_GRIDS = {"ARG": ("arg", 4), "ASN": ("asn", 2), "ASP": ("asp", 2), "CYS": ("cys", 1),
              "GLN": ("gln", 3), "GLU": ("glu", 3), "HIS": ("his", 2), "ILE": ("ile", 2),
              "LEU": ("leu", 2), "LYS": ("lys", 4), "MET": ("met", 3), "PHE": ("phetyr", 2),
              "PRO": ("pro", 1), "SER": ("ser", 1), "THR": ("thr", 1), "TRP": ("trp", 2),
              "TYR": ("phetyr", 2), "VAL": ("val", 1)}
GROUPS = [("ARG", ["ARG"]), ("LYS", ["LYS"]), ("GLU_GLN_MET", ["GLU", "GLN", "MET"]),
          ("ASP_ASN", ["ASP", "ASN"]), ("LEU_ILE", ["LEU", "ILE"]), ("PHE_HIS", ["PHE", "HIS"]),
          ("TRP_TYR", ["TRP", "TYR"]), ("CYS_SER_THR_VAL", ["CYS", "SER", "THR", "VAL"]), ("PRO", ["PRO"])]

# Manuscript scatter style (final_clustering/phi_psi_scatter_plot.ipynb)
POINT_COLOR, OUTLIER_COLOR = "#3F51B5", "#eb6834"
BOUND_COLOR, UNBOUND_COLOR = "#3F51B5", "#eb6834"
FAVORED_FILL, ALLOWED_FILL, CONTOUR_LINE = "#d6d5cf", "#efeeea", "#9a9993"
TEXT_COLOR, MUTED_COLOR = "#1f1f1e", "#6b6a64"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"]})


# ----------------------------------------------------------------------------------------------
# Angle conventions
# ----------------------------------------------------------------------------------------------
def display_range(aa, k):
    """(low, high, period) of chi k (1-based) as plotted."""
    if aa == "ARG" and k == 5:
        return -90.0, 90.0, 180.0
    if aa == "PRO":
        return -60.0, 60.0, 360.0
    sp2 = rotdef.terminal_sp2(aa, k - 1)
    if sp2 is not None and sp2[0] == 180:
        return sp2[1], sp2[1] + 180.0, 180.0
    return -180.0, 180.0, 360.0


def display_angles(aa, k, angles):
    low, _, period = display_range(aa, k)
    a = np.asarray(angles, float)
    if period == 180.0:
        return (a - low) % 180.0 + low
    return (a + 180.0) % 360.0 - 180.0


def bin_label(aa, k, angle):
    if aa == "ARG" and k == 5:
        return "planar"
    sp2 = rotdef.terminal_sp2(aa, k - 1)
    if sp2 is not None:
        return rotdef.terminal_sp2_label(aa, k - 1, angle)
    return "ptm"[rotdef.chi_class(aa, k - 1, angle)]


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------
def chi_bin_summary(df, region):
    rows = []
    for aa, n_chi in CHI_COUNTS.items():
        sub = df[df["AA"] == aa]
        for k in range(1, n_chi + 1):
            low, high, period = display_range(aa, k)
            for s in "BU":
                vals = display_angles(aa, k, sub[f"{s}_CHI{k}"].dropna())
                if len(vals) == 0:
                    continue
                labels = np.array([bin_label(aa, k, v) for v in vals])
                for lab in dict.fromkeys(labels):
                    v = vals[labels == lab]
                    rows.append({"REGION": region, "RESIDUE": aa, "CHI": f"CHI{k}", "STATE": s, "BIN": lab,
                                 "COUNT": len(v), "PCT": round(len(v) / len(vals) * 100, 2),
                                 "CIRC_MEAN": round(float(circmean(v, high=high, low=low)), 2),
                                 "CIRC_SD": round(float(circstd(v, high=high, low=low)), 2),
                                 "PERIOD": int(period)})
    return pd.DataFrame(rows)


def load_rotamer_grids():
    return {name: Top8000Grid(ROTA_GRID_DIR / f"rota8000-{name}.data") for name, _ in set(ROTA_GRIDS.values())}


def rotamer_validation(df, grids):
    out = df[["PDB", "LABEL", "AA", "SASA"]].copy()
    for s in "BU":
        score = np.full(len(df), np.nan)
        for aa, (name, n) in ROTA_GRIDS.items():
            m = (df["AA"] == aa).to_numpy()
            chis = [df.loc[m, f"{s}_CHI{k}"].to_numpy(float) for k in range(1, n + 1)]
            score[m] = grids[name](*chis)
        out[f"{s}_ROTA_SCORE"] = score
        out[f"{s}_ROTA_REGION"] = np.where(np.isnan(score), None,
                                           np.where(score >= FAVORED, "Favored",
                                                    np.where(score >= ALLOWED, "Allowed", "Outlier")))
    return out


def validation_summary(val):
    rows = []
    for s in "BU":
        d = val.dropna(subset=[f"{s}_ROTA_REGION"])
        for region in ["all", "I", "N", "C"]:
            r = d if region == "all" else d[d["SASA"] == region]
            for aa, g in [("All", r)] + list(r.groupby("AA")):
                counts = g[f"{s}_ROTA_REGION"].value_counts()
                n = len(g)
                rec = {"STATE": s, "REGION": region, "RESIDUE": aa, "TOTAL_COUNT": n}
                for lab in ("Favored", "Allowed", "Outlier"):
                    rec[f"{lab.upper()}_COUNT"] = int(counts.get(lab, 0))
                    rec[f"{lab.upper()}_PCT"] = round(counts.get(lab, 0) / n * 100, 2) if n else np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def save(fig, name):
    for out_dir in FIGURES_DIRS:
        (out_dir / Path(name).parent).mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, dpi=600, bbox_inches="tight")
    plt.close(fig)


def top8000_centres(aa, k, min_freq=1.0):
    """Chi k centres of the Top8000 rotamers of `aa` with frequency >= min_freq %, as plotted."""
    path = ROTA_CENTRES_DIR / f"{aa}.csv"
    col = f"chi{k}_mean"
    if not path.exists():
        return []
    c = pd.read_csv(path)
    if col not in c.columns:
        return []
    vals = display_angles(aa, k, c.loc[c["frequency%"] >= min_freq, col].dropna())
    out = []
    for v in sorted(vals):
        if all(min(abs(v - u), 360 - abs(v - u)) > 6 for u in out):
            out.append(v)
    return out


def plot_group(df, name, residues, region):
    n_cols = max(CHI_COUNTS[aa] for aa in residues)
    fig, axes = plt.subplots(len(residues), n_cols, figsize=(3.3 * n_cols + 0.4, 2.7 * len(residues)),
                             squeeze=False)
    for row, aa in enumerate(residues):
        sub = df[df["AA"] == aa]
        for col in range(n_cols):
            ax = axes[row, col]
            k = col + 1
            if k > CHI_COUNTS[aa]:
                ax.axis("off")
                continue
            low, high, period = display_range(aa, k)
            width = 5.0 if period == 180 or aa == "PRO" else 10.0
            bins = np.arange(low, high + width / 2, width)
            for s, color, kw in (("B", BOUND_COLOR, dict(histtype="stepfilled", alpha=0.35)),
                                 ("U", UNBOUND_COLOR, dict(histtype="step", linewidth=1.4))):
                vals = display_angles(aa, k, sub[f"{s}_CHI{k}"].dropna())
                weights = np.full(len(vals), 100.0 / max(len(vals), 1))
                ax.hist(vals, bins=bins, weights=weights, color=color, **kw)
            for c in top8000_centres(aa, k):
                ax.axvline(c, color=MUTED_COLOR, linewidth=0.6, linestyle=":", zorder=0)
            n = sub[[f"B_CHI{k}", f"U_CHI{k}"]].notna().all(axis=1).sum()
            note = " (mod 180°)" if period == 180 and not (aa == "ARG" and k == 5) else \
                (" (planarity, mod 180°)" if aa == "ARG" and k == 5 else "")
            ax.set_title(rf"{aa} $\chi_{k}${note}", fontsize=10)
            ax.text(0.98, 0.95, f"n = {n:,}", transform=ax.transAxes, ha="right", va="top", fontsize=7,
                    color=MUTED_COLOR)
            step = 30 if (high - low) <= 180 else 60
            ax.set_xlim(low, high)
            ax.set_xticks(np.arange(low, high + 1, step))
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col == 0:
                ax.set_ylabel("Residues (%)", fontsize=9)
    handles = [Patch(facecolor=BOUND_COLOR, alpha=0.35, label="Bound"),
               Patch(facecolor="none", edgecolor=UNBOUND_COLOR, linewidth=1.4, label="Unbound"),
               plt.Line2D([], [], color=MUTED_COLOR, linestyle=":", label="Top8000 rotamer centres (≥ 1%)")]
    fig.tight_layout(rect=(0, 0, 1, 0.94 if len(residues) > 1 else 0.86))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.94 if len(residues) > 1 else 0.86))
    if region != "all":
        fig.suptitle(REGION_NAMES[region].capitalize(), fontsize=11, y=1.0)
    save(fig, f"{name}_CHI_distribution{'' if region == 'all' else '_' + region}.png")


def chi1_chi2_regions(aa, grids):
    """Display mesh and Top8000 values for chi1 x chi2 (max over further chis)."""
    name, n = ROTA_GRIDS[aa]
    grid = grids[name].project_max([0, 1]) if n > 2 else grids[name]
    lo2, hi2, _ = display_range(aa, 2)
    x = np.arange(-180, 181, 2.0)
    y = np.arange(lo2, hi2 + 1, 2.0 if hi2 - lo2 > 180 else 1.0)
    xx, yy = np.meshgrid(x, y)
    return x, y, grid(xx, yy)


def plot_chi1_chi2(df, val, grids, aa, s, region):
    sub = df[df["AA"] == aa]
    keep = sub[[f"{s}_CHI1", f"{s}_CHI2"]].notna().all(axis=1)
    sub, v = sub[keep], val.loc[sub.index[keep], f"{s}_ROTA_REGION"]
    x1, x2 = display_angles(aa, 1, sub[f"{s}_CHI1"]), display_angles(aa, 2, sub[f"{s}_CHI2"])
    out = (v == "Outlier").to_numpy()
    lo2, hi2, period2 = display_range(aa, 2)

    fig, ax = plt.subplots(figsize=(4, 4 if period2 == 360 else 2.6))
    gx, gy, gz = chi1_chi2_regions(aa, grids)
    ax.contourf(gx, gy, gz, levels=[ALLOWED, FAVORED, np.inf], colors=[ALLOWED_FILL, FAVORED_FILL], zorder=0)
    ax.contour(gx, gy, gz, levels=[ALLOWED, FAVORED], colors=CONTOUR_LINE, linewidths=0.6, zorder=1)
    ax.scatter(x1[~out], x2[~out], s=10, alpha=0.6, color=POINT_COLOR, marker="P", linewidths=0,
               rasterized=True, zorder=2)
    ax.scatter(x1[out], x2[out], s=14, color=OUTLIER_COLOR, marker="X", linewidths=0, zorder=3)

    ax.set_xlim(-180, 180)
    ax.set_ylim(lo2, hi2)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(lo2, hi2 + 1, 60 if period2 == 360 else 30))
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r"$\chi_1$ (°)", fontsize=14)
    ax.set_ylabel(r"$\chi_2$ (°)" + (" mod 180" if period2 == 180 else ""), fontsize=14)
    for pos in (0,):
        ax.axvline(pos, linestyle="--", linewidth=0.8, color="#0072B2", zorder=1)
        if lo2 < pos < hi2:
            ax.axhline(pos, linestyle="--", linewidth=0.8, color="#0072B2", zorder=1)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    n_scored = v.notna().sum()
    ax.set_title(aa, fontsize=15, pad=16)
    scope = "" if ROTA_GRIDS[aa][1] <= 2 else f", judged on χ1–χ{ROTA_GRIDS[aa][1]}"
    ax.text(0.5, 1.01, f"n = {len(sub):,}  ·  Top8000 outliers {out.sum()} ({out.sum() / max(n_scored, 1):.1%}){scope}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#4a4a46")
    suffix = "" if region == "all" else f"_{region}"
    save(fig, f"chi1_chi2/{aa}_{s}_chi1_chi2{suffix}.png")


def main():
    parser = argparse.ArgumentParser(description="Side-chain torsion distributions with Top8000 rotamer validation.")
    parser.add_argument("--dataset-file", type=Path, default=TORSION_DATASET, help="Torsion angle dataset.")
    parser.add_argument("--region", default="all", choices=["all", "I", "N", "C"],
                        help="Residues plotted: all, interface (I), non-interface surface (N) or core (C).")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for tables.")
    args = parser.parse_args()

    df = load_torsion_dataset(args.dataset_file)
    df["AA"] = df["LABEL"].str[:3]
    df = df[df["AA"].isin(CHI_COUNTS)].reset_index(drop=True)
    grids = load_rotamer_grids()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    val = rotamer_validation(df, grids)
    val.to_csv(args.output_dir / "rotamer_validation.csv", index=False, float_format="%.5g")
    vsum = validation_summary(val)
    vsum.to_csv(args.output_dir / "rotamer_validation_summary.csv", index=False)
    for s in "BU":
        r = vsum[(vsum.STATE == s) & (vsum.REGION == "all") & (vsum.RESIDUE == "All")].iloc[0]
        logging.info(f"{s}: {r.TOTAL_COUNT} side chains scored, favored {r.FAVORED_PCT}%, "
                     f"Top8000 rotamer outliers {r.OUTLIER_COUNT} ({r.OUTLIER_PCT}%)")

    plot_df = df if args.region == "all" else df[df["SASA"] == args.region]
    chi_bin_summary(plot_df, args.region).to_csv(
        args.output_dir / f"chi_bin_summary{'' if args.region == 'all' else '_' + args.region}.csv", index=False)
    for name, residues in GROUPS:
        plot_group(plot_df, name, residues, args.region)
    for aa in [a for a, n in CHI_COUNTS.items() if n >= 2 and a != "PRO"]:
        for s in "BU":
            plot_chi1_chi2(plot_df, val, grids, aa, s, args.region)
    logging.info(f"Tables in {args.output_dir}; figures in {FIGURES_DIRS[-1]}")


if __name__ == "__main__":
    main()
