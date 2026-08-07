#!/usr/bin/env python3
"""
ramachandran_plot.py

Refactored Ramachandran Plot Analysis and Visualization Engine (RamachandranPlot).
Calculates backbone Phi (phi) and Psi (psi) dihedral angles from PDB/CIF structure files
or pre-calculated torsion angle datasets, and generates publication-quality scatter,
density contour, and multi-panel (General, Glycine, Proline, Pre-Proline) Ramachandran plots.

Usage:
    python script/ramachandran_plot.py [options]
"""

import sys
import argparse
import logging
from pathlib import Path

import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "input_files"
RESULTS_DIR = ROOT_DIR / "output_files" / "ramachandran"
FIGURES_DIR = ROOT_DIR / "figure" / "ramachandran"
ALT_FIGURES_DIR = ROOT_DIR / "figures" / "ramachandran"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ALT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from script.utils.mmcif_clean_reader import parse_file
except ImportError:
    from utils.mmcif_clean_reader import parse_file

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class RamachandranPlot:
    """
    Ramachandran plot calculator and visualizer.
    Supports PDB/CIF structure files (via mmcif_clean_reader.py) and pre-calculated torsion angle datasets.
    """

    RAMA_TYPES = ["General", "Glycine", "Proline", "Pre-Pro"]

    def __init__(self, data=None, input_path=None, precalculated_file=None):
        """
        Initialize RamachandranPlot.

        Parameters:
        -----------
        data : pd.DataFrame, optional
            Pre-loaded DataFrame containing Phi and Psi angles.
        input_path : Path or str, optional
            Path to PDB/CIF file, list file, or directory of structure files.
        precalculated_file : Path or str, optional
            Path to CSV/TSV dataset containing pre-calculated backbone torsion angles.
        """
        self.data_df = None

        if data is not None and isinstance(data, pd.DataFrame):
            self.data_df = data.copy()
        elif precalculated_file:
            self.load_precalculated_data(precalculated_file)
        elif input_path:
            self.process_structure_input(input_path)

    @staticmethod
    def calculate_dihedral(p1, p2, p3, p4):
        """Calculate dihedral angle in degrees for 4 3D points p1, p2, p3, p4."""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm == 0 or n2_norm == 0:
            return np.nan
        n1 /= n1_norm
        n2 /= n2_norm
        norm_b2 = np.linalg.norm(b2)
        if norm_b2 == 0:
            return np.nan
        b2_u = b2 / norm_b2
        m1 = np.cross(n1, b2_u)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return float(np.degrees(np.arctan2(y, x)))

    @staticmethod
    def classify_rama_type(res_name, next_res_name=None):
        """
        Classify residue into Ramachandran case: General, Glycine, Proline, Pre-Pro.
        """
        res = str(res_name).strip().upper()
        next_res = str(next_res_name).strip().upper() if next_res_name else ""

        if res == "GLY":
            return "Glycine"
        elif res == "PRO":
            return "Proline"
        elif next_res == "PRO":
            return "Pre-Pro"
        else:
            return "General"

    @staticmethod
    def classify_region(phi, psi, rama_type="General"):
        """
        Classify (phi, psi) pair into Favored, Allowed, or Outlier regions.
        """
        if pd.isna(phi) or pd.isna(psi):
            return "Outlier"

        phi = float(phi)
        psi = float(psi)

        # General / Pre-Pro Favored
        if (-100 <= phi <= -30 and -80 <= psi <= -10) or \
           (-180 <= phi <= -45 and (90 <= psi <= 180 or -180 <= psi <= -150)) or \
           (30 <= phi <= 100 and 10 <= psi <= 80):
            return "Favored"

        # General Allowed (expanded margins)
        if (-120 <= phi <= -20 and -100 <= psi <= 20) or \
           (-180 <= phi <= -30 and (60 <= psi <= 180 or -180 <= psi <= -120)) or \
           (20 <= phi <= 120 and -10 <= psi <= 100):
            return "Allowed"

        # Glycine has broader allowed regions
        if rama_type == "Glycine":
            if (30 <= phi <= 100 and -80 <= psi <= 10) or (-100 <= phi <= -30 and 10 <= psi <= 80):
                return "Favored"
            if (-180 <= phi <= 180 and -180 <= psi <= 180):
                return "Allowed"

        # Proline restricted phi around -65
        if rama_type == "Proline":
            if -100 <= phi <= -40 and (-80 <= psi <= 40 or 90 <= psi <= 180):
                return "Favored"

        return "Outlier"

    def extract_angles_from_structure_file(self, file_path):
        """
        Extract Phi and Psi backbone angles from a single PDB or mmCIF structure file
        using script/utils/mmcif_clean_reader.py.
        """
        file_path = Path(file_path)
        residue_angle_records = []

        try:
            cols, rows = parse_file(str(file_path))
        except Exception as e:
            logging.error(f"Failed to parse structure file {file_path} using mmcif_clean_reader: {e}")
            return residue_angle_records

        # Group atoms by residue key (chain, resnum, icode, resname)
        residues = {}
        for r in rows:
            chain_id = r.get("_atom_site.auth_asym_id") or r.get("_atom_site.label_asym_id")
            res_num = r.get("_atom_site.auth_seq_id") or r.get("_atom_site.label_seq_id")
            icode = r.get("_atom_site.pdbx_PDB_ins_code", ".").strip()
            if icode == ".":
                icode = ""
            res_name = r.get("_atom_site.auth_comp_id") or r.get("_atom_site.label_comp_id")
            atom_name = r.get("_atom_site.auth_atom_id") or r.get("_atom_site.label_atom_id")

            try:
                x = float(r["_atom_site.Cartn_x"])
                y = float(r["_atom_site.Cartn_y"])
                z = float(r["_atom_site.Cartn_z"])
                coord = np.array([x, y, z], dtype=float)
            except (KeyError, ValueError):
                continue

            res_key = (chain_id, str(res_num), str(icode), str(res_name))
            if res_key not in residues:
                residues[res_key] = {}
            residues[res_key][str(atom_name)] = coord

        res_keys = list(residues.keys())
        for idx in range(1, len(res_keys) - 1):
            prev_key = res_keys[idx - 1]
            curr_key = res_keys[idx]
            next_key = res_keys[idx + 1]

            # Ensure residues are on the same chain
            if prev_key[0] == curr_key[0] == next_key[0]:
                c_prev = residues[prev_key].get("C")
                n_curr = residues[curr_key].get("N")
                ca_curr = residues[curr_key].get("CA")
                c_curr = residues[curr_key].get("C")
                n_next = residues[next_key].get("N")

                if c_prev is not None and n_curr is not None and ca_curr is not None and c_curr is not None and n_next is not None:
                    phi_deg = self.calculate_dihedral(c_prev, n_curr, ca_curr, c_curr)
                    psi_deg = self.calculate_dihedral(n_curr, ca_curr, c_curr, n_next)

                    if np.isnan(phi_deg) or np.isnan(psi_deg):
                        continue

                    chain_id = curr_key[0]
                    res_num = curr_key[1]
                    res_name = curr_key[3]
                    next_res_name = next_key[3]

                    rama_type = self.classify_rama_type(res_name, next_res_name)
                    region = self.classify_region(phi_deg, psi_deg, rama_type)

                    record = {
                        "PDB": file_path.stem,
                        "CHAIN": chain_id,
                        "RESNUM": res_num,
                        "RESNAME": res_name,
                        "LABEL": f"{res_name} {chain_id} {res_num}",
                        "PHI": float(phi_deg),
                        "PSI": float(psi_deg),
                        "RAMA_TYPE": rama_type,
                        "REGION": region,
                    }
                    residue_angle_records.append(record)

        return residue_angle_records

    def process_structure_input(self, input_path):
        """
        Process PDB/CIF input path (single file, list file, or directory).
        """
        input_path = Path(input_path)
        all_records = []

        if input_path.is_file():
            # Check if it's a structure file or a list file
            if input_path.suffix.lower() in [".pdb", ".ent", ".cif", ".mmcif"]:
                logging.info(f"Parsing single structure file: {input_path}")
                all_records.extend(self.extract_angles_from_structure_file(input_path))
            else:
                # Text file list of paths
                logging.info(f"Parsing list file of structures: {input_path}")
                with open(input_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        p = Path(line.split()[0])
                        if p.exists():
                            all_records.extend(self.extract_angles_from_structure_file(p))
        elif input_path.is_dir():
            logging.info(f"Scanning directory for PDB/CIF files: {input_path}")
            struct_files = list(input_path.glob("**/*.pdb")) + list(input_path.glob("**/*.ent")) + list(input_path.glob("**/*.cif"))
            for p in struct_files:
                all_records.extend(self.extract_angles_from_structure_file(p))

        if all_records:
            self.data_df = pd.DataFrame(all_records)
            logging.info(f"Successfully extracted {len(self.data_df)} residue Phi/Psi angle pairs.")
        else:
            logging.warning("No structure angle records extracted.")

    def load_precalculated_data(self, file_path_or_df):
        """
        Load pre-calculated backbone torsion angle data from CSV/TSV or DataFrame.
        """
        if isinstance(file_path_or_df, pd.DataFrame):
            df = file_path_or_df.copy()
        else:
            p = Path(file_path_or_df)
            if not p.exists():
                logging.error(f"Precalculated file not found: {p}")
                return
            try:
                df = pd.read_csv(p, sep=",")
                if "LABEL" not in df.columns and "B_PHI" not in df.columns and "PHI" not in df.columns:
                    df = pd.read_csv(p, sep="\t")
            except Exception:
                df = pd.read_csv(p, sep="\t")

        # Map standard column names if using B_PHI/B_PSI or U_PHI/U_PSI
        if "B_PHI" in df.columns and "PHI" not in df.columns:
            df["PHI"] = df["B_PHI"]
            df["PSI"] = df["B_PSI"]
        elif "phi" in df.columns and "PHI" not in df.columns:
            df["PHI"] = df["phi"]
            df["PSI"] = df["psi"]

        if "RESNAME" not in df.columns and "LABEL" in df.columns:
            df["RESNAME"] = df["LABEL"].astype(str).str[:3]

        if "RAMA_TYPE" not in df.columns:
            df["RAMA_TYPE"] = df["RESNAME"].apply(self.classify_rama_type)

        if "REGION" not in df.columns:
            df["REGION"] = df.apply(lambda row: self.classify_region(row["PHI"], row["PSI"], row.get("RAMA_TYPE", "General")), axis=1)

        self.data_df = df
        logging.info(f"Loaded precalculated data with {len(self.data_df)} records.")

    def draw_ramachandran_regions(self, ax):
        """
        Draw reference region outlines and grid lines on Ramachandran axes.
        """
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Alpha-helix core box outline
        rect_alpha = mpatches.Rectangle((-100, -80), 70, 70, fill=False, edgecolor="#004225", linestyle="-", linewidth=1.5, alpha=0.8, label=r"$\alpha$-helix")
        # Beta-sheet core box outline
        rect_beta = mpatches.Rectangle((-180, 90), 135, 90, fill=False, edgecolor="#B22222", linestyle="-", linewidth=1.5, alpha=0.8, label=r"$\beta$-sheet")
        # Left-handed alpha core box outline
        rect_lalpha = mpatches.Rectangle((30, 10), 70, 70, fill=False, edgecolor="#663399", linestyle="-", linewidth=1.5, alpha=0.8, label=r"L-$\alpha$")

        ax.add_patch(rect_alpha)
        ax.add_patch(rect_beta)
        ax.add_patch(rect_lalpha)

    # =========================================================================
    # 1. SCATTER PLOT
    # =========================================================================
    def plot_scatter(
        self,
        phi_col="PHI",
        psi_col="PSI",
        hue_col=None,
        title="Ramachandran Scatter Plot",
        save_path=None,
        alpha=0.6,
        s=15,
    ):
        """
        Generate Ramachandran scatter plot of Phi vs Psi angles.

        Parameters:
        -----------
        phi_col : str
        psi_col : str
        hue_col : str, optional
            Column name to color points by (e.g. 'RAMA_TYPE', 'REGION', 'SASA').
        title : str
        save_path : Path or str, optional
        alpha : float
        s : int
        """
        if self.data_df is None or self.data_df.empty:
            logging.error("No angle data available for Ramachandran scatter plot.")
            return

        df = self.data_df.dropna(subset=[phi_col, psi_col]).copy()
        if df.empty:
            logging.error(f"Columns {phi_col} and {psi_col} contain no valid numerical data.")
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        self.draw_ramachandran_regions(ax)

        if hue_col and hue_col in df.columns:
            if HAS_SEABORN:
                sns.scatterplot(data=df, x=phi_col, y=psi_col, hue=hue_col, ax=ax, alpha=alpha, s=s, edgecolor="none", zorder=3)
            else:
                for val, grp in df.groupby(hue_col):
                    ax.scatter(grp[phi_col], grp[psi_col], alpha=alpha, s=s, label=str(val), zorder=3)
                ax.legend(title=hue_col, loc="upper right")
        else:
            ax.scatter(df[phi_col], df[psi_col], color="#1f77b4", alpha=alpha, s=s, zorder=3, label="Residues")

        ax.set_xlabel(r"$\Phi$ (degrees)", fontsize=13, fontweight="bold")
        ax.set_ylabel(r"$\Psi$ (degrees)", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        # Calculate region statistics
        total = len(df)
        favored = len(df[df["REGION"] == "Favored"])
        allowed = len(df[df["REGION"] == "Allowed"])
        outliers = len(df[df["REGION"] == "Outlier"])

        stats_text = (
            f"Total: {total}\n"
            f"Favored: {favored} ({favored/total*100:.1f}%)\n"
            f"Allowed: {allowed} ({allowed/total*100:.1f}%)\n"
            f"Outliers: {outliers} ({outliers/total*100:.1f}%)"
        )
        ax.text(
            0.03,
            0.03,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
            zorder=5,
        )

        plt.tight_layout()

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")
            logging.info(f"Saved Ramachandran scatter plot: {out_p}")

        plt.close(fig)

    # =========================================================================
    # 2. 2D DENSITY CONTOUR PLOT
    # =========================================================================
    def plot_contour(
        self,
        phi_col="PHI",
        psi_col="PSI",
        title="Ramachandran Density Contour Plot",
        save_path=None,
        cmap="YlOrRd",
        levels=10,
        scatter_overlay=True,
    ):
        """
        Generate 2D population density contour Ramachandran plot using Gaussian KDE.
        """
        if self.data_df is None or self.data_df.empty:
            logging.error("No angle data available for Ramachandran contour plot.")
            return

        df = self.data_df.dropna(subset=[phi_col, psi_col]).copy()
        if len(df) < 5:
            logging.error("Insufficient data points for 2D density contour calculation.")
            return

        phi_vals = df[phi_col].values
        psi_vals = df[psi_col].values

        # Grid calculation
        xi, yi = np.mgrid[-180:180:180j, -180:180:180j]
        coords = np.vstack([phi_vals, psi_vals])
        kde = gaussian_kde(coords)
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

        fig, ax = plt.subplots(figsize=(7, 7))
        self.draw_ramachandran_regions(ax)

        # Plot filled contours
        cf = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap, alpha=0.85, zorder=2)
        fig.colorbar(cf, ax=ax, label="Population Density")

        if scatter_overlay:
            ax.scatter(phi_vals, psi_vals, color="black", alpha=0.25, s=8, zorder=3)

        ax.set_xlabel(r"$\Phi$ (degrees)", fontsize=13, fontweight="bold")
        ax.set_ylabel(r"$\Psi$ (degrees)", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

        plt.tight_layout()

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")
            logging.info(f"Saved Ramachandran density contour plot: {out_p}")

        plt.close(fig)

    # =========================================================================
    # 3. FACETED 2X2 RAMACHANDRAN PLOT
    # =========================================================================
    def plot_faceted(
        self,
        phi_col="PHI",
        psi_col="PSI",
        title="Ramachandran Plot by Residue Type",
        save_path=None,
    ):
        """
        Generate 2x2 multi-panel Ramachandran plots: General, Glycine, Proline, Pre-Pro.
        """
        if self.data_df is None or self.data_df.empty:
            logging.error("No data available for faceted Ramachandran plot.")
            return

        df = self.data_df.dropna(subset=[phi_col, psi_col]).copy()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        axes = axes.flatten()

        for i, rama_type in enumerate(self.RAMA_TYPES):
            ax = axes[i]
            self.draw_ramachandran_regions(ax)

            sub_df = df[df["RAMA_TYPE"] == rama_type]

            if not sub_df.empty:
                ax.scatter(sub_df[phi_col], sub_df[psi_col], color="#1f77b4", alpha=0.5, s=12, zorder=3)

                tot = len(sub_df)
                fav = len(sub_df[sub_df["REGION"] == "Favored"])
                out = len(sub_df[sub_df["REGION"] == "Outlier"])
                ax.text(
                    0.04,
                    0.04,
                    f"N: {tot}\nFav: {fav/tot*100:.1f}%\nOut: {out/tot*100:.1f}%",
                    transform=ax.transAxes,
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
                    zorder=4,
                )

            ax.set_title(rama_type, fontsize=12, fontweight="bold")
            ax.set_xlabel(r"$\Phi$ (deg)" if i >= 2 else "", fontsize=11)
            ax.set_ylabel(r"$\Psi$ (deg)" if i % 2 == 0 else "", fontsize=11)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()

        if save_path:
            out_p = Path(save_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_p, dpi=300, bbox_inches="tight")
            logging.info(f"Saved faceted Ramachandran plot: {out_p}")

        plt.close(fig)

    # =========================================================================
    # 4. SUMMARY CSV EXPORT
    # =========================================================================
    def export_summary(self, output_csv_path=None):
        """
        Export statistical summary table of Ramachandran distribution by residue and region.
        """
        if self.data_df is None or self.data_df.empty:
            logging.error("No data available for summary export.")
            return None

        df = self.data_df.dropna(subset=["PHI", "PSI"]).copy()

        summary_rows = []
        for (resn, rtype), grp in df.groupby(["RESNAME", "RAMA_TYPE"]):
            tot = len(grp)
            fav = len(grp[grp["REGION"] == "Favored"])
            allow = len(grp[grp["REGION"] == "Allowed"])
            out = len(grp[grp["REGION"] == "Outlier"])

            row = {
                "RESNAME": resn,
                "RAMA_TYPE": rtype,
                "TOTAL_COUNT": tot,
                "FAVORED_COUNT": fav,
                "ALLOWED_COUNT": allow,
                "OUTLIER_COUNT": out,
                "FAVORED_PCT": round(fav / tot * 100.0, 2) if tot > 0 else 0.0,
                "ALLOWED_PCT": round(allow / tot * 100.0, 2) if tot > 0 else 0.0,
                "OUTLIER_PCT": round(out / tot * 100.0, 2) if tot > 0 else 0.0,
                "MEAN_PHI": round(grp["PHI"].mean(), 2),
                "STD_PHI": round(grp["PHI"].std(), 2),
                "MEAN_PSI": round(grp["PSI"].mean(), 2),
                "STD_PSI": round(grp["PSI"].std(), 2),
            }
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        if output_csv_path:
            out_p = Path(output_csv_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(out_p, index=False)
            logging.info(f"Saved Ramachandran summary CSV: {out_p}")

        return summary_df


def main():
    parser = argparse.ArgumentParser(description="Ramachandran Plot Generator and Torsion Angle Calculator.")
    parser.add_argument("--input-path", type=Path, default=None, help="Path to single PDB/CIF file, list file, or directory of structures.")
    parser.add_argument("--precalculated-tsv", type=Path, default=DATASET_DIR / "PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv", help="Path to precalculated torsion angle CSV/TSV file.")
    parser.add_argument("--plot-type", type=str, default="all", choices=["scatter", "contour", "faceted", "all"], help="Plot type to generate.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Output directory for figure plots.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory for CSV summary tables.")

    args = parser.parse_args()

    # Load dataset or structure input
    if args.input_path and args.input_path.exists():
        plotter = RamachandranPlot(input_path=args.input_path)
    else:
        plotter = RamachandranPlot(precalculated_file=args.precalculated_tsv)

    out_fig_dir = args.figures_dir
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_type in ["scatter", "all"]:
        plotter.plot_scatter(save_path=out_fig_dir / "ramachandran_scatter.png")
        plotter.plot_scatter(hue_col="RAMA_TYPE", title="Ramachandran Plot by Residue Type", save_path=out_fig_dir / "ramachandran_scatter_by_type.png")

    if args.plot_type in ["contour", "all"]:
        plotter.plot_contour(save_path=out_fig_dir / "ramachandran_contour.png")

    if args.plot_type in ["faceted", "all"]:
        plotter.plot_faceted(save_path=out_fig_dir / "ramachandran_faceted.png")

    # Export summary CSV
    plotter.export_summary(output_csv_path=args.output_dir / "ramachandran_summary.csv")

    logging.info("RamachandranPlot execution finished successfully.")


if __name__ == "__main__":
    main()
