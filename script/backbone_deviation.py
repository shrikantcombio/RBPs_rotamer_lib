import logging
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

plt.style.use('tableau-colorblind10')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Directory configurations
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "input_files"
RESULTS_DIR = ROOT_DIR / "output_files"
FIGURES_DIR = ROOT_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Amino acid definitions for Chi angles
CHI_RESIDUES = {
    'CHI1': ["ARG", "ASN", "ASP", "GLU", "GLN", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "TRP", "TYR", "CYS", "SER", "THR", "VAL"],
    'CHI2': ["ARG", "ASN", "ASP", "GLU", "GLN", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "TRP", "TYR"],
    'CHI3': ["ARG", "LYS", "GLU", "GLN", "MET"],
    'CHI4': ["ARG", "LYS"],
    'CHI5': ["ARG"]
}
SASA_TYPES = ["I", "N"]


def circ_diff(b, u):
    """Computes signed shortest circular difference in degrees in [-180, 180]."""
    diff_rad = np.radians(np.asarray(b, dtype=float)) - np.radians(np.asarray(u, dtype=float))
    return np.degrees(np.arctan2(np.sin(diff_rad), np.cos(diff_rad)))


def calculate_delta_angles(df_torsion):
    """
    Calculate normalized delta angles for backbone and side-chains.
    Filters out sentinel zero backbone values (U_PHI/U_PSI == 0.0) and uses RMS backbone deviation.
    """
    # Filter sentinel zeros and missing values in backbone dihedrals
    mask = (
        df_torsion["U_PHI"].notna() & (df_torsion["U_PHI"].astype(float) != 0.0) &
        df_torsion["U_PSI"].notna() & (df_torsion["U_PSI"].astype(float) != 0.0) &
        df_torsion["B_PHI"].notna() & (df_torsion["B_PHI"].astype(float) != 0.0) &
        df_torsion["B_PSI"].notna() & (df_torsion["B_PSI"].astype(float) != 0.0)
    )
    df = df_torsion[mask].copy()

    delta_phi = circ_diff(df["B_PHI"], df["U_PHI"])
    delta_psi = circ_diff(df["B_PSI"], df["U_PSI"])

    # RMS Backbone deviation: captures total magnitude regardless of sign cancellation
    delta_phi_psi = np.sqrt((delta_phi**2 + delta_psi**2) / 2.0)

    label_col = 'LABEL' if 'LABEL' in df.columns else ('D_LABEL' if 'D_LABEL' in df.columns else 'AA')

    dict_delta = {
        "PDB": df["PDB"] if "PDB" in df.columns else df.index,
        "D_LABEL": df[label_col],
        "D_PHI_PSI": delta_phi_psi,
        "D_PHI": delta_phi,
        "D_PSI": delta_psi,
        "SS": df["B_SS"] if "B_SS" in df.columns else "L",
        "SASA": df["SASA"] if "SASA" in df.columns else "N"
    }

    for i in range(1, 6):
        b_col, u_col = f"B_CHI{i}", f"U_CHI{i}"
        if b_col in df.columns and u_col in df.columns:
            dict_delta[f"D_CHI{i}"] = circ_diff(df[b_col], df[u_col])

    df_delta = pd.DataFrame(dict_delta)
    return df_delta


def benjamini_hochberg(p_values):
    """Computes Benjamini-Hochberg False Discovery Rate (FDR) adjusted p-values."""
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([])
    valid_mask = ~np.isnan(p_vals)
    adj_p = np.full(n, np.nan)
    if not np.any(valid_mask):
        return adj_p

    valid_p = p_vals[valid_mask]
    n_valid = len(valid_p)
    s_idx = np.argsort(valid_p)
    s_p = valid_p[s_idx]

    adj_valid = np.zeros(n_valid)
    c_min = 1.0
    for i in range(n_valid - 1, -1, -1):
        rank = i + 1
        a = (s_p[i] * n_valid) / rank
        c_min = min(c_min, a)
        adj_valid[s_idx[i]] = min(1.0, c_min)

    adj_p[valid_mask] = adj_valid
    return adj_p


def compute_spearman_correlations(df_delta, chi_col, residues):
    """
    Compute Spearman rank correlation and FDR-adjusted p-values between RMS D_PHI_PSI and specified D_CHI column
    at both Residue-level and PDB-structure level for Interface (I) and Non-interface (N) states.
    """
    label_col = 'D_LABEL' if 'D_LABEL' in df_delta.columns else ('LABEL' if 'LABEL' in df_delta.columns else 'AA')
    corr_matrices = {}

    for sasa in SASA_TYPES:
        sasa_df = df_delta[df_delta['SASA'].astype(str).str.startswith(sasa)]
        res_data_list = []
        p_raw_list = []

        for res in residues:
            res_df = sasa_df[sasa_df[label_col].astype(str).str[:3] == res].dropna(subset=['D_PHI_PSI', f"D_{chi_col}"])
            if len(res_df) > 2:
                r_val, p_val = stats.spearmanr(res_df['D_PHI_PSI'], np.abs(res_df[f"D_{chi_col}"]))

                # PDB structure-level aggregation
                pdb_agg = res_df.groupby('PDB')[['D_PHI_PSI', f"D_{chi_col}"]].mean()
                if len(pdb_agg) > 2:
                    pdb_r, pdb_p = stats.spearmanr(pdb_agg['D_PHI_PSI'], np.abs(pdb_agg[f"D_{chi_col}"]))
                else:
                    pdb_r, pdb_p = np.nan, np.nan
            else:
                r_val, p_val = np.nan, np.nan
                pdb_r, pdb_p = np.nan, np.nan

            res_data_list.append([r_val, p_val])
            p_raw_list.append(p_val)
            print("%s %s RMS D_PHI_PSI vs D_%s (Spearman): Residue rho=%7.4f p=%7.4e | PDB rho=%7.4f p=%7.4e" % 
                  (res, sasa, chi_col, r_val, p_val, pdb_r, pdb_p))

        # FDR BH Adjustment
        p_adj = benjamini_hochberg(p_raw_list)
        res_arr = np.array(res_data_list)
        if len(res_arr) > 0:
            res_arr[:, 1] = p_adj

        corr_matrices[sasa] = res_arr

    return corr_matrices["I"], corr_matrices["N"]


def draw_heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """Create a heatmap from a 2D numpy array and label lists."""
    if not ax:
        ax = plt.gca()

    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:7.4f}", textcolors=("black", "white"), threshold=None, **textkw):
    """Annotate heatmap cells with numerical text."""
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_correlation_figures(chi_idx, chi_name, residues, int_data, nint_data, output_dir):
    """Generate and save heatmap correlation figures."""
    labels = ["Spearman rho", "FDR p-adj"]

    if len(residues) <= 5:
        fig_size = (5.0, 5.0)
        cbar_size = "10%" if len(residues) > 2 else "5%"
        title_font = 10
        adj_kwargs = dict(left=0.025, right=0.4, top=0.8, bottom=0.6, wspace=0, hspace=0)
    else:
        fig_size = (6.0, 12.0)
        cbar_size = "15%"
        title_font = 14
        adj_kwargs = dict(left=1.6, right=2.93, top=0.8, bottom=0.6, wspace=0, hspace=0.25)

    fig, (ax1, ax2) = plt.subplots(figsize=fig_size, ncols=2)
    plt.subplots_adjust(**adj_kwargs)

    im1, cbar1 = draw_heatmap(int_data, residues, labels, ax=ax1, cmap="Greys", vmin=-1, vmax=1, cbarlabel="Spearman rho")
    im2, cbar2 = draw_heatmap(nint_data, residues, labels, ax=ax2, cmap="Greys", vmin=-1, vmax=1, cbarlabel="Spearman rho")

    cbar2.minorticks_on()
    cbar1.remove()
    cbar2.remove()

    annotate_heatmap(im1, valfmt="{x:7.4f}")
    annotate_heatmap(im2, valfmt="{x:7.4f}")

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size=cbar_size, pad=0.1 if len(residues) <= 5 else 0.2)
    plt.colorbar(im2, cax=cax)

    ax2.set_yticklabels("")
    ax1.set_title("Interface", fontsize=title_font)
    ax2.set_title("Non-interface", fontsize=title_font)

    fig.tight_layout()
    output_filepath = output_dir / f"{chi_idx}.bkbn_correlation_ΔΦΨ_Δ{chi_name}.tiff"
    fig.savefig(output_filepath, dpi=300)
    logging.info(f"Saved figure: {output_filepath}")
    plt.close(fig)


def main():
    input_rotamer_data = DATASET_DIR / 'ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187.json'
    if not input_rotamer_data.exists():
        input_rotamer_data = DATASET_DIR / 'ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv'

    logging.info(f"Reading torsion angle dataset from: {input_rotamer_data}")
    if input_rotamer_data.suffix.lower() == '.json':
        df_torsion = pd.read_json(input_rotamer_data)
    else:
        try:
            df_torsion = pd.read_csv(input_rotamer_data, sep=",")
            if "LABEL" not in df_torsion.columns and "D_LABEL" not in df_torsion.columns:
                df_torsion = pd.read_csv(input_rotamer_data, sep="\t")
        except Exception:
            df_torsion = pd.read_csv(input_rotamer_data, sep="\t")

    # Calculate delta angles
    df_delta_angles = calculate_delta_angles(df_torsion)

    # Save output delta angles
    output_delta_tsv = RESULTS_DIR / "DELTA_PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_final.tsv"
    df_delta_angles.round(2).to_csv(output_delta_tsv, sep='\t', index=False)
    logging.info(f"Saved delta torsion angles to: {output_delta_tsv}")

    # Calculate correlations and generate plots for CHI1..CHI4
    chi_list = [("1", "CHI1"), ("2", "CHI2"), ("3", "CHI3"), ("4", "CHI4")]

    for idx, chi_name in chi_list:
        residues = CHI_RESIDUES[chi_name]
        int_data, nint_data = compute_spearman_correlations(df_delta_angles, chi_name, residues)
        plot_correlation_figures(idx, chi_name, residues, int_data, nint_data, FIGURES_DIR)


if __name__ == "__main__":
    main()