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


def normalize_delta_angles(angles):
    """Normalize angular difference to range [-180, 180] degrees."""
    return ((np.array(angles) + 180) % 360) - 180


def calculate_delta_angles(df_torsion):
    """Calculate normalized delta angles for backbone and side-chains."""
    delta_phi = normalize_delta_angles(df_torsion["B_PHI"] - df_torsion["U_PHI"])
    delta_psi = normalize_delta_angles(df_torsion["B_PSI"] - df_torsion["U_PSI"])
    delta_phi_psi = (delta_phi + delta_psi) / 2.0

    dict_delta = {
        "PDB": df_torsion["PDB"],
        "D_LABEL": df_torsion["LABEL"],
        "D_PHI_PSI": delta_phi_psi,
        "D_PHI": delta_phi,
        "D_PSI": delta_psi,
        "SS": df_torsion["B_SS"],
        "SASA": df_torsion["SASA"]
    }

    for i in range(1, 6):
        chi_col = f"CHI{i}"
        dict_delta[f"D_{chi_col}"] = normalize_delta_angles(df_torsion[f"B_{chi_col}"] - df_torsion[f"U_{chi_col}"])

    df_delta = pd.DataFrame(dict_delta)
    return df_delta


def compute_pearson_correlations(df_delta, chi_col, residues):
    """
    Compute Pearson correlation and p-value matrix between D_PHI_PSI and specified D_CHI column
    for Interface (I) and Non-interface (N) states across specified residues.
    """
    corr_matrices = {}
    for sasa in SASA_TYPES:
        sasa_df = df_delta[df_delta['SASA'].str.startswith(sasa)]
        data_list = []

        for res in residues:
            res_df = sasa_df[sasa_df['D_LABEL'].str[:3] == res]
            if len(res_df) > 1:
                r_val, p_val = stats.pearsonr(res_df['D_PHI_PSI'], res_df[f"D_{chi_col}"])
            else:
                r_val, p_val = np.nan, np.nan

            data_list.append([r_val, p_val])
            print("%s %s D_PHI_PSI vs D_%s: %7.4f\t%7.4f" % (res, sasa, chi_col, r_val, p_val))

        corr_matrices[sasa] = np.array(data_list)

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
    labels = ["correlation", "p-value"]

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

    im1, cbar1 = draw_heatmap(int_data, residues, labels, ax=ax1, cmap="Greys", vmin=-1, vmax=1, cbarlabel="correlation coeff")
    im2, cbar2 = draw_heatmap(nint_data, residues, labels, ax=ax2, cmap="Greys", vmin=-1, vmax=1, cbarlabel="correlation coeff")

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
    input_rotamer_data = DATASET_DIR / 'PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv'
    logging.info(f"Reading torsion angle dataset from: {input_rotamer_data}")
    df_torsion = pd.read_csv(input_rotamer_data, sep=",")

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
        int_data, nint_data = compute_pearson_correlations(df_delta_angles, chi_name, residues)
        plot_correlation_figures(idx, chi_name, residues, int_data, nint_data, FIGURES_DIR)


if __name__ == "__main__":
    main()