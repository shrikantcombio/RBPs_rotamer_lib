import logging
from pathlib import Path
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

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

# Amino acid definitions
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

    return pd.DataFrame(dict_delta)


# ----------------------------------------------------
# 3D Delta Projection Scatter Plots
# ----------------------------------------------------
def plot_3d(residue_name, int_res, x, y, z, chi_angle_name):
    """Plot 3D projection between Delta Phi, Delta Psi, and Delta Chi angles."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.array(x), np.array(y), np.array(z)
    ax.scatter3D(x, y, z, c='black', marker='.', s=20, alpha=1)

    ax.set_title(f"{residue_name}:{int_res} ({chi_angle_name})", fontsize=12)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    ax.set_zticks(np.arange(-180, 181, 60))

    ax.set_xlim3d(-180, 180)
    ax.set_ylim3d(-180, 180)
    ax.set_zlim3d(-180, 180)

    ax.set_xlabel(r'$\Delta\phi$', fontsize=10)
    ax.set_ylabel(r'$\Delta\psi$', fontsize=10)
    ax.set_zlabel(f'$\Delta\chi_{chi_angle_name[-1]}$', fontsize=10)

    save_path = FIGURES_DIR / f"{residue_name}_{int_res}_{chi_angle_name}_3D_projection.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_all_3d_plots(df_delta):
    """Generate 3D scatter plots for all chi angles (CHI1 to CHI4)."""
    logging.info("Generating 3D Delta projection scatter plots...")
    for chi_key in ['CHI1', 'CHI2', 'CHI3', 'CHI4']:
        chi_name = chi_key.lower()
        residues = CHI_RESIDUES[chi_key]

        for sasa in SASA_TYPES:
            sasa_df = df_delta[df_delta['SASA'].str.startswith(sasa)]
            for res in residues:
                res_df = sasa_df[sasa_df['D_LABEL'].str[:3] == res]
                if len(res_df) > 0:
                    plot_3d(res, sasa, res_df['D_PHI'], res_df['D_PSI'], res_df[f"D_{chi_key}"], chi_name)


# ----------------------------------------------------
# Ramachandran Scatter Plot Helpers
# ----------------------------------------------------
def setup_ramachandran_grid(ax, rows, cols):
    """Format tick limits, labels, and gridlines on a grid of subplots."""
    for r in range(rows):
        for c in range(cols):
            curr_ax = ax[r, c]
            curr_ax.set_xlim(-180, 181)
            curr_ax.set_ylim(-180, 181)
            curr_ax.set_xticks(np.arange(-180, 181, 60))
            curr_ax.set_yticks(np.arange(-180, 181, 60))

            for val in [30, 90, 150, 180, -30, -90, -150, -180]:
                curr_ax.axvline(x=val, color='grey', linestyle='--', linewidth=0.6)
                curr_ax.axhline(y=val, color='grey', linestyle='--', linewidth=0.6)

            if c > 0:
                curr_ax.set_yticks([])


def plot_ramachandran_group(df_torsion, res_list, chi_count_list, state_prefix='B', state_title='Bound', filename_suffix='bound'):
    """Generic function to plot Ramachandran scatter plots for a list of residues and chi angle columns."""
    marker_type = 'o'
    marker_size = 8
    norm = colors.Normalize(-180, 180)
    cmap = cm.twilight_shifted

    num_rows = len(res_list)
    max_cols = max(chi_count_list)

    fig, ax = plt.subplots(num_rows, max_cols, figsize=(4 * max_cols, 4 * num_rows), squeeze=False)

    for row_idx, (res, chi_count) in enumerate(zip(res_list, chi_count_list)):
        res_data = df_torsion[df_torsion.LABEL.str[:3] == res]
        phi_vals = res_data[f"{state_prefix}_PHI"]
        psi_vals = res_data[f"{state_prefix}_PSI"]

        for col_idx in range(max_cols):
            curr_ax = ax[row_idx, col_idx]
            if col_idx < chi_count:
                chi_col = f"{state_prefix}_CHI{col_idx + 1}"
                chi_vals = res_data[chi_col]
                curr_ax.scatter(phi_vals, psi_vals, c=chi_vals, cmap=cmap, alpha=0.7, norm=norm, marker=marker_type, s=marker_size)
                curr_ax.set_title(rf'${res}\ ({state_title}\ \chi_{{{col_idx+1}}})$', fontsize=12)
            else:
                curr_ax.axis('off')

            if row_idx == num_rows - 1:
                curr_ax.set_xlabel(r'$\phi$', fontsize=12)
            if col_idx == 0:
                curr_ax.set_ylabel(r'$\psi$', fontsize=12)

    setup_ramachandran_grid(ax, num_rows, max_cols)

    smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(smap, ax=ax, fraction=0.08 / max_cols, shrink=0.7)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel(r'$\chi_{n}$', rotation=0, labelpad=15, fontdict={"size": 12})

    save_path = FIGURES_DIR / f"{filename_suffix}_scatter_plot.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


# ----------------------------------------------------
# Side-by-Side (Bound vs Unbound) Ramachandran Plots
# ----------------------------------------------------
def plot_side_by_side_ramachandran(df_torsion, res_list, chi_count_list, filename_prefix):
    """Plot side-by-side comparison of Bound vs Unbound states for amino acid groups."""
    marker_type = 'o'
    marker_size = 8
    norm = colors.Normalize(-180, 180)
    cmap = cm.twilight_shifted

    num_res = len(res_list)

    for res, chi_count in zip(res_list, chi_count_list):
        fig, ax = plt.subplots(chi_count, 2, figsize=(8, 4 * chi_count), squeeze=False)
        res_data = df_torsion[df_torsion.LABEL.str[:3] == res]

        for chi_idx in range(chi_count):
            b_phi, b_psi, b_chi = res_data["B_PHI"], res_data["B_PSI"], res_data[f"B_CHI{chi_idx + 1}"]
            u_phi, u_psi, u_chi = res_data["U_PHI"], res_data["U_PSI"], res_data[f"U_CHI{chi_idx + 1}"]

            # Bound
            ax[chi_idx, 0].scatter(b_phi, b_psi, c=b_chi, cmap=cmap, alpha=0.7, norm=norm, marker=marker_type, s=marker_size)
            ax[chi_idx, 0].set_title(rf'${res}\ (Bound\ \chi_{{{chi_idx+1}}})$', fontsize=11)
            ax[chi_idx, 0].set_ylabel(r'$\psi$', fontsize=11)

            # Unbound
            ax[chi_idx, 1].scatter(u_phi, u_psi, c=u_chi, cmap=cmap, alpha=0.7, norm=norm, marker=marker_type, s=marker_size)
            ax[chi_idx, 1].set_title(rf'${res}\ (Unbound\ \chi_{{{chi_idx+1}}})$', fontsize=11)

            if chi_idx == chi_count - 1:
                ax[chi_idx, 0].set_xlabel(r'$\phi$', fontsize=11)
                ax[chi_idx, 1].set_xlabel(r'$\phi$', fontsize=11)

        setup_ramachandran_grid(ax, chi_count, 2)

        smap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(smap, ax=ax, fraction=0.08, shrink=0.7)
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_ylabel(r'$\chi_{n}$', rotation=0, labelpad=15, fontdict={"size": 11})

        save_path = FIGURES_DIR / f"{res}_side_by_side_bound_unbound_scatter.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)


def generate_all_ramachandran_plots(df_torsion):
    """Generate Ramachandran scatter plots for Bound, Unbound, and Side-by-Side states."""
    logging.info("Generating Ramachandran scatter plots...")

    groups = [
        (["ARG", "LYS"], [4, 4], "chi4_amino_acids"),
        (["GLU", "GLN", "MET"], [3, 3, 3], "chi3_amino_acids"),
        (["ASN", "ASP", "LEU", "ILE"], [2, 2, 2, 2], "chi2_amino_acids"),
        (["CYS", "SER", "THR", "VAL"], [1, 1, 1, 1], "chi1_amino_acids"),
        (["PHE", "TRP", "TYR", "HIS"], [2, 2, 2, 2], "aromatic_amino_acids"),
        (["PRO"], [2], "proline_amino_acid")
    ]

    for res_list, chi_counts, name_suffix in groups:
        # Bound State
        plot_ramachandran_group(df_torsion, res_list, chi_counts, state_prefix='B', state_title='Bound', filename_suffix=f"Only_B_{name_suffix}")
        # Unbound State
        plot_ramachandran_group(df_torsion, res_list, chi_counts, state_prefix='U', state_title='Unbound', filename_suffix=f"Only_U_{name_suffix}")
        # Side-by-Side Comparison
        plot_side_by_side_ramachandran(df_torsion, res_list, chi_counts, name_suffix)


def main():
    input_rotamer_data = DATASET_DIR / 'PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv'
    logging.info(f"Reading dataset from: {input_rotamer_data}")
    df_torsion = pd.read_csv(input_rotamer_data, sep=",")

    # Calculate delta angles Dataframe
    df_delta = calculate_delta_angles(df_torsion)

    # 1. 3D Delta Projection Scatter Plots
    generate_all_3d_plots(df_delta)

    # 2. Ramachandran Scatter Plots (Bound, Unbound, Side-by-Side)
    generate_all_ramachandran_plots(df_torsion)

    logging.info(f"All scatter plots successfully generated and saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()