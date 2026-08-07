import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATASET_DIR = ROOT_DIR / "input_files"
RESULTS_DIR = ROOT_DIR / "output_files"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Definition of amino acid residues per chi angle
RESIDUE_GROUPS = {
    'CHI1': ["ARG", "ASN", "ASP", "GLU", "GLN", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "TRP", "TYR", "CYS", "SER", "THR", "VAL"],
    'CHI2': ["ARG", "ASN", "ASP", "GLU", "GLN", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "TRP", "TYR"],
    'CHI3': ["ARG", "LYS", "GLU", "GLN", "MET"],
    'CHI4': ["ARG", "LYS"],
    'CHI5': ["ARG"]
}

# Rotamer bin classification
# On-rotamers:  g+ (p), t (t), g- (m)
# Off-rotamers: gg (T), gt (P), tg (M)
def classify_rotamer(angle_val):
    if pd.isna(angle_val):
        return None
    val = float(angle_val)
    if -30.0 < val <= 30.0:
        return 'T'  # gg / off-rotamer
    elif 30.0 < val <= 90.0:
        return 'p'  # g+ / on-rotamer
    elif 90.0 < val <= 150.0:
        return 'P'  # gt / off-rotamer
    elif val > 150.0 or val <= -150.0:
        return 't'  # t  / on-rotamer
    elif -150.0 < val <= -90.0:
        return 'M'  # tg / off-rotamer
    elif -90.0 < val <= -30.0:
        return 'm'  # g- / on-rotamer
    return None


def calculate_chi_distribution_df(df_torsion, chi_name, residues):
    """Calculate rotamer counts and percentages for a specific Chi angle across residues."""
    b_col = f"B_{chi_name}"
    u_col = f"U_{chi_name}"

    df_work = df_torsion.copy()
    df_work['B_ROT'] = df_work[b_col].apply(classify_rotamer)
    df_work['U_ROT'] = df_work[u_col].apply(classify_rotamer)

    records = []
    tot_b_int, tot_b_nint = 0, 0
    tot_u_int, tot_u_nint = 0, 0

    rot_states = ['p', 't', 'm', 'T', 'P', 'M']

    for res in residues:
        res_df = df_work[df_work['LABEL'].str[:3] == res]

        b_i = res_df[res_df['SASA'] == 'I']
        b_n = res_df[res_df['SASA'] == 'N']
        u_i = res_df[res_df['SASA'] == 'I']
        u_n = res_df[res_df['SASA'] == 'N']

        n_b_i = len(b_i)
        n_b_n = len(b_n)
        n_u_i = len(u_i)
        n_u_n = len(u_n)

        tot_b_int += n_b_i
        tot_b_nint += n_b_n
        tot_u_int += n_u_i
        tot_u_nint += n_u_n

        rec = {
            'Amino_acid': res,
            'Total': len(res_df),
            'Int': n_b_i,
            'non_int': n_b_n
        }

        # Calculate percentages for on and off rotamers
        for state in rot_states:
            rec[f'B_I_{state}_pct'] = round((b_i['B_ROT'] == state).sum() / n_b_i * 100.0, 1) if n_b_i > 0 else 0.0
            rec[f'B_N_{state}_pct'] = round((b_n['B_ROT'] == state).sum() / n_b_n * 100.0, 1) if n_b_n > 0 else 0.0
            rec[f'U_I_{state}_pct'] = round((u_i['U_ROT'] == state).sum() / n_u_i * 100.0, 1) if n_u_i > 0 else 0.0
            rec[f'U_N_{state}_pct'] = round((u_n['U_ROT'] == state).sum() / n_u_n * 100.0, 1) if n_u_n > 0 else 0.0

        records.append(rec)

    res_df_out = pd.DataFrame(records)

    # Average / Total row
    avg_rec = {
        'Amino_acid': 'Total',
        'Total': res_df_out['Total'].sum(),
        'Int': tot_b_int,
        'non_int': tot_b_nint
    }
    for state in rot_states:
        avg_rec[f'B_I_{state}_pct'] = round(res_df_out[f'B_I_{state}_pct'].mean(), 2)
        avg_rec[f'B_N_{state}_pct'] = round(res_df_out[f'B_N_{state}_pct'].mean(), 2)
        avg_rec[f'U_I_{state}_pct'] = round(res_df_out[f'U_I_{state}_pct'].mean(), 2)
        avg_rec[f'U_N_{state}_pct'] = round(res_df_out[f'U_N_{state}_pct'].mean(), 2)

    res_df_out = pd.concat([res_df_out, pd.DataFrame([avg_rec])], ignore_index=True)
    return res_df_out


def create_detailed_counts_df(df_torsion, chi_name, residues, sasa_type='I'):
    """Create counts table for interface or non-interface for a Chi angle."""
    b_col = f"B_{chi_name}"
    u_col = f"U_{chi_name}"

    df_work = df_torsion[df_torsion['SASA'] == sasa_type].copy()
    df_work['B_ROT'] = df_work[b_col].apply(classify_rotamer)
    df_work['U_ROT'] = df_work[u_col].apply(classify_rotamer)

    records = []
    rot_states = ['T', 'p', 'P', 't', 'M', 'm']

    for res in residues:
        res_df = df_work[df_work['LABEL'].str[:3] == res]
        tot = len(res_df)

        for form_type, col_name in [('B', 'B_ROT'), ('U', 'U_ROT')]:
            rec = {'Amino_acid': res, 'forms': form_type}
            for st in rot_states:
                rec[st] = (res_df[col_name] == st).sum()
            rec['Total'] = tot
            records.append(rec)

    return pd.DataFrame(records)


def main():
    input_file = DATASET_DIR / "PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"
    logging.info(f"Loading dataset from: {input_file}")
    df_torsion = pd.read_csv(input_file, sep=",")

    output_excel = RESULTS_DIR / "Figure_2_Tabulated_187_calculated.xlsx"
    tsv_summary_file = RESULTS_DIR / "chi_angle_distributions_summary.tsv"

    writer = pd.ExcelWriter(output_excel, engine='openpyxl')
    all_summary_dfs = []

    for chi_name in ['CHI1', 'CHI2', 'CHI3', 'CHI4', 'CHI5']:
        residues = RESIDUE_GROUPS[chi_name]

        # Calculate percentage summary table
        summary_df = calculate_chi_distribution_df(df_torsion, chi_name, residues)
        summary_df.to_excel(writer, sheet_name=f"{chi_name.capitalize()}_all", index=False)

        summary_df_copy = summary_df.copy()
        summary_df_copy.insert(0, 'Chi_Angle', chi_name)
        all_summary_dfs.append(summary_df_copy)

        # Calculate interface and non-interface counts
        df_int = create_detailed_counts_df(df_torsion, chi_name, residues, sasa_type='I')
        df_int.to_excel(writer, sheet_name=f"{chi_name.lower()}_int", index=False)

        df_nonint = create_detailed_counts_df(df_torsion, chi_name, residues, sasa_type='N')
        df_nonint.to_excel(writer, sheet_name=f"{chi_name.lower()}_nonint", index=False)

    writer.close()
    logging.info(f"Successfully saved calculated Excel tables to: {output_excel}")

    final_tsv_df = pd.concat(all_summary_dfs, ignore_index=True)
    final_tsv_df.to_csv(tsv_summary_file, sep='\t', index=False)
    logging.info(f"Successfully saved summary TSV to: {tsv_summary_file}")


if __name__ == "__main__":
    main()
