#!/usr/bin/env python3
"""
calculate_chi_distributions.py

Rotamer state tables behind the stacked side-chain distribution figures (Figure 2): for every chi,
residue type, region (interface I / non-interface surface N) and form (bound B / unbound U), the
counts and percentages of the six chi states.

Counts come from RBPRotamerDistributionPlotter.calculate_rotamer_tables_from_json, the same code
that draws the figures, so the tables and figures always agree:
  - states   : T (-30, 30], p (30, 90], P (90, 150], t (> 150 or <= -150), M (-150, -90],
               m (-90, -30]; p / t / m are the staggered minima of an sp3 chi
  - residues : only those with the chi present in both the bound and the unbound structure, so B
               and U of a residue type share one total
  - missing  : angles that could not be computed are empty in the dataset and are not counted

The 6-state letters describe the angle, not whether it is a rotamer for every chi: the terminal
sp2 chis (ASP/PHE/TYR/HIS/TRP/ASN chi2, GLU/GLN chi3) are planar groups whose normal positions
fall in T / P / M (e.g. PHE chi2 ~ 90 is P), and ARG chi5 is planar with chi5 ~ 0 (T) the normal
state. These rows are flagged in the summary (column Planar_chi), so their T / P / M are not read as
off-rotamer.

Outputs (output_files/):
  Figure_2_Tabulated_187_calculated.xlsx  sheets Chi<n>_all (percentages) and chi<n>_int /
                                          chi<n>_nonint (counts)
  chi_angle_distributions_summary.tsv     the Chi<n>_all tables of every chi

Usage:
    python script/calculate_chi_distributions.py [--dataset-file FILE]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

import rotamer_definitions as rotdef
from rotamer_distribution_plotter import RBPRotamerDistributionPlotter, TORSION_DATASET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = ROOT_DIR / "output_files"

STATES = ["p", "t", "m", "T", "P", "M"]
REGIONS = [("I", "int"), ("N", "nonint")]


def is_planar_chi(aa, chi):
    """Terminal sp2 chi (rotamer_definitions.TERMINAL_SP2_CHI) or ARG chi5."""
    return rotdef.terminal_sp2(aa, chi - 1) is not None or (aa == "ARG" and chi == 5)


def counts_by_residue(sheet):
    """Counts table of one chi and region, indexed by (amino acid, form)."""
    sheet = sheet.assign(Amino_acid=sheet["Amino_acid"].str.replace(r"_[IN]$", "", regex=True))
    return sheet.set_index(["Amino_acid", "forms"])


def summary_table(tables, chi):
    """
    Percentage of each state per residue type, form and region, with a final 'Total' row pooled
    over residue types (total counts / total residues, not a mean of percentages).
    """
    counts = {region: counts_by_residue(tables[f"chi{chi}_{label}"]) for region, label in REGIONS}
    residues = list(dict.fromkeys(counts["I"].index.get_level_values(0)))

    def row(name, n, state_counts, planar):
        rec = {"Amino_acid": name, "Total": n["I"] + n["N"], "Int": n["I"], "non_int": n["N"],
               "Planar_chi": planar}
        for state in STATES:
            for form in "BU":
                for region in ("I", "N"):
                    tot = n[region]
                    rec[f"{form}_{region}_{state}_pct"] = (
                        round(state_counts[(form, region)][state] / tot * 100.0, 1) if tot else float("nan"))
        return rec

    rows = []
    for aa in residues:
        n = {region: int(counts[region].loc[(aa, "B"), "Total"]) for region in ("I", "N")}
        state_counts = {(form, region): counts[region].loc[(aa, form), STATES]
                        for form in "BU" for region in ("I", "N")}
        rows.append(row(aa, n, state_counts, is_planar_chi(aa, chi)))

    n_all = {region: int(counts[region].xs("B", level="forms")["Total"].sum()) for region in ("I", "N")}
    pooled = {(form, region): counts[region].xs(form, level="forms")[STATES].sum()
              for form in "BU" for region in ("I", "N")}
    rows.append(row("Total", n_all, pooled, ""))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Chi rotamer state tables (counts and percentages).")
    parser.add_argument("--dataset-file", type=Path, default=TORSION_DATASET, help="Torsion angle dataset.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR, help="Output directory.")
    args = parser.parse_args()

    logging.info(f"Loading dataset from: {args.dataset_file}")
    tables = RBPRotamerDistributionPlotter(dataset_file=args.dataset_file).calculate_rotamer_tables_from_json()
    if tables is None:
        raise SystemExit(f"Torsion dataset not found: {args.dataset_file}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_excel = args.output_dir / "Figure_2_Tabulated_187_calculated.xlsx"
    summaries = []
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for chi in range(1, 6):
            summary = summary_table(tables, chi)
            summary.to_excel(writer, sheet_name=f"Chi{chi}_all", index=False)
            summaries.append(summary.assign(Chi_Angle=f"CHI{chi}"))
            for _, label in REGIONS:
                tables[f"chi{chi}_{label}"].to_excel(writer, sheet_name=f"chi{chi}_{label}", index=False)
    logging.info(f"Saved Excel tables to: {output_excel}")

    tsv_file = args.output_dir / "chi_angle_distributions_summary.tsv"
    combined = pd.concat(summaries, ignore_index=True)
    combined = combined[["Chi_Angle"] + [c for c in combined.columns if c != "Chi_Angle"]]
    combined.to_csv(tsv_file, sep="\t", index=False)
    logging.info(f"Saved summary TSV to: {tsv_file}")


if __name__ == "__main__":
    main()
