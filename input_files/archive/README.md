# Archived input datasets

Superseded torsion datasets and tables, kept unchanged for comparison with earlier results and
manuscript drafts. **No script reads them.**

The current dataset is
`output_files/torsion_dataset/PRDBv3_187_bound_unbound_protein_torsions.json`, written by
`script/calculate_187_prdbv3_protein_torsions.py` and loaded through
`utils/torsion_dataset.load_torsion_dataset()`. Its angles are identical to
`ALL_AA_..._187_clean.tsv` below. Its secondary structure is corrected, and it adds the
`U_PDB`, `U_LABEL`, `B_DSSP` and `U_DSSP` columns.

## Problems shared by the July 2026 files

All files dated 2026-07-27 (commit `4ef9c86`) come from the legacy pipeline
(`scripts_protein/chiN_calculation.py` and its helpers). Unless noted otherwise, they share
these problems:

- Missing angles were written as 0.0.
- Unbound values were taken by list position.
- Core residues were dropped.

## Files

| File | Added | Origin | Why superseded |
|---|---|---|---|
| `ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv` | 2026-10-20 (`4e01864`) | `generate_187_protein_torsion_datasets.py` after the pairing fix | Angles correct. Comma-separated despite the `.tsv` name. Secondary structure: disulfide CYS and insertion-code residues default to L; 3-state rules differ from `assign_dssp.py` |
| `ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187.json` | 2026-08-09 (`8bf63ad`) | August `generate_187_protein_torsion_datasets.py` | Unbound residues matched by label: 12,987 residues in 84 complexes have unbound angles of 0.0 or from the wrong residue. Missing angles are 0.0 |
| `PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187_clean.tsv` | 2026-07-27 (`4ef9c86`) | Legacy pipeline | Shared July problems (missing χ as 0.0, e.g. 12,512 χ2 values). No GLY/ALA. Interface and non-interface only |
| `PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_187.tsv` | 2026-07-27 | Legacy pipeline | As above |
| `PROTEIN_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv` / `.csv` | 2026-07-27 | Legacy pipeline, with ω | As above. Undefined ω written as 0.0, which classifies as cis |
| `PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_288_COMPLEXES_clean.tsv` | 2026-07-27 | Legacy pipeline, 288 bound-only complexes | Replaced by `script/calculate_288_prdbv3_protein_torsions.py` |
| `Figure_2_Tabulated_187.xlsx` | 2026-07-27 | Rotamer state counts and percentages for the Figure 2 stacked bars | Computed from the legacy data. The counts are now computed from the current dataset (`rotamer_distribution_plotter.py`, `calculate_chi_distributions.py`). Can still be passed with `--excel-file` |
| `rotamer_disrtibution.xlsx` | 2026-07-27 | χ4 rotamer state counts (one sheet) | Computed from the legacy data |
