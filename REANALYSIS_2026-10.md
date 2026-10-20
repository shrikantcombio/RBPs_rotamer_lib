# Reanalysis of the RBP rotamer study — October 2026

This note records defects found in the data pipeline behind the RBP rotamer manuscript, how they were
fixed, and how each reported result changed. All "before" values are the tables committed on `main`
(`output_files/analysis*`); "after" values come from the corrected pipeline on branch
`fix/unbound-pairing-missing-values` (this repository), `fix/rotamer-assignment-torsions`
(rota_assign), and the now version-controlled `prdbv3_unbound_rotamer_analysis` and `protein-protein`
projects.

## 1. Defects and their reach

| # | Defect | Where | Effect on data |
|---|---|---|---|
| 1 | Bound/unbound residues joined by residue label; unbound chains are usually relabelled (5X6B: IJEF vs ABCD) or renumbered (2ZZM) | `generate_187_protein_torsion_datasets.py`, `macromol_torsion/calculator.py` | 12,987 residues in 84 of 187 complexes got all unbound angles = 0.0; renumbered chains were paired with the wrong residues |
| 2 | Uncomputable angles written as 0.0 instead of missing | same, and the PPI pipeline | ~1,700 bound / ~1,500 unbound residues with truncated side chains got χ = 0°, i.e. a fake off-rotamer |
| 3 | φ/ψ/ω computed across chain breaks and chain boundaries | `macromol_torsion/protein_backbone.py` | ~0.9 % of residues (1,382 breaks) |
| 4 | Rotamer assignment compared only χ1…χ(n−1) | `rota_assign/script/get_rotastate.py` | LEU/ILE classified on χ1 only, LYS on χ1–3, MET on χ1–2; their last-χ transitions were invisible |
| 5 | Analysis tables not reproducible from the repository | `prdbv3_unbound_rotamer_analysis` | Analyses 2–5 read a file that no longer existed; the tables for analyses 2–4 predate the dataset recalculation of 8 Aug (6,336 vs 5,858 interface residues) |
| 6 | Synthetic PPI comparison data | `analysis_5_rbp_vs_ppi_rotamer_divergence.py` | Interface = every 4th residue, χ2 = 180°, φ/ψ = −60/−40° for all, "unbound" rotamer from the bound χ1 |
| 7 | Rotamer-recovery benchmark measured nothing | `analysis_6_rotamer_recovery_benchmark.py` | "PPI library" was the RBP SSD library, which failed to load (0 % recovery); reference rotamers came from the evaluated library; test complexes were in the RBP library |
| 8 | PPI unbound columns filled with bound values when pairing failed | `protein-protein/script/run_ppi_torsion_recalculation.py` | 15.7 % (partner 1) and 23.6 % (partner 2) of rows; all rows of 18 and 35 complexes |
| 9 | DSSP/DSSR caches keyed by PDB code | `macromol_torsion/dssp.py`, `dssr.py` | Bound and unbound files of one complex could share a secondary-structure result (no effect on the regenerated data: the per-file `.dssp` files were used) |

Smaller fixes: insertion codes kept in residue labels; rota_assign's library cache now rebuilds when the
libraries change and the SSD library loads; analysis 1 no longer assigns Core/Rim interface by row
parity when a PRince file is missing (43 residues; the label was not used in any statistic);
`GAL → GLU` removed from the structure cleaner's modified-residue map.

## 2. Rotamer libraries

| Library | Residues before → after | χ = 0 placeholder rotamers before → after | Fake (0°, 0°) cell before → after |
|---|---|---|---|
| BBD unbound, all (`RBPs_bbd_U`) | 63,056 → 60,328 | 13,452 → 0 | 12,993 (20.6 %) → 0 |
| BBD bound, all (`RBPs_bbd_B`) | 63,010 → 60,485 | 449 → 0 | 0 → 0 |
| BBI unbound, all (`RBPs_bbi_U`) | 65,574 → 63,662 | 5,152 → 0 | — |
| BBI bound, all (`RBPs_bbi_B`) | 65,574 → 63,700 | 547 → 0 | — |

In the BBI unbound library the most probable rotamer changed for 8 of 18 amino acids; bound top
rotamers are unchanged. Rerunning the dataset and library scripts reproduces all 49 files byte for byte.

### 2a. v2 libraries: terminal sp2 chis (branch `fix/sp2-terminal-chi-rotamers`)

The libraries above (v1) binned every chi into three 120° bins. For the planar terminal group of ASP,
GLU, PHE, TYR, ASN, GLN, HIS and TRP this created rotamers that are not distinct conformations:
PHE/TYR χ2 was folded into [−90°, 90°), whose boundary lies on the most populated ring orientation
(±90°), so one rotamer was reported as two (BBI bound PHE "m −70°" P = 0.38 and "m +79°" P = 0.17 are
the same ring orientation); the 0° edge split the single ASP χ2 / GLU χ3 peak (+27° / −30° pairs), and
the TRP χ2 mode near 0°. v2 bins these chis by residue-specific ranges at density minima and computes
statistics of symmetric groups on doubled angles (README, "Terminal sp2 chis"). Other residues are
unchanged, apart from 21 bound / 27 unbound BBD cells (|ΔP| ≤ 0.05) affected by the corrected
backbone-bin edges (ties at exact bin edges, 165° labelled 180°).

v1 files are kept unchanged in `RBPs_BBD_rotamer_lib/`, `RBPs_BBI_rotamer_lib/` and
`protein-protein/library/`; v2 is in the `_v2` directories.

| Library | Rows v1 → v2 |
|---|---|
| BBD bound / unbound (all residues) | 1,732 → 1,542 / 1,718 → 1,552 |
| BBI bound / unbound (all residues) | 242 → 209 / 248 → 213 |

Most probable BBI bound rotamer (terminal χ shown as an angle; PHE/TYR χ2 100° ≡ −80°):

| AA | Rotamers v1 → v2 | Top v1 (P) | Top v2 (P) |
|---|---|---|---|
| ASN | 9 → 6 | m-40 (0.40) | m-36 (0.46) |
| ASP | 6 → 3 | m-30 (0.42) | m-24 (0.52) |
| GLN | 23 → 9 | mt-42 (0.17) | mt-4 (0.37) |
| GLU | 17 → 9 | mt-31 (0.17) | mt-2 (0.34) |
| HIS | 9 → 9 | m-72 (0.29) | m-72 (0.29) |
| PHE | 6 → 3 | m-70 (0.38) | m100 (0.55) |
| TRP | 8 → 9 | m94 (0.32) | m99 (0.30) |
| TYR | 6 → 3 | m-70 (0.36) | m100 (0.52) |

Analyses 1–4 compare only the letter part of rotamer names (sp3 chis), but the names come from
rota_assign's own copy of the libraries (still v1); they are to be rerun once rota_assign uses v2.
Analysis 6 predicts the most probable library row and must be rerun with v2.

## 3. Results

### Analysis 1 — interface vs non-interface rotamer transitions
| | Before | After |
|---|---|---|
| Interface transition rate | 49.93 % (n = 5,858) | 41.21 % (n = 5,545) |
| Non-interface transition rate | 37.53 % (n = 53,581) | 28.82 % (n = 50,787) |
| Odds ratio (Fisher) | 1.66, p = 1.2e-74 | 1.73, p = 2.1e-77 |
| Amino acids significant after FDR | 13 | 13 |

**Conclusion unchanged**: interface residues change rotamer more often than non-interface residues.

### Analysis 2 — off-rotamer energetics
| | Before | After |
|---|---|---|
| Interface OFF→ON | 138 (2.18 %) | 43 (0.78 %) |
| Interface ON→OFF | 100 (1.58 %) | 36 (0.65 %) |
| Non-interface OFF→ON | 820 (1.57 %) | 489 (0.96 %) |
| Interface ΔE, OFF→ON / ON→OFF | 0.069 / 0.036 | 0.360 / −0.359 |

Many former "off-rotamers" were χ = 0 placeholders. OFF↔ON switches are about 3× rarer than reported.

### Analysis 3 — conformational selection vs induced fit
| | Before | After |
|---|---|---|
| Interface rigid / CS / IF | 61.7 / 26.0 / 12.3 % | 58.8 / 30.2 / 11.0 % |
| Non-interface rigid / CS / IF | 73.8 / 16.9 / 9.3 % | 71.2 / 17.5 / 11.3 % |
| Transitioning interface CS : IF | 67.8 : 32.2 | 73.3 : 26.7 |
| Fisher odds ratio | 1.15, p = 0.0027 | 1.77, p = 1.9e-31 |

**Conclusion unchanged and stronger.**

### Analysis 4 — flexibility and structural classes
Interface transition rate, R / S / F: 35.8 / 39.9 / 43.2 % → 39.3 / 42.4 / 44.4 % (same ordering).
By RNA structural class (A tRNA, B rRNA, C duplex, D ssRNA): 40.0 / 46.6 / 42.7 / 33.6 % →
40.9 / 41.5 / 45.4 / 38.3 %; class B is no longer the highest, class C is.

### Analysis 5 — RBP vs PPI interface rotamer preferences
Rebuilt on the real Protein-Protein Docking Benchmark 5.5 data (both partners; PRince interface labels;
both datasets classified with the same library-free rule).

| | Before (synthetic PPI data) | After |
|---|---|---|
| Largest JS divergence | ARG, LYS = 1.00 bit (maximum) | MET 0.16, LYS 0.16, CYS 0.12 bit |
| Amino acids with a significant difference | 16 of 18 (p < 0.05) | 1 of 18 (CYS, FDR q = 1.2e-5) |

**The claimed RBP-specific rotamer preferences (including aromatic residues) are not supported.**

### Analysis 6 — rotamer recovery, RBP vs PPI library
Rebuilt as a real benchmark: predict the most probable rotamer from the bound backbone cell; reference
class from the residue's own χ angles; RBP library with 5-fold cross-validation by complex; PPI library
from Benchmark 5.5.

| | Before | After |
|---|---|---|
| Interface recovery, RBP library | 82.35 % | 38.80 % (coverage 90.0 %) |
| Interface recovery, PPI library | 0.00 % (library never loaded) | 42.63 % (coverage 98.5 %) |
| Gain (RBP − PPI) | +82.35 % | −3.83 % (McNemar p = 5.6e-14) |
| Residues covered by both libraries | — | 43.10 % vs 44.18 % (n = 5,525) |
| Non-interface recovery, RBP / PPI | 74.07 / 0.00 % | 50.60 / 50.50 % |

**The claimed advantage of the RBP library is not supported**: it performs on par with (residues both
libraries cover) or slightly worse than (all residues, owing to lower coverage) a library built from
protein–protein interfaces.

## 4. Not yet redone

- Figures produced by the RBPs_rotamer_lib plotting scripts must be regenerated from the corrected
  dataset before submission.
- RNA-side torsions (`rna_sidechain.py`, `calculator.analyze_rna_7d_suite`) still default missing values
  to 0.0; the NtC/CANA analyses were not re-examined.
- DockBench-X outputs that use rota_assign (`matched_protein_rotamer_transitions.csv`) were produced with
  the old rotamer assignment.
- `input_files/PROTEIN_BACKBONE_SIDECHAIN_TORSION_ANGLES_288_COMPLEXES_clean.tsv` (288-complex set,
  bound only) was produced before the fixes and still contains χ = 0.0 placeholders. Its script is fixed
  (NaN, configured paths) but was not rerun because it repeats the PRince batch; no analysis reads it.
