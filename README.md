# RNA Binding Proteins (RBPs) Rotamer Libraries

A high-precision, circular-statistics-derived backbone-dependent (BBD) and backbone-independent (BBI) rotamer library suite for RNA-binding proteins in bound and unbound states across interface and surface locations.

---

## 📁 Directory Structure

```text
RBPs_rotamer_lib/
├── RBPs_BBD_rotamer_lib_v2/      # Current BBD libraries (v2, same file names as below)
├── RBPs_BBI_rotamer_lib_v2/      # Current BBI libraries (v2, same file names as below)
├── RBPs_BBD_rotamer_lib/         # v1 Backbone-Dependent (30° x 30° Phi/Psi bins) Libraries, kept unchanged
│   ├── RBPs_I_bbd_B_rotamer_lib.[csv|tsv|json]   # Bound Interface (I)
│   ├── RBPs_I_bbd_U_rotamer_lib.[csv|tsv|json]   # Unbound Interface (I)
│   ├── RBPs_N_bbd_B_rotamer_lib.[csv|tsv|json]   # Bound Non-Interface Surface (N)
│   ├── RBPs_N_bbd_U_rotamer_lib.[csv|tsv|json]   # Unbound Non-Interface Surface (N)
│   ├── RBPs_S_bbd_B_rotamer_lib.[csv|tsv|json]   # Bound Surface (S = I + N)
│   ├── RBPs_S_bbd_U_rotamer_lib.[csv|tsv|json]   # Unbound Surface (S = I + N)
│   ├── RBPs_bbd_B_rotamer_lib.[csv|tsv|json]     # Bound, all residues (I + N + C)
│   └── RBPs_bbd_U_rotamer_lib.[csv|tsv|json]     # Unbound, all residues (I + N + C)
├── RBPs_BBI_rotamer_lib/         # v1 Backbone-Independent (Global) Libraries, kept unchanged
│   ├── RBPs_I_bbi_B_rotamer_lib.[csv|tsv|json]   # Bound Interface (I)
│   ├── RBPs_I_bbi_U_rotamer_lib.[csv|tsv|json]   # Unbound Interface (I)
│   ├── RBPs_N_bbi_B_rotamer_lib.[csv|tsv|json]   # Bound Non-Interface Surface (N)
│   ├── RBPs_N_bbi_U_rotamer_lib.[csv|tsv|json]   # Unbound Non-Interface Surface (N)
│   ├── RBPs_S_bbi_B_rotamer_lib.[csv|tsv|json]   # Bound Surface (S = I + N)
│   ├── RBPs_S_bbi_U_rotamer_lib.[csv|tsv|json]   # Unbound Surface (S = I + N)
│   ├── RBPs_bbi_B_rotamer_lib.[csv|tsv|json]     # Bound, all residues (I + N + C)
│   └── RBPs_bbi_U_rotamer_lib.[csv|tsv|json]     # Unbound, all residues (I + N + C)
├── input_files/                  # Cleaned input datasets (PRDBv3 187/288 complexes)
├── output_files/                 # Torsion angle dataset exports (CSV, TSV, JSON)
├── script/                       # Core python engines for derivation & analysis
└── utils/                        # Structure parsing utilities (mmcif_clean_reader)
```

---

## ⚙️ Implementation & Derivation Methodology

The rotamer library derivation pipeline incorporates modern circular statistics and kernel regression methods (Zhang et al., *Bioinformatics* 2024, doi:[10.1093/bioinformatics/btae444](https://doi.org/10.1093/bioinformatics/btae444)):

### 1. Torsion Angle Calculation Engine
- Built on `macromol_torsion` (`geometry.py`, `protein_backbone.py`, `protein_sidechain.py`).
- Enforces strict **IUPAC right-handed sign convention** ($\pm 180^\circ$) matching BioPython `calc_dihedral` with 100.00% precision.
- Assigns 3-state secondary structure via DSSP (`get_secondary_structure`).

### 2. SASA & Spatial Residue Classification Logic
Residues are categorized based on surface accessibility and interface burial using PRince (`.int` and `.sur` outputs):
- **Interface Residues (`'I'`)**: Residues present in PRince `.int` output files that lose solvent accessible surface area (SASA) upon complexation (buried in the binding interface).
- **Non-Interface Surface Residues (`'N'`)**: Residues present in PRince `.sur` output files (surface-exposed residues) that are *not* involved in the binding interface.
- **Core Buried Residues (`'C'`)**: Interior buried residues with zero surface accessibility (absent from both `.int` and `.sur` files).
- **Surface Population (`'S'`)**: All surface-exposed residues combined ($\text{Surface } S = \text{Interface } I + \text{Non-Interface } N$, derived by filtering `SASA.isin(["I", "N", "S"])`).
- **Overall Population (`'ALL'`)**: Complete dataset combining Interface, Non-Interface Surface, and Core Interior residues ($I + N + C$).

### 3. Depth-Valid Sample Masking & Symmetry Folding
- **Multi-Chi Depth Validation**: Residues with missing terminal atoms are validated up to their deepest non-null chi depth, ensuring partial sidechains contribute accurately without introducing `NaN` binning artifacts.
- **Terminal sp2 chis (v2)**: the last chi of ASP, GLU, PHE, TYR, ASN, GLN, HIS and TRP rotates a planar
  group and has no staggered minima, so it is binned by residue-specific ranges whose edges lie at density
  minima (`script/rotamer_definitions.py`, following Dunbrack 2010/2011 and Lovell et al. 2000):

  | Residue χ | Period | Canonical range | Bins (edges) | Modes |
  |---|---|---|---|---|
  | ASP χ2, GLU χ3 | 180° | [−90°, 90°) | 1 | ~0° |
  | PHE χ2, TYR χ2 | 180° | [−30°, 150°) | 1 | ~90° |
  | HIS χ2 | 360° | — | 3 (−120°, 0°, 120°) | ~−75°, ~75°, ~170° |
  | TRP χ2 | 360° | — | 3 (−50°, 50°, 180°) | ~−100°, ~0°, ~90° |
  | ASN χ2 | 360° | — | 2 (−150°, 60°) | ~−30°, ~120° |
  | GLN χ3 | 360° | — | 1 | broad |

  For the two-fold symmetric groups (OD1/OD2, OE1/OE2, ring flip) the mean and σ are computed on doubled
  angles, so x and x + 180° are the same conformation. In v1 every chi used the three 120° bins after
  folding symmetric chis into [−90°, 90°): the fold boundary fell on the most populated PHE/TYR ring
  orientation (±90°) and the 0° edge split the single ASP χ2 / GLU χ3 peak, so these residues had
  roughly twice as many rotamers as real conformations (e.g. BBI PHE: 6 rows in v1, 3 in v2).

### 4. Von Mises Kernel-Weighted Circular Statistics
- **Grid Binning**: Backbone $(\Phi, \Psi)$ angles are binned into $30^\circ \times 30^\circ$ grid cells centered at $[-180^\circ, -150^\circ, \dots, +150^\circ]$; each cell is $[g - 15^\circ, g + 15^\circ)$ (v1 rounded ties to even and labelled 165° as 180°).
- **Probabilities**: `PROB` = rotamer count / all residues in the cell (BBD) or of the residue type (BBI). Rotamers with fewer than 5 observations are not reported, so the probabilities of a cell can sum to less than 1.
- **Rotamer States**: Sidechain $\chi$ angles are binned into standard staggered states ($p / g^+$ at $+60^\circ$, $t$ at $180^\circ$, $m / g^-$ at $-60^\circ$).
- **Circular Statistics**: Weighted circular mean ($\mu = \text{angle}(\sum w_i e^{j\theta_i})$) and circular standard deviation ($\sigma$) are calculated using `scipy.stats.vonmises` fitting, eliminating artificial $-180^\circ / +180^\circ$ boundary discontinuities.

### 5. Unified Multi-Format Library Outputs
- All rotamer libraries are exported in unified **CSV**, **TSV**, and **JSON** files combining all 18 standard sidechain amino acids for each state and surface classification.

---

## 🌀 Backbone-Dependent (BBD) Rotamer Libraries

### File Descriptions

| File Name | Description |
|---|---|
| `RBPs_I_bbd_B_rotamer_lib.[csv\|tsv\|json]` | Bound interface rotamers ($I$, Bound $B$) |
| `RBPs_I_bbd_U_rotamer_lib.[csv\|tsv\|json]` | Unbound interface rotamers ($I$, Unbound $U$) |
| `RBPs_N_bbd_B_rotamer_lib.[csv\|tsv\|json]` | Bound non-interface rotamers ($N$, Bound $B$) |
| `RBPs_N_bbd_U_rotamer_lib.[csv\|tsv\|json]` | Unbound non-interface rotamers ($N$, Unbound $U$) |
| `RBPs_S_bbd_B_rotamer_lib.[csv\|tsv\|json]` | Bound overall surface rotamers ($S$, Bound $B$) |
| `RBPs_S_bbd_U_rotamer_lib.[csv\|tsv\|json]` | Unbound overall surface rotamers ($S$, Unbound $U$) |
| `RBPs_bbd_B_rotamer_lib.[csv\|tsv\|json]` | All bound residues, including core ($I + N + C$, Bound $B$) |
| `RBPs_bbd_U_rotamer_lib.[csv\|tsv\|json]` | All unbound residues, including core ($I + N + C$, Unbound $U$) |

### Column Schema

| Column | Description |
|---|---|
| `AA` | Amino acid 3-letter code (e.g. `ARG`, `LYS`) |
| `PHI` | Backbone $\Phi$ bin center angle (degrees) |
| `PSI` | Backbone $\Psi$ bin center angle (degrees) |
| `_Count` | Number of observations in specific $(\Phi, \Psi, \chi)$ rotamer bin |
| `Count` | Total observations in the $(\Phi, \Psi)$ backbone bin |
| `Prob` | Conditional probability $P(\text{rotamer} \mid \Phi, \Psi)$ |
| `B_CHI[1-4]` / `U_CHI[1-4]` | Weighted circular mean $\chi_n$ angle (degrees) |
| `B_CHI[1-4]Sig` / `U_CHI[1-4]Sig` | Weighted circular standard deviation of $\chi_n$ angle (degrees) |

---

## 📊 Backbone-Independent (BBI) Rotamer Libraries

### File Descriptions

| File Name | Description |
|---|---|
| `RBPs_I_bbi_B_rotamer_lib.[csv\|tsv\|json]` | Bound interface rotamers ($I$, Backbone Independent) |
| `RBPs_I_bbi_U_rotamer_lib.[csv\|tsv\|json]` | Unbound interface rotamers ($I$, Backbone Independent) |
| `RBPs_N_bbi_B_rotamer_lib.[csv\|tsv\|json]` | Bound non-interface rotamers ($N$, Backbone Independent) |
| `RBPs_N_bbi_U_rotamer_lib.[csv\|tsv\|json]` | Unbound non-interface rotamers ($N$, Backbone Independent) |
| `RBPs_S_bbi_B_rotamer_lib.[csv\|tsv\|json]` | Bound overall surface rotamers ($S$, Backbone Independent) |
| `RBPs_S_bbi_U_rotamer_lib.[csv\|tsv\|json]` | Unbound overall surface rotamers ($S$, Backbone Independent) |
| `RBPs_bbi_B_rotamer_lib.[csv\|tsv\|json]` | All bound residues, including core (Backbone Independent) |
| `RBPs_bbi_U_rotamer_lib.[csv\|tsv\|json]` | All unbound residues, including core (Backbone Independent) |

### Column Schema

| Column | Description |
|---|---|
| `AA` | Amino acid 3-letter code |
| `_Count` | Observations in specific rotamer state |
| `Count` | Total observations for amino acid |
| `Prob` | Global rotamer probability $P(\text{rotamer})$ |
| `CHI[1-4]` | Global weighted circular mean $\chi_n$ angle (degrees) |
| `CHI[1-4]Sig` | Global weighted circular standard deviation of $\chi_n$ angle (degrees) |

---

## 🔁 Regenerating the Dataset and Libraries

1. Point the scripts at your data: `cp config/paths.example.json config/paths.json` and set
   `prdbv3_rotamer_dir` (equal-residue `<C_PDB>_<U_PDB>_C.pdb` / `_U.pdb` pairs with PRince `.int`/`.sur`
   files), `rota_assign_dir` and `dssp_bin`.
2. Torsion dataset: `python script/generate_187_protein_torsion_datasets.py`
   (writes `input_files/ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv`, the per-complex
   pairing log `output_files/..._187_pairing_log.csv`, and CSV/TSV/JSON copies in `output_files/`).
3. Libraries: `python script/derive_all_updated_rotamer_libraries.py` (writes the v2 libraries to
   `RBPs_BBD_rotamer_lib_v2/`, `RBPs_BBI_rotamer_lib_v2/` and `protein-protein/library_v2/`; the v1
   directories are never overwritten).
4. Tests: `python test/test_rotamer_definitions.py`.
5. Transition matrices: `python script/rotamer_transition_matrix.py` (writes
   `output_files/transition_matrices_v2/` and `figure(s)/transition_matrices_v2/`).
6. Backbone log-odds: `python script/backbone_logodds_analysis_final.py --torsion
   input_files/ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv --bbd-dir
   ../rota_assign/rota_lib_dict/v2/bbd --output-dir output_files/logodds_v2` (the BBD cache is built
   by rota_assign on first use).

Both steps are deterministic: rerunning them on the same inputs reproduces the committed files byte for byte.

**Bound/unbound pairing.** Unbound residues are paired with bound residues by chain order and sequence
(`rota_assign/macromol_torsion/pairing.py`), not by residue label, because unbound chains are usually
relabelled or renumbered.

**Missing values.** Angles that cannot be computed are left empty (NaN in pandas, `null` in JSON),
never written as 0.0: chain termini and residues next to chain breaks (no φ or ψ), missing side-chain
atoms, and residues without an unbound partner (all `U_*` columns empty). Backbone-dependent libraries
use only residues with both φ and ψ defined.

---

## 💻 Usage Example

```python
import pandas as pd

# Load Backbone-Dependent Bound Interface Rotamer Library
df_bbd = pd.read_csv("RBPs_BBD_rotamer_lib_v2/RBPs_I_bbd_B_rotamer_lib.csv")

# Filter rotamers for Arg at phi = -60, psi = -30
arg_rotamers = df_bbd[(df_bbd["AA"] == "ARG") & (df_bbd["PHI"] == -60) & (df_bbd["PSI"] == -30)]
print(arg_rotamers[["AA", "PHI", "PSI", "PROB", "B_CHI1", "B_CHI2", "B_CHI1Sig"]])
```

---

## 🔗 References

1. Kant, S., Nithin, C., Mukherjee, S., Maity, A., Bahadur, R.P., 2025. Protein–RNA Docking Benchmark v3.0 Integrated With Binding Affinity. *Proteins*, prot.26825. https://doi.org/10.1002/prot.26825
2. Zhang, O., Naik, S.A., Liu, Z.H., Forman-Kay, J., Head-Gordon, T., 2024. A curated rotamer library for common post-translational modifications of proteins. *Bioinformatics* 40. https://doi.org/10.1093/bioinformatics/btae444
3. Shapovalov, M.V., Dunbrack, R.L., 2011. A Smoothed Backbone-Dependent Rotamer Library for Proteins Derived from Adaptive Kernel Density Estimates and Regressions. *Structure* 19, 844–858. https://doi.org/10.1016/j.str.2011.03.019
4. Dunbrack, R.L., Karplus, M., 1993. Backbone-dependent Rotamer Library for Proteins Application to Side-chain Prediction. *Journal of Molecular Biology* 230, 543–574. https://doi.org/10.1006/jmbi.1993.1170
5. Mukherjee, S., Bahadur, R.P., 2018. An account of solvent accessibility in protein-RNA recognition. *Sci Rep* 8, 10546. https://doi.org/10.1038/s41598-018-28373-2

---

## 📄 Citation

If you use these rotamer libraries or codebase in your work, please cite:

Sunandan Mukherjee, Shri Kant, and Ranjit P. Bahadur (2026). *Conformational Transitions and Rotamer Preferences of RNA-Binding Proteins Upon Binding RNA*.