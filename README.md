# RNA Binding Proteins (RBPs) Rotamer Libraries

A high-precision, circular-statistics-derived backbone-dependent (BBD) and backbone-independent (BBI) rotamer library suite for RNA-binding proteins in bound and unbound states across interface and surface locations.

---

## 📁 Directory Structure

```text
RBPs_rotamer_lib/
├── RBPs_BBD_rotamer_lib/         # Backbone-Dependent (30° x 30° Phi/Psi bins) Libraries
│   ├── RBPs_I_bbd_B_rotamer_lib.[csv|tsv|json]   # Bound Interface (I)
│   ├── RBPs_I_bbd_U_rotamer_lib.[csv|tsv|json]   # Unbound Interface (I)
│   ├── RBPs_N_bbd_B_rotamer_lib.[csv|tsv|json]   # Bound Non-Interface Surface (N)
│   ├── RBPs_N_bbd_U_rotamer_lib.[csv|tsv|json]   # Unbound Non-Interface Surface (N)
│   ├── RBPs_bbd_B_rotamer_lib.[csv|tsv|json]     # Bound Overall Surface (B)
│   └── RBPs_bbd_U_rotamer_lib.[csv|tsv|json]     # Unbound Overall Surface (U)
├── RBPs_BBI_rotamer_lib/         # Backbone-Independent (Global) Libraries
│   ├── RBPs_I_bbi_B_rotamer_lib.[csv|tsv|json]   # Bound Interface (I)
│   ├── RBPs_I_bbi_U_rotamer_lib.[csv|tsv|json]   # Unbound Interface (I)
│   ├── RBPs_N_bbi_B_rotamer_lib.[csv|tsv|json]   # Bound Non-Interface Surface (N)
│   ├── RBPs_N_bbi_U_rotamer_lib.[csv|tsv|json]   # Unbound Non-Interface Surface (N)
│   ├── RBPs_bbi_B_rotamer_lib.[csv|tsv|json]     # Bound Overall Surface (B)
│   └── RBPs_bbi_U_rotamer_lib.[csv|tsv|json]     # Unbound Overall Surface (U)
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
- Tags interface (`I`) vs non-interface (`N`) residues based on solvent accessible surface area (SASA) burial using PRince/NACCESS.

### 2. Depth-Valid Sample Masking & Symmetry Folding
- **Multi-Chi Depth Validation**: Residues with missing terminal atoms are validated up to their deepest non-null chi depth, ensuring partial sidechains contribute accurately without introducing `NaN` binning artifacts.
- **Symmetric Planar Ring Folding**: Symmetric terminal chi angles ($\text{ASP }\chi_2$, $\text{GLU }\chi_3$, $\text{PHE }\chi_2$, $\text{TYR }\chi_2$) are folded into $[ -90^\circ, 90^\circ )$ prior to rotamer binning to avoid splitting chemically identical conformations into separate states.

### 3. Von Mises Kernel-Weighted Circular Statistics
- **Grid Binning**: Backbone $(\Phi, \Psi)$ angles are binned into $30^\circ \times 30^\circ$ grid cells centered at $[-180^\circ, -150^\circ, \dots, +150^\circ]$.
- **Rotamer States**: Sidechain $\chi$ angles are binned into standard staggered states ($p / g^+$ at $+60^\circ$, $t$ at $180^\circ$, $m / g^-$ at $-60^\circ$).
- **Circular Statistics**: Weighted circular mean ($\mu = \text{angle}(\sum w_i e^{j\theta_i})$) and circular standard deviation ($\sigma$) are calculated using `scipy.stats.vonmises` fitting, eliminating artificial $-180^\circ / +180^\circ$ boundary discontinuities.

### 4. Unified Multi-Format Library Outputs
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
| `RBPs_bbd_B_rotamer_lib.[csv\|tsv\|json]` | Overall surface bound rotamers (Bound $B$) |
| `RBPs_bbd_U_rotamer_lib.[csv\|tsv\|json]` | Overall surface unbound rotamers (Unbound $U$) |

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
| `RBPs_bbi_B_rotamer_lib.[csv\|tsv\|json]` | Overall bound residues (Backbone Independent) |
| `RBPs_bbi_U_rotamer_lib.[csv\|tsv\|json]` | Overall unbound residues (Backbone Independent) |

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

## 💻 Usage Example

```python
import pandas as pd

# Load Backbone-Dependent Bound Interface Rotamer Library
df_bbd = pd.read_csv("RBPs_BBD_rotamer_lib/RBPs_I_bbd_B_rotamer_lib.csv")

# Filter rotamers for Arg at phi = -60, psi = -30
arg_rotamers = df_bbd[(df_bbd["AA"] == "ARG") & (df_bbd["PHI"] == -60) & (df_bbd["PSI"] == -30)]
print(arg_rotamers[["AA", "PHI", "PSI", "Prob", "B_CHI1", "B_CHI2", "B_CHI1Sig"]])
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