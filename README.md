# RNA Binding Proteins (RBPs) Rotamer Libraries

## 📁 Directory Structure
<pre> RBPs_rotamer_libraries/
├── RBPs_BBD_rotamer_lib/
│   ├── RBPs_I_bbd_B_rotamer_lib.csv
│   ├── RBPs_I_bbd_U_rotamer_lib.csv
│   ├── RBPs_N_bbd_B_rotamer_lib.csv
│   ├── RBPs_N_bbd_U_rotamer_lib.csv
│   ├── RBPs_bbd_B_rotamer_lib.csv
│   └── RBPs_bbd_U_rotamer_lib.csv
└── RBPs_BBI_rotamer_lib/
    ├── RBPs_I_bbi_B_rotamer_lib.csv
    ├── RBPs_I_bbi_U_rotamer_lib.csv
    ├── RBPs_N_bbi_B_rotamer_lib.csv
    ├── RBPs_N_bbi_U_rotamer_lib.csv
    ├── RBPs_bbi_B_rotamer_lib.csv
    └── RBPs_bbi_U_rotamer_lib.csv </pre>



## 🌀 Backbone Dependent (BBD) Rotamer Library

### File Descriptions

| File Name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| `RBPs_I_bbd_B_rotamer_lib.csv`     | Rotamers at the RNA-binding interface (I) in bound (B) conformation         |
| `RBPs_I_bbd_U_rotamer_lib.csv`     | Rotamers at the RNA-binding interface (I) in unbound (U) conformation      |
| `RBPs_N_bbd_B_rotamer_lib.csv`     | Rotamers at the non RNA-binding site (N) in bound (B) conformation         |
| `RBPs_N_bbd_U_rotamer_lib.csv`     | Rotamers at the non RNA-binding site (N) in unbound (U) conformation       |
| `RBPs_bbd_B_rotamer_lib.csv`       | Overall in bound (B) conformation                                          |
| `RBPs_bbd_U_rotamer_lib.csv`       | Overall in unbound (U) conformation                                        |

### Column Descriptions

| Column          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `AA`            | Amino acid residue (3-letter code)                                          |
| `PHI`           | Backbone φ torsion angle (degrees)                                          |
| `PSI`           | Backbone ψ torsion angle (degrees)                                          |
| `_Count`        | Data points in specific (φ, ψ) bin                                          |
| `Count`         | Total data points in (φ, ψ) neighborhood                                    |
| `Prob`          | Probability of occurrence in bin (0.0-1.0)                                  |
| `B/U_CHI[1-4]`  | Bound/Unbound χₙ weighted circular mean (degrees)                           |
| `B/U_CHI[1-4]Sig` | Bound/Unbound χₙ weighted circular standard deviation (degrees)             |

## 📊 Backbone Independent (BBI) Rotamer Library

### File Descriptions

| File Name                          | Description                                                                 |
|------------------------------------|-----------------------------------------------------------------------------|
| `RBPs_I_bbi_B_rotamer_lib.csv`     | Interface (I) bound (B) residues - Backbone independent                    |
| `RBPs_I_bbi_U_rotamer_lib.csv`     | Interface (I) unbound (U) residues - Backbone independent                  |
| `RBPs_N_bbi_B_rotamer_lib.csv`     | Non-interface (N) bound (B) residues - Backbone independent                |
| `RBPs_N_bbi_U_rotamer_lib.csv`     | Non-interface (N) unbound (U) residues - Backbone independent              |
| `RBPs_bbi_B_rotamer_lib.csv`       | Overall bound (B) residues - Backbone independent                          |
| `RBPs_bbi_U_rotamer_lib.csv`       | Overall unbound (U) residues - Backbone independent                        |

### Column Descriptions

| Column          | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `AA`            | Amino acid residue (3-letter code)                                          |
| `Count`         | Total data points in bin                                                    |
| `Prob`          | Probability of occurrence (0.0-1.0)                                         |
| `B/U_CHI[1-4]`  | Bound/Unbound χₙ weighted circular mean (degrees)                           |
| `B/U_CHI[1-4]Sig` | Bound/Unbound χₙ weighted circular standard deviation (degrees)             |

## 💻 Usage Example


## 📝 Notes
1. All angular measurements are in degrees
2. Probability values range from 0.0 (never observed) to 1.0 (always observed)
3. Standard deviations calculated using circular statistics methods
4. Interface/non-interface classification based on [reference to criteria]

## 🔗 References
1. Kant, S., Nithin, C., Mukherjee, S., Maity, A., Bahadur, R.P., 2025. Protein‐ RNA Docking Benchmark v3.0 Integrated With Binding Affinity. Proteins prot.26825. https://doi.org/10.1002/prot.26825
   
2. Lovell, S.C., Word, J.M., Richardson, J.S., Richardson, D.C., 2000. The penultimate rotamer library. Proteins 40, 389–408. https://doi.org/10.1002/1097-0134(20000815)40:3<389::AID-PROT50>3.0.CO;2-2
3. Shapovalov, M.V., Dunbrack, R.L., 2011. A Smoothed Backbone-Dependent Rotamer Library for Proteins Derived from Adaptive Kernel Density Estimates and Regressions. Structure 19, 844–858. https://doi.org/10.1016/j.str.2011.03.019
   
4. Dunbrack, R.L., Karplus, M., 1993. Backbone-dependent Rotamer Library for Proteins Application to Side-chain Prediction. Journal of Molecular Biology 230, 543–574. https://doi.org/10.1006/jmbi.1993.1170
   
5. Zhang, O., Naik, S.A., Liu, Z.H., Forman-Kay, J., Head-Gordon, T., 2024. A curated rotamer library for common post-translational modifications of proteins. Bioinformatics 40. https://doi.org/10.1093/BIOINFORMATICS/BTAE444


6. Mardia, K.V., Zemroch, P.J., 1975. Algorithm AS 86: The Von Mises Distribution Function. Applied Statistics 24, 268. https://doi.org/10.2307/2346578

7. Mukherjee, S., Bahadur, R.P., 2018. An account of solvent accessibility in protein-RNA recognition. Sci Rep 8, 10546. https://doi.org/10.1038/s41598-018-28373-2

## 📄 License
[Please suggest the license type] @Sunandan Da
