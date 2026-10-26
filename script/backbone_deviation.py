#!/usr/bin/env python3
"""
backbone_deviation.py

Does a backbone change between the unbound and bound protein go with a side-chain change, and is
that coupling specific to the RNA-binding interface?

Paired static structures cannot show that the backbone causes the side-chain change (both can
follow binding, crystal packing, resolution or local flexibility). The analysis therefore asks:
  1. Dose-response: chi1 / chi2 rotamer-switch rate, and within-well |delta chi| of residues that
     do not switch, by backbone change, for interface (I), non-interface surface (N) and core (C)
     residues. Surface and core residues are the internal baseline for how much side chains move
     when the backbone moves without binding.
  2. Adjusted model: logistic regression of a switch on log2(1 + delta_bb) x region, residue type,
     bound secondary structure, resolution of both structures and side-chain B-factor, with
     standard errors clustered by complex. The interaction gives whether backbone coupling is
     stronger at the interface; the region term gives the interface excess at fixed backbone change.
  3. 2x2 tables: backbone changed (delta_bb >= 10 / 20 / 40 degrees) x side chain switched.
  4. Backbone-dependent prediction: for residues whose rotamer switches and whose (phi, psi) moves
     to another 30-degree library bin, S = log P(rot_B | bb_B) - log P(rot_B | bb_U)
                                           + log P(rot_U | bb_U) - log P(rot_U | bb_B).
     S > 0 when each rotamer is more probable on its own backbone, i.e. the new rotamer is the one
     the backbone-dependent library expects on the new backbone. Swapping bound and unbound flips
     the sign of S, so with no backbone information S is symmetric around 0. Probabilities come
     from leave-one-complex-out counts (bound and unbound pooled, library binning), smoothed
     towards the backbone-independent distribution, so no complex is scored with its own counts.
     Null: the same library, with each residue's (bound, unbound) backbone pair taken from another
     scored residue of the same type.

Definitions:
  delta_bb       : distance on the (phi, psi) torus, sqrt(dphi^2 + dpsi^2), shortest-arc degrees.
  rotamer switch : the chi's library bin differs between bound and unbound (sp3: p / t / m;
                   terminal sp2: residue-specific bins, rotamer_definitions.py). Chis with a single
                   bin (ASP/PHE/TYR chi2) cannot switch and are left out.
  side-chain B   : mean B-factor of side-chain heavy atoms, as a z-score within each structure;
                   NMR structures have no B-factors.
  Excluded       : PRO (pucker, not a staggered rotamer), residues without phi/psi in either
                   state, and residues whose peptide bond flips cis/trans (|delta omega| > 90).

Main model: complexes whose bound and unbound structures both report a resolution (the NMR
unbound structures are left out). Sensitivity: (a) all complexes, without resolution and B-factor
terms; (b) main set without residues whose side-chain B z-score exceeds 2 in either structure.

Outputs:
  output_files/backbone_deviation/   residue table, dose-response, model and 2x2 tables, S scores
  figure(s)/backbone_deviation/      dose_response.png, adjusted_odds_ratios.png,
                                     backbone_dependent_prediction.png

Usage:
    python script/backbone_deviation.py [--rebuild-covariates] [--n-boot 1000]
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm

import rotamer_definitions as rotdef

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.paths import get_path

DATASET_DIR = ROOT_DIR / "input_files"
TORSION_DATASET = DATASET_DIR / "ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv"
COMPLEX_LIST = DATASET_DIR / "PRDBv3_pdbList_chains.json"
PRDB_INFO = ROOT_DIR / "PRDBv3_dataset" / "PRDBv3.0" / "PRDBv3_info.csv"
RESULTS_DIR = ROOT_DIR / "output_files" / "backbone_deviation"
FIGURES_DIRS = [ROOT_DIR / "figure" / "backbone_deviation", ROOT_DIR / "figures" / "backbone_deviation"]
COVARIATE_CACHE = RESULTS_DIR / "residue_side_chain_bfactors_187.tsv"

REGIONS = ["I", "N", "C"]
REGION_NAMES = {"I": "Interface", "N": "Non-interface surface", "C": "Core"}
# Validated categorical slots (dataviz palette); markers carry identity as well
REGION_COLORS = {"I": "#eb6834", "N": "#2a78d6", "C": "#1baf7a"}
REGION_MARKERS = {"I": "o", "N": "s", "C": "^"}
TEXT_COLOR, MUTED_COLOR, GRID_COLOR = "#1f1f1e", "#6b6a64", "#e4e3de"

DBB_EDGES = [0, 5, 10, 20, 40, 80, np.inf]
DBB_LABELS = ["0–5", "5–10", "10–20", "20–40", "40–80", ">80"]
CHANGE_THRESHOLDS = [10, 20, 40]
CHIS = [1, 2]
BB_BIN = 30            # library phi/psi bin width (backbone_dependent_rotamer_lib.py)
SMOOTHING = 10.0       # pseudo-counts towards the backbone-independent distribution
BZ_OUTLIER = 2.0
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": MUTED_COLOR, "axes.labelcolor": TEXT_COLOR, "xtick.color": MUTED_COLOR,
    "ytick.color": MUTED_COLOR, "text.color": TEXT_COLOR, "axes.spines.top": False,
    "axes.spines.right": False,
})


def angle_diff(a, b, period=360.0):
    """Unsigned shortest difference between angles (degrees) for the given period."""
    return np.abs((np.asarray(a, float) - np.asarray(b, float) + period / 2) % period - period / 2)


def library_bin(angle):
    """30-degree backbone grid point, as in BackboneDependentRotamerLib.get_closest_angle."""
    g = np.floor((np.asarray(angle, float) + BB_BIN / 2.0) / BB_BIN) * BB_BIN
    return np.where(g >= 180, g - 360, g).astype(int)


# ----------------------------------------------------------------------------------------------
# Covariates: resolution and side-chain B-factors
# ----------------------------------------------------------------------------------------------
def read_resolutions():
    """Bound and unbound resolution (A) per complex; NaN for NMR structures."""
    info = pd.read_csv(PRDB_INFO)
    return pd.DataFrame({
        "PDB": info["C_PDB"].str.strip(),
        "RES_B": pd.to_numeric(info["C_resolution"], errors="coerce"),
        "RES_U": pd.to_numeric(info["U_pro_resolution"], errors="coerce"),
    }).drop_duplicates("PDB")


def residue_bfactors(pdb_file):
    """
    Ordered residue labels with N, CA and C (the labels of ProteinBackboneCalculator) and the
    within-structure z-score of each residue's mean side-chain heavy-atom B-factor.
    """
    order, backbone, side_b = [], {}, {}
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            name, resname = line[12:16].strip(), line[17:20].strip()
            chain, seq, icode = line[21:22].strip(), line[22:26].strip(), line[26:27].strip()
            label = f"{resname:<3s} {chain:1s} {seq:>4s}{icode}".strip()
            if name in ("N", "CA", "C"):
                if label not in backbone:
                    order.append(label)
                backbone.setdefault(label, set()).add(name)
            elem = line[76:78].strip() or name[0]
            if name in BACKBONE_ATOMS or elem in ("H", "D"):
                continue
            try:
                side_b.setdefault(label, []).append(float(line[60:66]))
            except ValueError:
                pass
    labels = [lbl for lbl in order if backbone[lbl] == {"N", "CA", "C"}]
    mean_b = pd.Series({lbl: np.mean(side_b[lbl]) for lbl in labels if lbl in side_b}, dtype=float)
    sd = mean_b.std()
    z = (mean_b - mean_b.mean()) / sd if sd > 1e-6 else mean_b * np.nan
    return labels, z


def build_bfactor_table():
    """Side-chain B z-scores of every bound residue and its paired unbound residue."""
    rota_dir = get_path("rota_assign_dir")
    if str(rota_dir) not in sys.path:
        sys.path.insert(0, str(rota_dir))
    from macromol_torsion.pairing import pair_equal_residues

    structure_dir = get_path("prdbv3_rotamer_dir")
    rows = []
    for _, item in pd.read_json(COMPLEX_LIST).iterrows():
        bound = item["C_PDB"].strip()
        unbound = str(item.get("U_pro_PDB") or "").replace("*", "").strip()
        c_file = structure_dir / f"{bound}_{unbound}_C.pdb"
        u_file = structure_dir / f"{bound}_{unbound}_U.pdb"
        if not unbound or not c_file.exists() or not u_file.exists():
            continue
        b_labels, b_z = residue_bfactors(c_file)
        u_labels, u_z = residue_bfactors(u_file)
        try:
            pair_map, _ = pair_equal_residues(b_labels, u_labels)
        except ValueError:
            pair_map = {}
        for lbl in b_labels:
            u_lbl = pair_map.get(lbl)
            rows.append({"PDB": bound, "LABEL": lbl, "BZ_B": b_z.get(lbl, np.nan),
                         "BZ_U": u_z.get(u_lbl, np.nan) if u_lbl else np.nan})
    table = pd.DataFrame(rows).drop_duplicates(["PDB", "LABEL"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(COVARIATE_CACHE, sep="\t", index=False)
    logging.info(f"Side-chain B-factors of {len(table)} residues written to {COVARIATE_CACHE}")
    return table


# ----------------------------------------------------------------------------------------------
# Residue table
# ----------------------------------------------------------------------------------------------
def n_chi_bins(aa, chi_idx):
    """Number of library bins of chi `chi_idx` (0-based)."""
    sp2 = rotdef.terminal_sp2(aa, chi_idx)
    return 3 if sp2 is None else max(1, len(sp2[2]))


def build_residue_table(rebuild_covariates=False):
    df = pd.read_csv(TORSION_DATASET)
    df["AA"] = df["LABEL"].str[:3]
    n0 = len(df)
    df = df[df["AA"].isin(rotdef.RESIDUE_CHI_COUNTS) & (df["AA"] != "PRO")]
    df = df.dropna(subset=["B_PHI", "B_PSI", "U_PHI", "U_PSI", "B_CHI1", "U_CHI1"])
    omega_flip = angle_diff(df["B_OMEGA"], df["U_OMEGA"]) > 90
    df = df[~omega_flip].copy()
    logging.info(f"{len(df)} of {n0} residues kept ({int(omega_flip.sum())} cis/trans flips dropped)")

    df["DBB"] = np.hypot(angle_diff(df["B_PHI"], df["U_PHI"]), angle_diff(df["B_PSI"], df["U_PSI"]))
    df["DBB_BIN"] = pd.cut(df["DBB"], DBB_EDGES, labels=DBB_LABELS, right=False)
    df["LOG2_DBB"] = np.log2(1 + df["DBB"])
    df["REGION"] = df["SASA"]
    df["SS"] = df["B_SS"].fillna("L")
    df["BB_BIN_CHANGED"] = ((library_bin(df["B_PHI"]) != library_bin(df["U_PHI"]))
                            | (library_bin(df["B_PSI"]) != library_bin(df["U_PSI"])))

    for k in CHIS:
        switch = np.full(len(df), np.nan)
        dchi = np.full(len(df), np.nan)
        for aa, idx in df.groupby("AA").indices.items():
            if rotdef.RESIDUE_CHI_COUNTS[aa] < k or n_chi_bins(aa, k - 1) < 2:
                continue
            b, u = df[f"B_CHI{k}"].values[idx], df[f"U_CHI{k}"].values[idx]
            ok = np.isfinite(b) & np.isfinite(u)
            sb = np.array([rotdef.chi_class(aa, k - 1, a) for a in b[ok]])
            su = np.array([rotdef.chi_class(aa, k - 1, a) for a in u[ok]])
            sp2 = rotdef.terminal_sp2(aa, k - 1)
            switch[idx[ok]] = (sb != su).astype(float)
            dchi[idx[ok]] = angle_diff(b[ok], u[ok], sp2[0] if sp2 else 360.0)
        df[f"SWITCH_CHI{k}"] = switch
        df[f"DCHI{k}"] = dchi

    bfac = (pd.read_csv(COVARIATE_CACHE, sep="\t")
            if COVARIATE_CACHE.exists() and not rebuild_covariates else build_bfactor_table())
    df = df.merge(read_resolutions(), on="PDB", how="left").merge(bfac, on=["PDB", "LABEL"], how="left")
    df["BZ_MEAN"] = df[["BZ_B", "BZ_U"]].mean(axis=1, skipna=False)
    df["HAS_RESOLUTION"] = df["RES_B"].notna() & df["RES_U"].notna()
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 1. Dose-response with complex-level bootstrap
# ----------------------------------------------------------------------------------------------
def cluster_bootstrap_ratio(df, num_col, den_col, cell_cols, n_boot, rng):
    """
    sum(num) / sum(den) per cell, with a 95% interval from resampling complexes (Poisson weights).
    """
    pdbs = df["PDB"].astype("category")
    cells = df.groupby(cell_cols, observed=True).ngroup()
    n_pdb, n_cell = len(pdbs.cat.categories), cells.max() + 1
    num = np.zeros((n_pdb, n_cell))
    den = np.zeros((n_pdb, n_cell))
    np.add.at(num, (pdbs.cat.codes.values, cells.values), df[num_col].values)
    np.add.at(den, (pdbs.cat.codes.values, cells.values), df[den_col].values)
    w = rng.poisson(1.0, size=(n_boot, n_pdb))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (w @ num) / (w @ den)
    keys = df.groupby(cell_cols, observed=True).size().reset_index(name="N")
    keys["VALUE"] = num.sum(0) / den.sum(0)
    keys["CI_LOW"] = np.nanpercentile(boot, 2.5, axis=0)
    keys["CI_HIGH"] = np.nanpercentile(boot, 97.5, axis=0)
    return keys


def dose_response(df, n_boot, rng):
    tables = []
    for k in CHIS:
        sub = df.dropna(subset=[f"SWITCH_CHI{k}"]).assign(ONE=1.0)
        sw = cluster_bootstrap_ratio(sub, f"SWITCH_CHI{k}", "ONE", ["REGION", "DBB_BIN"], n_boot, rng)
        sw.insert(0, "MEASURE", "switch_rate")
        stay = sub[sub[f"SWITCH_CHI{k}"] == 0]
        wd = cluster_bootstrap_ratio(stay, f"DCHI{k}", "ONE", ["REGION", "DBB_BIN"], n_boot, rng)
        wd.insert(0, "MEASURE", "within_well_mean_abs_dchi")
        for t in (sw, wd):
            t.insert(0, "CHI", f"chi{k}")
        tables += [sw, wd]
    return pd.concat(tables, ignore_index=True)


# ----------------------------------------------------------------------------------------------
# 2. Adjusted logistic models, standard errors clustered by complex
# ----------------------------------------------------------------------------------------------
REG = "C(REGION, Treatment('N'))"
CONTRASTS = {
    "dbb_doubling_interface": {"X": 1, f"X:{REG}[T.I]": 1},
    "dbb_doubling_surface": {"X": 1},
    "dbb_doubling_core": {"X": 1, f"X:{REG}[T.C]": 1},
    "dbb_slope_ratio_interface_vs_surface": {f"X:{REG}[T.I]": 1},
    "interface_vs_surface_at_median_dbb": {f"{REG}[T.I]": 1},
    "core_vs_surface_at_median_dbb": {f"{REG}[T.C]": 1},
}
COVARIATE_CONTRASTS = {
    "resolution_bound_per_A": {"RES_B": 1},
    "resolution_unbound_per_A": {"RES_U": 1},
    "side_chain_B_z_per_sd": {"BZ_MEAN": 1},
}


def fit_model(data, k, adjusted):
    """
    Odds ratios of the CONTRASTS terms. X = log2(1 + delta_bb) centred on its median, so the region
    terms compare regions at the median backbone change and exp(X coefficient) is the odds ratio
    per doubling of (1 + delta_bb).
    """
    data = data.assign(X=data["LOG2_DBB"] - data["LOG2_DBB"].median())
    formula = (f"SWITCH_CHI{k} ~ X * {REG} + C(AA) + C(SS)"
               + (" + RES_B + RES_U + BZ_MEAN" if adjusted else ""))
    fit = smf.glm(formula, data, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data["PDB"])[0]})

    contrasts = dict(CONTRASTS, **(COVARIATE_CONTRASTS if adjusted else {}))
    cov = fit.cov_params()
    rows = []
    for name, terms in contrasts.items():
        vec = pd.Series(0.0, index=fit.params.index)
        for term, coef in terms.items():
            vec[term] = coef
        est = float(vec @ fit.params)
        se = float(np.sqrt(vec @ cov @ vec))
        rows.append({"TERM": name, "OR": np.exp(est), "CI_LOW": np.exp(est - 1.96 * se),
                     "CI_HIGH": np.exp(est + 1.96 * se), "P": 2 * norm.sf(abs(est / se))})
    out = pd.DataFrame(rows)
    out.insert(0, "N_COMPLEXES", data["PDB"].nunique())
    out.insert(0, "N_RESIDUES", len(data))
    return out


def adjusted_models(df):
    tables = []
    for k in CHIS:
        base = df.dropna(subset=[f"SWITCH_CHI{k}", "LOG2_DBB", "REGION", "AA", "SS"])
        main = base[base["HAS_RESOLUTION"]].dropna(subset=["BZ_MEAN"])
        clean = main[(main["BZ_B"].abs() <= BZ_OUTLIER) & (main["BZ_U"].abs() <= BZ_OUTLIER)]
        for label, data, adjusted in [("main", main, True), ("all_complexes_unadjusted", base, False),
                                      ("main_without_high_B", clean, True)]:
            t = fit_model(data, k, adjusted)
            t.insert(0, "MODEL", label)
            t.insert(0, "CHI", f"chi{k}")
            tables.append(t)
            logging.info(f"chi{k} {label}: {len(data)} residues, {data['PDB'].nunique()} complexes")
    return pd.concat(tables, ignore_index=True)


# ----------------------------------------------------------------------------------------------
# 3. Backbone changed x side chain switched
# ----------------------------------------------------------------------------------------------
def change_tables(df):
    rows = []
    for k in CHIS:
        sub = df.dropna(subset=[f"SWITCH_CHI{k}"])
        for thr in CHANGE_THRESHOLDS:
            for region in REGIONS:
                r = sub[sub["REGION"] == region]
                moved = r["DBB"] >= thr
                sw = r[f"SWITCH_CHI{k}"] == 1
                n_sw = int(sw.sum())
                rows.append({
                    "CHI": f"chi{k}", "THRESHOLD_DEG": thr, "REGION": region,
                    "BB_FIXED_STAYED": int((~moved & ~sw).sum()), "BB_FIXED_SWITCHED": int((~moved & sw).sum()),
                    "BB_MOVED_STAYED": int((moved & ~sw).sum()), "BB_MOVED_SWITCHED": int((moved & sw).sum()),
                    "SWITCH_RATE_BB_FIXED": sw[~moved].mean(), "SWITCH_RATE_BB_MOVED": sw[moved].mean(),
                    "FRACTION_OF_SWITCHES_WITH_BB_FIXED": (~moved & sw).sum() / n_sw if n_sw else np.nan,
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 4. Backbone-dependent prediction of the switch (leave-one-complex-out)
# ----------------------------------------------------------------------------------------------
def rotamer_keys(df, state):
    """Full rotamer (library bin of every chi) per residue; None if a chi is missing."""
    keys = pd.Series(None, index=df.index, dtype=object)
    for aa, sub in df.groupby("AA"):
        nchi = rotdef.RESIDUE_CHI_COUNTS[aa]
        chis = sub[[f"{state}_CHI{i + 1}" for i in range(nchi)]].to_numpy(float)
        for i, row in zip(sub.index, chis):
            if np.isfinite(row).all():
                keys[i] = "".join(str(rotdef.chi_class(aa, j, a)) for j, a in enumerate(row))
    return keys


def backbone_prediction(df, n_null=0, rng=None):
    """
    S of every residue whose full rotamer and backbone bin both change. With n_null > 0, also the
    mean S and fraction S > 0 when each residue's (bound, unbound) backbone pair is replaced by that
    of another such residue of the same type (library unchanged): the value expected if the
    backbone carried no information about which rotamer is adopted.
    """
    d = df.assign(ROT_B=rotamer_keys(df, "B"), ROT_U=rotamer_keys(df, "U")).dropna(subset=["ROT_B", "ROT_U"])
    d["BB_B"] = list(zip(library_bin(d["B_PHI"]), library_bin(d["B_PSI"])))
    d["BB_U"] = list(zip(library_bin(d["U_PHI"]), library_bin(d["U_PSI"])))

    # Observations: bound and unbound states pooled, as in the libraries
    cols = ["PDB", "AA", "BB", "ROT"]
    obs = pd.concat([d[["PDB", "AA", "BB_B", "ROT_B"]].set_axis(cols, axis=1),
                     d[["PDB", "AA", "BB_U", "ROT_U"]].set_axis(cols, axis=1)])
    c_bb_rot, c_bb = Counter(zip(obs.AA, obs.BB, obs.ROT)), Counter(zip(obs.AA, obs.BB))
    c_rot, c_aa = Counter(zip(obs.AA, obs.ROT)), Counter(obs.AA)
    p_bb_rot, p_bb = Counter(zip(obs.PDB, obs.AA, obs.BB, obs.ROT)), Counter(zip(obs.PDB, obs.AA, obs.BB))
    p_rot, p_aa = Counter(zip(obs.PDB, obs.AA, obs.ROT)), Counter(zip(obs.PDB, obs.AA))
    n_rot = obs.groupby("AA")["ROT"].nunique().to_dict()

    def log_prob(pdb, aa, bb, rot):
        """log P(rot | aa, bb) from the other complexes, smoothed towards P(rot | aa)."""
        bbi = (c_rot[(aa, rot)] - p_rot[(pdb, aa, rot)] + 1) / (c_aa[aa] - p_aa[(pdb, aa)] + n_rot[aa])
        n = c_bb[(aa, bb)] - p_bb[(pdb, aa, bb)]
        k = c_bb_rot[(aa, bb, rot)] - p_bb_rot[(pdb, aa, bb, rot)]
        return np.log((k + SMOOTHING * bbi) / (n + SMOOTHING))

    def scores(frame, bb_b, bb_u):
        return np.array([log_prob(p, aa, b, rb) - log_prob(p, aa, u, rb) + log_prob(p, aa, u, ru) - log_prob(p, aa, b, ru)
                         for p, aa, rb, ru, b, u in zip(frame.PDB, frame.AA, frame.ROT_B, frame.ROT_U, bb_b, bb_u)])

    d = d[(d["ROT_B"] != d["ROT_U"]) & (d["BB_B"] != d["BB_U"])].copy()
    d["S"] = scores(d, d["BB_B"], d["BB_U"])

    null = []
    for _ in range(n_null):
        perm = np.arange(len(d))
        for idx in d.groupby("AA").indices.values():
            perm[idx] = rng.permutation(idx)
        s = scores(d, d["BB_B"].values[perm], d["BB_U"].values[perm])
        null.append({"FRACTION_S_POSITIVE": (s > 0).mean(), "MEAN_S": s.mean()})
    return d[["PDB", "LABEL", "AA", "REGION", "DBB", "ROT_B", "ROT_U", "S"]], pd.DataFrame(null)


def summarise_prediction(scored, n_boot, rng):
    scored = scored.assign(AGREE=(scored["S"] > 0).astype(float), ONE=1.0,
                           DBB_CLASS=np.where(scored["DBB"] >= 20, ">=20", "<20"))
    tables = []
    for cell_cols in (["REGION"], ["REGION", "DBB_CLASS"]):
        frac = cluster_bootstrap_ratio(scored, "AGREE", "ONE", cell_cols, n_boot, rng).rename(
            columns={"VALUE": "FRACTION_S_POSITIVE", "CI_LOW": "FRAC_CI_LOW", "CI_HIGH": "FRAC_CI_HIGH"})
        mean = cluster_bootstrap_ratio(scored, "S", "ONE", cell_cols, n_boot, rng)
        frac["MEAN_S"], frac["MEAN_S_CI_LOW"], frac["MEAN_S_CI_HIGH"] = mean["VALUE"], mean["CI_LOW"], mean["CI_HIGH"]
        if "DBB_CLASS" not in frac:
            frac["DBB_CLASS"] = "all"
        tables.append(frac)
    return pd.concat(tables, ignore_index=True)[
        ["REGION", "DBB_CLASS", "N", "FRACTION_S_POSITIVE", "FRAC_CI_LOW", "FRAC_CI_HIGH",
         "MEAN_S", "MEAN_S_CI_LOW", "MEAN_S_CI_HIGH"]]


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
def style_axis(ax):
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def region_legend(fig, y):
    handles = [plt.Line2D([], [], color=REGION_COLORS[r], marker=REGION_MARKERS[r], linewidth=2,
                          markersize=7, label=REGION_NAMES[r]) for r in REGIONS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=3, frameon=False, fontsize=10)


def save(fig, name):
    for out_dir in FIGURES_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, dpi=600, bbox_inches="tight")
    logging.info(f"Saved {name}")
    plt.close(fig)


def plot_dose_response(table):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True)
    x = np.arange(len(DBB_LABELS))
    offsets = {"I": -0.12, "N": 0.0, "C": 0.12}
    rows = [("switch_rate", "Rotamer switch rate (%)", 100.0),
            ("within_well_mean_abs_dchi", "Within-rotamer |Δχ| (°)\n(residues that do not switch)", 1.0)]
    for col, k in enumerate(CHIS):
        for row, (measure, ylabel, scale) in enumerate(rows):
            ax = axes[row, col]
            sub = table[(table["CHI"] == f"chi{k}") & (table["MEASURE"] == measure)]
            for region in REGIONS:
                r = sub[sub["REGION"] == region].set_index("DBB_BIN").reindex(DBB_LABELS)
                ax.errorbar(x + offsets[region], r["VALUE"] * scale,
                            yerr=[(r["VALUE"] - r["CI_LOW"]) * scale, (r["CI_HIGH"] - r["VALUE"]) * scale],
                            color=REGION_COLORS[region], marker=REGION_MARKERS[region], markersize=6,
                            linewidth=2, capsize=0, elinewidth=1.2, markeredgecolor="white", markeredgewidth=1)
            style_axis(ax)
            ax.set_ylim(bottom=0)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == 0:
                ax.set_title(rf"$\chi_{k}$", fontsize=12, pad=8)
            else:
                ax.set_xticks(x, DBB_LABELS)
                ax.set_xlabel("Backbone change Δφψ (°)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    region_legend(fig, y=0.965)
    save(fig, "dose_response.png")


def plot_odds_ratios(models):
    terms = [("dbb_doubling_interface", "Δφψ doubling: interface"),
             ("dbb_doubling_surface", "Δφψ doubling: non-interface surface"),
             ("dbb_doubling_core", "Δφψ doubling: core"),
             ("dbb_slope_ratio_interface_vs_surface", "Interface / surface Δφψ slope"),
             ("interface_vs_surface_at_median_dbb", "Interface vs surface, same Δφψ"),
             ("core_vs_surface_at_median_dbb", "Core vs surface, same Δφψ")]
    model_style = {"main": ("Main (X-ray/EM, adjusted)", "o", TEXT_COLOR),
                   "all_complexes_unadjusted": ("All complexes, unadjusted", "s", "#2a78d6"),
                   "main_without_high_B": ("Main, without high-B residues", "D", "#eb6834")}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    y = np.arange(len(terms))[::-1]
    for ax, k in zip(axes, CHIS):
        for m_idx, (model, (label, marker, color)) in enumerate(model_style.items()):
            sub = models[(models["CHI"] == f"chi{k}") & (models["MODEL"] == model)].set_index("TERM")
            sub = sub.reindex([t for t, _ in terms])
            ax.errorbar(sub["OR"], y + (1 - m_idx) * 0.22, xerr=[sub["OR"] - sub["CI_LOW"], sub["CI_HIGH"] - sub["OR"]],
                        fmt=marker, color=color, markersize=6, elinewidth=1.2, capsize=0,
                        markeredgecolor="white", markeredgewidth=0.8, label=label)
        ax.axvline(1, color=MUTED_COLOR, linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3], ["0.5", "0.75", "1", "1.5", "2", "3"])
        ax.minorticks_off()
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.set_title(rf"$\chi_{k}$ rotamer switch", fontsize=11, pad=8)
        ax.set_xlabel("Odds ratio (95% CI, clustered by complex)", fontsize=10)
    axes[0].set_yticks(y, [t for _, t in terms], fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.55, 0.93), ncol=3, frameon=False, fontsize=9)
    save(fig, "adjusted_odds_ratios.png")


def plot_prediction(summary, scored, null):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0), gridspec_kw={"width_ratios": [1, 1.25]})
    classes = [("all", "All"), ("<20", "Δφψ < 20°"), (">=20", "Δφψ ≥ 20°")]
    x = np.arange(len(classes))
    offsets = {"I": -0.2, "N": 0.0, "C": 0.2}
    for region in REGIONS:
        r = summary[summary["REGION"] == region].set_index("DBB_CLASS").reindex([c for c, _ in classes])
        ax1.errorbar(x + offsets[region], r["FRACTION_S_POSITIVE"] * 100,
                     yerr=[(r["FRACTION_S_POSITIVE"] - r["FRAC_CI_LOW"]) * 100,
                           (r["FRAC_CI_HIGH"] - r["FRACTION_S_POSITIVE"]) * 100],
                     fmt=REGION_MARKERS[region], color=REGION_COLORS[region], markersize=7, elinewidth=1.2,
                     capsize=0, markeredgecolor="white", markeredgewidth=1)
    null_pct = null["FRACTION_S_POSITIVE"].mean() * 100 if len(null) else 50.0
    ax1.axhline(null_pct, color=MUTED_COLOR, linewidth=1, linestyle="--")
    ax1.text(-0.45, null_pct + 0.4, "shuffled backbones", fontsize=8, color=MUTED_COLOR, ha="left", va="bottom")
    ax1.set_xticks(x, [c for _, c in classes])
    ax1.set_xlim(-0.5, len(classes) - 0.5)
    ax1.set_ylabel("Switches towards the rotamer the\nlibrary favours on the new backbone (%)", fontsize=10)
    style_axis(ax1)

    bins = np.linspace(-6, 6, 49)
    for region in REGIONS:
        s = scored.loc[scored["REGION"] == region, "S"].clip(-6, 6)
        ax2.hist(s, bins=bins, density=True, histtype="step", linewidth=2, color=REGION_COLORS[region])
    ax2.axvline(0, color=MUTED_COLOR, linewidth=1, linestyle="--")
    ax2.set_xlabel("S: log-probability gain of each rotamer on its own backbone", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    style_axis(ax2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    region_legend(fig, y=0.93)
    save(fig, "backbone_dependent_prediction.png")


# ----------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Backbone change vs side-chain rotamer change, bound vs unbound.")
    parser.add_argument("--rebuild-covariates", action="store_true",
                        help="Re-read side-chain B-factors from the structure files instead of the cached table.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Complex-level bootstrap replicates.")
    parser.add_argument("--n-null", type=int, default=200, help="Shuffled-backbone replicates for the S null.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the bootstrap and null.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = build_residue_table(args.rebuild_covariates)
    keep = ["PDB", "LABEL", "AA", "REGION", "SS", "DBB", "BB_BIN_CHANGED", "SWITCH_CHI1", "DCHI1",
            "SWITCH_CHI2", "DCHI2", "RES_B", "RES_U", "BZ_B", "BZ_U"]
    df[keep].round(3).to_csv(RESULTS_DIR / "residue_backbone_sidechain_changes_187.tsv", sep="\t", index=False)

    dose = dose_response(df, args.n_boot, rng)
    dose.round(4).to_csv(RESULTS_DIR / "dose_response.tsv", sep="\t", index=False)

    models = adjusted_models(df)
    models.to_csv(RESULTS_DIR / "adjusted_odds_ratios.tsv", sep="\t", index=False, float_format="%.4g")

    change_tables(df).round(4).to_csv(RESULTS_DIR / "backbone_change_x_switch.tsv", sep="\t", index=False)

    scored, null = backbone_prediction(df, args.n_null, rng)
    scored.round(4).to_csv(RESULTS_DIR / "backbone_dependent_prediction_scores.tsv", sep="\t", index=False)
    null.round(4).to_csv(RESULTS_DIR / "backbone_dependent_prediction_null.tsv", sep="\t", index=False)
    logging.info(f"Shuffled-backbone null: S > 0 in {null['FRACTION_S_POSITIVE'].mean():.3f} "
                 f"(95% {null['FRACTION_S_POSITIVE'].quantile(0.025):.3f}-{null['FRACTION_S_POSITIVE'].quantile(0.975):.3f}), "
                 f"mean S {null['MEAN_S'].mean():.3f}; observed {(scored['S'] > 0).mean():.3f}, {scored['S'].mean():.3f}")
    pred = summarise_prediction(scored, args.n_boot, rng)
    pred.round(4).to_csv(RESULTS_DIR / "backbone_dependent_prediction_summary.tsv", sep="\t", index=False)

    plot_dose_response(dose)
    plot_odds_ratios(models)
    plot_prediction(pred, scored, null)
    logging.info(f"Tables in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
