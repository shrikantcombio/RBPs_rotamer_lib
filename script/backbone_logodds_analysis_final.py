#!/usr/bin/env python3
"""
backbone_logodds_analysis_final.py
===================================

Backbone-Conditioned Rotamer Log-Odds Analysis for Protein-RNA Complexes
-------------------------------------------------------------------------

For each residue with a bound/unbound pair, computes:

    log_odds = log2[ P(rotamer_B | phi_B, psi_B)
                   / P(rotamer_B | phi_U, psi_U) ]

using the RBP backbone-dependent (BBD) rotamer library.

Interpretation
--------------
  log_odds > 0  : Passive — backbone movement drove the rotamer change.
                  The bound rotamer is MORE probable at the new backbone
                  position than at the old one.

  log_odds < 0  : Active — RNA contact induced rotamer strain.
                  The bound rotamer is LESS probable at the new backbone
                  position than at the old one; the RNA contact overrides
                  backbone geometry.

  log_odds = 0  : No net backbone-mediated effect, or rotamer not in library
                  at either position (both-pseudocount case; excluded from
                  primary analysis).

Outputs
-------
  per_residue_logodds.csv       — one row per residue, all computed fields
  per_aa_stats_final.csv        — per-amino-acid Wilcoxon + BH correction
  statistical_summary_final.csv — global, PDB-level, Fisher, BB-shift tests
  fig1_logodds_violin.png       — I vs N violin/box with passive/active labels
  fig2_per_aa_heatmap.png       — per-AA mean log-odds heatmap (I vs N)
  fig3_passive_active_mechanism.png — passive/active mechanism classification
  fig4_logodds_vs_bbrms.png     — log-odds vs backbone RMS scatter

Usage
-----
  python script/backbone_logodds_analysis_final.py \\
      --torsion  input_files/ALL_AA_BACKBONE_OMEGA_SIDECHAIN_TORSION_ANGLES_187_clean.tsv \\
      --bbd-dir  ../rota_assign/rota_lib_dict/v2/bbd \\
      --output-dir output_files/logodds_v2 \\
      --bb-sep 30

  --bbd-dir is a rota_assign BBD cache: rota_lib_dict/v2/bbd (v2 libraries) or
  rota_lib_dict/bbd (v1). It must contain these four files:
      full_rota_lib_bbd_b_I.json
      full_rota_lib_bbd_b_N.json
      full_rota_lib_bbd_ub_I.json
      full_rota_lib_bbd_ub_N.json

Dependencies
------------
  numpy, pandas, scipy, matplotlib, statsmodels

Notes
-----
  - Residues whose rotamer is absent from the library at BOTH backbone cells
    (rotamer outliers; both probabilities at pseudocount = 0.01, log-odds = 0)
    are reported in the summary but excluded from statistical tests.
  - The BBD library was built from the same 187 structures used here,
    so absolute probability values are not fully independent. The log-odds
    ratio (bound vs unbound backbone cell) is internally consistent.
  - The rotamer key is the library bin (rotamer_definitions.chi_class: p/t/m
    for sp3 chis) of chi1..chi(n-1); the terminal chi is not part of the key.
    Library rows are grouped by the bins of their mean chis, and
    P(rotamer | cell) sums the rows sharing the key. The residue and the library
    are classified with the same bins: the 6-state labels used before (edges at
    +-30, +-90, +-150) split PRO's exo pucker (chi1 ~ -27) between 'T' and 'm'
    while every PRO library row was 'T', so such residues fell back to a
    neighbouring cell.
  - Residues without a defined bound and unbound phi/psi (NaN: chain termini,
    chain breaks, residues without an unbound partner) are skipped.
"""

import argparse
import json
import logging
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rotamer_definitions as rotdef

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def benjamini_hochberg(p_values):
    """Computes Benjamini-Hochberg False Discovery Rate (FDR) adjusted p-values."""
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([])
    valid_mask = ~np.isnan(p_vals)
    adj_p = np.full(n, np.nan)
    if not np.any(valid_mask):
        return adj_p

    valid_p = p_vals[valid_mask]
    n_valid = len(valid_p)
    s_idx = np.argsort(valid_p)
    s_p = valid_p[s_idx]

    adj_valid = np.zeros(n_valid)
    c_min = 1.0
    for i in range(n_valid - 1, -1, -1):
        rank = i + 1
        a = (s_p[i] * n_valid) / rank
        c_min = min(c_min, a)
        adj_valid[s_idx[i]] = min(1.0, c_min)

    adj_p[valid_mask] = adj_valid
    return adj_p

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Number of rotatable chi angles per residue type
AA_NCHI = {
    "ARG": 4, "LYS": 4, "GLU": 3, "GLN": 3, "MET": 3,
    "ASP": 2, "ASN": 2, "HIS": 2, "LEU": 2, "ILE": 2,
    "PHE": 2, "PRO": 2, "TRP": 2, "TYR": 2,
    "CYS": 1, "SER": 1, "THR": 1, "VAL": 1,
}

# Display order for per-AA plots (ascending mean log-odds at interface,
# from the 187-structure dataset)
AA_DISPLAY_ORDER = [
    "TRP", "CYS", "PHE", "VAL", "THR", "TYR",
    "SER", "LEU", "ILE", "ASP", "PRO", "ARG",
    "MET", "ASN", "GLU", "LYS", "GLN", "HIS",
]

# Pseudocount floor for library probabilities (prevents log(0))
PSEUDOCOUNT = 0.01

plt.style.use("tableau-colorblind10")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Utility functions
# ─────────────────────────────────────────────────────────────────────────────

_CLASS_LETTERS = {0: "p", 1: "t", 2: "m"}  # rotdef.rotameric_class (3 splits) -> letter


def _n_key_chis(aa: str) -> int:
    nchi = AA_NCHI.get(aa, 0)
    return nchi - 1 if nchi > 1 else nchi


def get_rotamer_key(aa: str, chi_vals: list) -> str:
    """Rotamer key of chi1..chi(n-1) (chi1 for single-chi residues) in the library bins."""
    n_key = _n_key_chis(aa)
    if n_key == 0 or len(chi_vals) < n_key:
        return ""
    return "".join(_CLASS_LETTERS[rotdef.chi_class(aa, i, float(chi_vals[i]))] for i in range(n_key))


def regroup_library(bbd: dict) -> dict:
    """
    {aa: {cell: {rotamer_key: summed probability}}} from a rota_assign BBD cache, whose rows
    ([chi1, ..., chiN, prob]) are keyed by 6-state letters; rows are re-keyed by the library
    bins of their mean chis (get_rotamer_key).
    """
    out = {}
    for aa, cells in bbd.items():
        if aa not in AA_NCHI:
            continue
        for cell, groups in cells.items():
            acc = out.setdefault(aa, {}).setdefault(cell, {})
            for rows in groups.values():
                for row in rows:
                    key = get_rotamer_key(aa, row[:-1])
                    acc[key] = acc.get(key, 0.0) + float(row[-1])
    return out


def snap_to_grid(angle: float, bb_sep: int = 30) -> int:
    """Backbone grid centre of an angle, as in the library build
    (BackboneDependentRotamerLibrary.get_closest_angle): bins
    [g - bb_sep/2, g + bb_sep/2), the bin centred on 180 labelled -180."""
    g = int(np.floor((angle + bb_sep / 2.0) / bb_sep)) * bb_sep
    return g - 360 if g >= 180 else g


def circular_diff(b: float, u: float) -> float:
    """True circular difference b - u in (-180, 180]."""
    d = np.radians(b) - np.radians(u)
    return float(np.degrees(np.arctan2(np.sin(d), np.cos(d))))


def get_prob_from_bbd(bbd: dict, aa: str, rotamer_key: str,
                      phi: float, psi: float,
                      bb_sep: int = 30,
                      max_radius: int = 2) -> float:
    """Look up P(rotamer_key | phi, psi) from a BBD library dict.

    Searches an expanding Chebyshev shell of backbone grid cells up to
    max_radius steps away from the snapped (phi, psi) centre.
    Returns PSEUDOCOUNT if the rotamer is not found in any nearby cell.

    Parameters
    ----------
    bbd          : loaded BBD JSON dict (one of the four condition files)
    aa           : three-letter amino acid code
    rotamer_key  : output of get_rotamer_key()
    phi, psi     : backbone angles in degrees
    bb_sep       : grid resolution in degrees (must match library build)
    max_radius   : maximum Chebyshev distance in grid steps to search

    Returns
    -------
    float : probability ∈ (0, 1], floored at PSEUDOCOUNT
    """
    if aa not in bbd:
        return PSEUDOCOUNT

    aa_lib = bbd[aa]

    for radius in range(max_radius + 1):
        phi_snap = snap_to_grid(phi, bb_sep)
        psi_snap = snap_to_grid(psi, bb_sep)

        for dp in range(-radius, radius + 1):
            for dq in range(-radius, radius + 1):
                if max(abs(dp), abs(dq)) != radius:
                    continue  # only the outer shell at this radius

                test_phi = int(((phi_snap + dp * bb_sep + 180) % 360) - 180)
                test_psi = int(((psi_snap + dq * bb_sep + 180) % 360) - 180)
                conf_key = f"{test_phi},{test_psi}"

                if conf_key in aa_lib:
                    cell = aa_lib[conf_key]
                    if rotamer_key in cell:
                        # regroup_library: summed probability of the key's rows
                        return max(cell[rotamer_key], PSEUDOCOUNT)

    return PSEUDOCOUNT


# ─────────────────────────────────────────────────────────────────────────────
# 2. Library loader
# ─────────────────────────────────────────────────────────────────────────────

def load_bbd_libraries(bbd_dir: Path) -> dict:
    """Load all four BBD JSON files with flexible pattern matching and fallbacks.

    Returns
    -------
    dict : libs[state][sasa] where state ∈ {'b','ub'}, sasa ∈ {'I','N'}
    """
    fallback_dir = Path(__file__).resolve().parent.parent.parent / "rota_assign" / "rota_lib_dict" / "v2" / "bbd"

    candidates = {
        ("b",  "I"): ["full_rota_lib_bbd_b_I.json", "RBPs_I_bbd_B_rotamer_lib.json", "RBPs_I_bbd_b_rotamer_lib.json"],
        ("b",  "N"): ["full_rota_lib_bbd_b_N.json", "RBPs_N_bbd_B_rotamer_lib.json", "RBPs_N_bbd_b_rotamer_lib.json"],
        ("ub", "I"): ["full_rota_lib_bbd_ub_I.json", "RBPs_I_bbd_U_rotamer_lib.json", "RBPs_I_bbd_ub_rotamer_lib.json"],
        ("ub", "N"): ["full_rota_lib_bbd_ub_N.json", "RBPs_N_bbd_U_rotamer_lib.json", "RBPs_N_bbd_ub_rotamer_lib.json"],
    }
    libs = {}
    for (state, sasa), c_list in candidates.items():
        found_file = None
        for fname in c_list:
            fpath = bbd_dir / fname
            if fpath.exists():
                found_file = fpath
                break
            fpath_fall = fallback_dir / fname
            if fpath_fall.exists():
                found_file = fpath_fall
                break

        if found_file is not None:
            with open(found_file) as f:
                libs.setdefault(state, {})[sasa] = regroup_library(json.load(f))
            logging.info(f"Loaded {found_file}")
        else:
            logging.warning(f"BBD file not found for state={state}, sasa={sasa} in {bbd_dir}")
            libs.setdefault(state, {})[sasa] = {}
    return libs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_logodds(df: pd.DataFrame, libs: dict, bb_sep: int = 30) -> pd.DataFrame:
    """Compute per-residue backbone-conditioned log-odds.

    For each residue:
      1. Identify the bound rotamer key from B_CHI1..N.
      2. Look up P(rotamer | phi_B, psi_B) from the bound BBD library.
      3. Look up P(rotamer | phi_U, psi_U) from the unbound BBD library.
      4. log_odds = log2(p_bound_bb / p_unbound_bb).

    Filters applied
    ---------------
    - Residue type must be in AA_NCHI.
    - SASA must be 'I' or 'N'.
    - B_PHI, B_PSI, U_PHI, U_PSI must be defined (not NaN).
    - At least chi1 must be non-NaN in the bound state.

    Returns
    -------
    pd.DataFrame with one row per valid residue and columns:
        PDB, LABEL, AA, SASA, B_PHI, B_PSI, U_PHI, U_PSI, B_SS, U_SS,
        rotamer_key, p_at_bound_bb, p_at_unbound_bb, log_odds,
        backbone_shifted, chi1_shifted, ss_change, bb_rms_deg
    """
    required = ["PDB", "LABEL", "SASA",
                "B_PHI", "B_PSI", "U_PHI", "U_PSI",
                "B_SS", "U_SS", "B_CHI1", "U_CHI1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in torsion file: {missing}")

    records = []

    for _, row in df.iterrows():
        aa = str(row["LABEL"])[:3].strip()
        if aa not in AA_NCHI:
            continue

        sasa = str(row["SASA"]).strip()
        if sasa not in ("I", "N"):
            continue

        phi_b = row["B_PHI"]
        psi_b = row["B_PSI"]
        phi_u = row["U_PHI"]
        psi_u = row["U_PSI"]

        # Backbone must be defined in both states
        if any(pd.isna(v) for v in [phi_b, psi_b, phi_u, psi_u]):
            continue

        phi_b, psi_b = float(phi_b), float(psi_b)
        phi_u, psi_u = float(phi_u), float(psi_u)

        # Collect bound chi angles (stop at first NaN — depth rule)
        nchi = AA_NCHI[aa]
        b_chis = []
        for i in range(1, nchi + 1):
            bc = row.get(f"B_CHI{i}", np.nan)
            if pd.isna(bc):
                break
            b_chis.append(float(bc))

        # The rotamer key needs chi1..chi(n-1); truncated side chains are skipped rather
        # than counted as library outliers
        if len(b_chis) < max(1, nchi - 1):
            continue

        # Collect unbound chi1 (for transition detection)
        u_chi1_raw = row.get("U_CHI1", np.nan)
        u_chi1 = float(u_chi1_raw) if not pd.isna(u_chi1_raw) else None

        # Build rotamer key from bound chis
        rot_key = get_rotamer_key(aa, b_chis)

        # BBD library lookups (SASA-stratified)
        bbd_b  = libs.get("b",  {}).get(sasa, {})
        bbd_ub = libs.get("ub", {}).get(sasa, {})

        p_bound_bb   = get_prob_from_bbd(bbd_b,  aa, rot_key, phi_b, psi_b, bb_sep)
        p_unbound_bb = get_prob_from_bbd(bbd_ub, aa, rot_key, phi_u, psi_u, bb_sep)

        log_odds = float(np.log2(p_bound_bb / p_unbound_bb))

        # Backbone grid shift flag
        phi_b_g = snap_to_grid(phi_b, bb_sep)
        phi_u_g = snap_to_grid(phi_u, bb_sep)
        psi_b_g = snap_to_grid(psi_b, bb_sep)
        psi_u_g = snap_to_grid(psi_u, bb_sep)
        backbone_shifted = int((phi_b_g != phi_u_g) or (psi_b_g != psi_u_g))

        # Chi1 rotamer state change (library bins)
        b_chi1_state = rotdef.chi_class(aa, 0, b_chis[0])
        u_chi1_state = rotdef.chi_class(aa, 0, u_chi1) if u_chi1 is not None else None
        chi1_shifted = int(b_chi1_state != u_chi1_state) if u_chi1_state is not None else 0

        # Secondary structure change
        b_ss = str(row.get("B_SS", "?")).strip()
        u_ss = str(row.get("U_SS", "?")).strip()
        ss_change = int(b_ss != u_ss)

        # RMS backbone deviation (circular)
        dphi = circular_diff(phi_b, phi_u)
        dpsi = circular_diff(psi_b, psi_u)
        bb_rms = float(np.sqrt((dphi**2 + dpsi**2) / 2.0))

        records.append({
            "PDB":              row["PDB"],
            "LABEL":            row["LABEL"],
            "AA":               aa,
            "SASA":             sasa,
            "B_PHI":            round(phi_b, 2),
            "B_PSI":            round(psi_b, 2),
            "U_PHI":            round(phi_u, 2),
            "U_PSI":            round(psi_u, 2),
            "B_SS":             b_ss,
            "U_SS":             u_ss,
            "rotamer_key":      rot_key,
            "p_at_bound_bb":    round(p_bound_bb,   4),
            "p_at_unbound_bb":  round(p_unbound_bb, 4),
            "log_odds":         round(log_odds, 4),
            "backbone_shifted": backbone_shifted,
            "chi1_shifted":     chi1_shifted,
            "ss_change":        ss_change,
            "bb_rms_deg":       round(bb_rms, 2),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Filtering: exclude both-pseudocount rows
# ─────────────────────────────────────────────────────────────────────────────

def filter_valid(df_lo: pd.DataFrame) -> tuple:
    """Split into valid (at least one prob > pseudocount) and excluded rows.

    Rows where both p_at_bound_bb and p_at_unbound_bb equal PSEUDOCOUNT
    represent rotamer outliers absent from the library at both backbone
    positions. Log-odds = 0 for these is uninformative and they are excluded
    from all statistical tests. Their count is reported separately.
    """
    valid_mask = (df_lo["p_at_bound_bb"] > PSEUDOCOUNT) | \
                 (df_lo["p_at_unbound_bb"] > PSEUDOCOUNT)
    return df_lo[valid_mask].copy(), df_lo[~valid_mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def run_statistics(df_valid: pd.DataFrame) -> dict:
    """Run all statistical tests on valid (non-pseudocount) log-odds rows.

    Tests
    -----
    1. Global Mann-Whitney U: Interface vs Non-interface log-odds (residue-level)
    2. PDB-level Mann-Whitney U: mean per-PDB log-odds I vs N
    3. Per-amino-acid Wilcoxon signed-rank (log-odds vs 0) + BH correction
    4. Passive vs Active mechanism: Fisher's exact on χ1-shifted residues I vs N
    5. SS-change × negative log-odds: Fisher's exact per SASA
    6. Backbone-shifted vs stable: Mann-Whitney U per SASA
    """
    results = {}

    lo_I = df_valid[df_valid["SASA"] == "I"]["log_odds"].dropna()
    lo_N = df_valid[df_valid["SASA"] == "N"]["log_odds"].dropna()

    # ── 5.1 Global residue-level ──────────────────────────────────────────────
    stat, p = stats.mannwhitneyu(lo_I, lo_N, alternative="two-sided")
    results["global"] = {
        "n_I": len(lo_I), "n_N": len(lo_N),
        "median_I": round(float(lo_I.median()), 4),
        "median_N": round(float(lo_N.median()), 4),
        "mean_I":   round(float(lo_I.mean()),   4),
        "mean_N":   round(float(lo_N.mean()),   4),
        "MWU_stat": round(float(stat), 2),
        "p_residue_level": float(p),
    }

    # ── 5.2 PDB-level (pseudoreplication control) ─────────────────────────────
    pdb_I = df_valid[df_valid["SASA"] == "I"].groupby("PDB")["log_odds"].mean()
    pdb_N = df_valid[df_valid["SASA"] == "N"].groupby("PDB")["log_odds"].mean()
    stat2, p2 = stats.mannwhitneyu(pdb_I, pdb_N, alternative="two-sided")
    results["pdb_level"] = {
        "n_pdbs_I": len(pdb_I), "n_pdbs_N": len(pdb_N),
        "median_I": round(float(pdb_I.median()), 4),
        "median_N": round(float(pdb_N.median()), 4),
        "MWU_stat": round(float(stat2), 2),
        "p_pdb_level": float(p2),
    }

    # ── 5.3 Per-amino-acid Wilcoxon + BH ─────────────────────────────────────
    aa_rows = []
    for sasa in ("I", "N"):
        sub = df_valid[df_valid["SASA"] == sasa]
        for aa in sorted(AA_NCHI.keys()):
            s = sub[sub["AA"] == aa]["log_odds"].dropna()
            if len(s) >= 5:
                try:
                    stat_w, p_w = stats.wilcoxon(
                        s, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    stat_w, p_w = np.nan, np.nan
                aa_rows.append({
                    "AA": aa, "SASA": sasa, "n": len(s),
                    "median_lo": round(float(s.median()), 4),
                    "mean_lo":   round(float(s.mean()),   4),
                    "frac_passive": round(float((s > 0).mean()), 4),
                    "frac_active":  round(float((s < 0).mean()), 4),
                    "wilcoxon_stat": round(float(stat_w), 2) if not np.isnan(stat_w) else np.nan,
                    "p_raw": float(p_w),
                })

    aa_df = pd.DataFrame(aa_rows)
    if len(aa_df) > 0:
        aa_df["p_adj_BH"] = benjamini_hochberg(aa_df["p_raw"].fillna(1.0))
        aa_df["significant_BH"] = aa_df["p_adj_BH"] < 0.05
    results["per_aa"] = aa_df

    # ── 5.4 Passive vs Active on χ1-shifted residues (Fisher's I vs N) ───────
    sub_I_sh = df_valid[(df_valid["SASA"] == "I") & (df_valid["chi1_shifted"] == 1)]
    sub_N_sh = df_valid[(df_valid["SASA"] == "N") & (df_valid["chi1_shifted"] == 1)]
    ct_mech = np.array([
        [(sub_I_sh["log_odds"] > 0).sum(), (sub_I_sh["log_odds"] < 0).sum()],
        [(sub_N_sh["log_odds"] > 0).sum(), (sub_N_sh["log_odds"] < 0).sum()],
    ])
    OR_mech, p_mech = stats.fisher_exact(ct_mech, alternative="two-sided")
    results["mechanism_fisher"] = {
        "n_chi1_shifted_I":      len(sub_I_sh),
        "n_chi1_shifted_N":      len(sub_N_sh),
        "passive_pct_I":  round(100 * float((sub_I_sh["log_odds"] > 0).mean()), 1),
        "active_pct_I":   round(100 * float((sub_I_sh["log_odds"] < 0).mean()), 1),
        "passive_pct_N":  round(100 * float((sub_N_sh["log_odds"] > 0).mean()), 1),
        "active_pct_N":   round(100 * float((sub_N_sh["log_odds"] < 0).mean()), 1),
        "odds_ratio":     round(float(OR_mech), 4),
        "p_fisher":       float(p_mech),
    }

    # ── 5.5 SS-change × negative log-odds Fisher's exact ─────────────────────
    for sasa in ("I", "N"):
        sub = df_valid[df_valid["SASA"] == sasa].copy()
        ct = pd.crosstab(sub["ss_change"], (sub["log_odds"] < 0).astype(int))
        ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        OR_f, p_f = stats.fisher_exact(ct.values, alternative="two-sided")
        results[f"ss_fisher_{sasa}"] = {
            "odds_ratio": round(float(OR_f), 4),
            "p_fisher":   float(p_f),
        }

    # ── 5.6 Backbone-shifted vs stable ───────────────────────────────────────
    for sasa in ("I", "N"):
        sub = df_valid[df_valid["SASA"] == sasa]
        shifted = sub[sub["backbone_shifted"] == 1]["log_odds"].dropna()
        stable  = sub[sub["backbone_shifted"] == 0]["log_odds"].dropna()
        if len(shifted) >= 5 and len(stable) >= 5:
            stat_s, p_s = stats.mannwhitneyu(
                shifted, stable, alternative="two-sided")
            results[f"bb_shift_{sasa}"] = {
                "n_shifted": len(shifted), "n_stable": len(stable),
                "median_shifted": round(float(shifted.median()), 4),
                "median_stable":  round(float(stable.median()), 4),
                "p_MWU": float(p_s),
            }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(df_valid: pd.DataFrame, n_excluded: int,
             stats_results: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    gt  = stats_results["global"]
    pt  = stats_results["pdb_level"]
    mf  = stats_results["mechanism_fisher"]
    aa_df = stats_results["per_aa"]

    colors = {"I": "#E69F00", "N": "#56B4E9"}

    # ── Fig 1: Violin + box, I vs N ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for pos, sasa in enumerate(("I", "N")):
        lo = df_valid[df_valid["SASA"] == sasa]["log_odds"].dropna()
        vp = ax.violinplot(lo, positions=[pos], showmedians=False,
                           showextrema=False, widths=0.6)
        for pc in vp["bodies"]:
            pc.set_facecolor(colors[sasa])
            pc.set_alpha(0.35)
        ax.boxplot(lo, positions=[pos], widths=0.18,
                   patch_artist=True,
                   boxprops=dict(facecolor=colors[sasa], alpha=0.85),
                   medianprops=dict(color="black", lw=2.5),
                   flierprops=dict(marker=".", ms=2, alpha=0.2),
                   whiskerprops=dict(lw=1.2),
                   capprops=dict(lw=1.2))
        frac_pos = float((lo > 0).mean())
        frac_neg = float((lo < 0).mean())
        ax.text(pos, float(lo.quantile(0.97)) + 0.12,
                f"n={len(lo)}\n↑{frac_pos:.0%} passive\n↓{frac_neg:.0%} active",
                ha="center", va="bottom", fontsize=9.5)

    ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.7)
    ax.text(-0.42, -2.2, "← Active (RNA-induced strain)",
            fontsize=9, color="#CC0000", fontstyle="italic")
    ax.text(-0.42,  2.1, "Passive (backbone-driven) →",
            fontsize=9, color="#009E73", fontstyle="italic")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Interface", "Non-interface"], fontsize=12)
    ax.set_ylabel("log₂(P[rotamer|φ_B,ψ_B] / P[rotamer|φ_U,ψ_U])", fontsize=11)
    ax.set_ylim(-4.8, 4.8)
    ax.set_title(
        f"Backbone-Conditioned Rotamer Log-Odds\n"
        f"Residue-level p={gt['p_residue_level']:.1e}  |  "
        f"PDB-level p={pt['p_pdb_level']:.1e}  |  "
        f"n_excluded={n_excluded} (both off-library)",
        fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_logodds_violin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved fig1_logodds_violin.png")

    # ── Fig 2: Per-AA mean log-odds heatmap ──────────────────────────────────
    if len(aa_df) > 0 and "AA" in aa_df.columns and not aa_df["AA"].empty:
        aa_order = [aa for aa in AA_DISPLAY_ORDER if aa in aa_df["AA"].values]
        if len(aa_order) > 0:
            pivot_mean = aa_df.pivot(index="AA", columns="SASA", values="mean_lo").reindex(aa_order)
            pivot_sig  = aa_df.pivot(index="AA", columns="SASA", values="significant_BH").reindex(aa_order)

    data = pivot_mean[["I", "N"]].values
    vmax = max(np.nanmax(np.abs(data)), 0.5)

    fig, ax = plt.subplots(figsize=(5.5, max(6, len(aa_order) * 0.42)))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean log₂ odds", shrink=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Interface", "Non-interface"], fontsize=11)
    ax.set_yticks(range(len(aa_order)))
    ax.set_yticklabels(aa_order, fontsize=10)
    ax.set_title("Mean Backbone-Conditioned Log-Odds\n(* BH-adj p < 0.05)",
                 fontweight="bold")

    for i, aa in enumerate(aa_order):
        for j, sasa in enumerate(["I", "N"]):
            v = data[i, j]
            if np.isnan(v):
                continue
            sig = bool(pivot_sig.loc[aa, sasa]) if sasa in pivot_sig.columns else False
            txt_col = "white" if abs(v) > vmax * 0.55 else "black"
            label = f"{v:.2f}" + ("*" if sig else "")
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=8, color=txt_col)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_per_aa_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved fig2_per_aa_heatmap.png")

    # ── Fig 3: Passive vs Active mechanism bar chart ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, sasa in zip(axes, ("I", "N")):
        sub = df_valid[(df_valid["SASA"] == sasa) &
                       (df_valid["chi1_shifted"] == 1)]
        n_pass = int((sub["log_odds"] > 0).sum())
        n_act  = int((sub["log_odds"] < 0).sum())
        n_ind  = int((sub["log_odds"] == 0).sum())
        total  = len(sub)

        bars = ax.bar(
            ["Passive\n(backbone-driven)", "Active\n(RNA-induced)", "Indeterminate"],
            [n_pass, n_act, n_ind],
            color=["#009E73", "#CC0000", "#999999"],
            edgecolor="black", linewidth=0.8, width=0.55)

        for bar, n in zip(bars, [n_pass, n_act, n_ind]):
            pct = 100 * n / total if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{n}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

        title = "Interface" if sasa == "I" else "Non-interface"
        ax.set_title(f"{title} — χ1-shifted residues\n(n={total})",
                     fontweight="bold")
        ax.set_ylabel("Number of residues")
        ax.set_ylim(0, max(n_pass, n_act) * 1.3)

    fig.suptitle(
        f"Mechanism Classification of χ1 Rotamer Transitions\n"
        f"Fisher (I vs N passive/active ratio): "
        f"OR={mf['odds_ratio']:.2f}, p={mf['p_fisher']:.2e}",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_passive_active_mechanism.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved fig3_passive_active_mechanism.png")

    # ── Fig 4: Log-odds vs backbone RMS scatter (backbone-shifted only) ───────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, sasa in zip(axes, ("I", "N")):
        sub = df_valid[
            (df_valid["SASA"] == sasa) &
            (df_valid["backbone_shifted"] == 1)
        ].dropna(subset=["log_odds", "bb_rms_deg"])

        passive = sub[sub["log_odds"] > 0]
        active  = sub[sub["log_odds"] < 0]

        ax.scatter(active["bb_rms_deg"],  active["log_odds"],
                   s=10, alpha=0.30, color="#CC0000",
                   label="Active (log-odds<0)", rasterized=True)
        ax.scatter(passive["bb_rms_deg"], passive["log_odds"],
                   s=10, alpha=0.30, color="#009E73",
                   label="Passive (log-odds>0)", rasterized=True)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)

        rho, p_rho = stats.spearmanr(sub["bb_rms_deg"], sub["log_odds"])
        title = "Interface" if sasa == "I" else "Non-interface"
        ax.set_title(
            f"{title} — backbone-shifted (n={len(sub)})\n"
            f"Spearman ρ={rho:.3f}, p={p_rho:.2e}",
            fontsize=10, fontweight="bold")
        ax.set_xlabel("Backbone RMS |Δφ,Δψ| (°)", fontsize=10)
        ax.legend(fontsize=8, markerscale=2)

    axes[0].set_ylabel("log₂(P_bound_bb / P_unbound_bb)", fontsize=10)
    fig.suptitle("Log-Odds vs Backbone Deviation (backbone-shifted residues only)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "fig4_logodds_vs_bbrms.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved fig4_logodds_vs_bbrms.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Export
# ─────────────────────────────────────────────────────────────────────────────

def export_all(df_lo: pd.DataFrame, df_valid: pd.DataFrame,
               n_excluded: int, stats_results: dict,
               output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full per-residue table (all rows, including excluded)
    df_lo.to_csv(output_dir / "per_residue_logodds.csv", index=False)

    # Per-AA stats
    aa_df = stats_results.get("per_aa", pd.DataFrame())
    if not aa_df.empty:
        aa_df.to_csv(output_dir / "per_aa_stats_final.csv", index=False)

    # Statistical summary
    gt = stats_results["global"]
    pt = stats_results["pdb_level"]
    mf = stats_results["mechanism_fisher"]

    summary = {
        "n_total_rows":          len(df_lo),
        "n_excluded_off_library": n_excluded,
        "n_valid":               len(df_valid),
        "n_I":                   gt["n_I"],
        "n_N":                   gt["n_N"],
        "median_I":              gt["median_I"],
        "median_N":              gt["median_N"],
        "mean_I":                gt["mean_I"],
        "mean_N":                gt["mean_N"],
        "p_residue_level_MWU":   gt["p_residue_level"],
        "n_pdbs_I":              pt["n_pdbs_I"],
        "n_pdbs_N":              pt["n_pdbs_N"],
        "pdb_median_I":          pt["median_I"],
        "pdb_median_N":          pt["median_N"],
        "p_pdb_level_MWU":       pt["p_pdb_level"],
        "chi1_shifted_I_passive_pct": mf["passive_pct_I"],
        "chi1_shifted_I_active_pct":  mf["active_pct_I"],
        "chi1_shifted_N_passive_pct": mf["passive_pct_N"],
        "chi1_shifted_N_active_pct":  mf["active_pct_N"],
        "mechanism_fisher_OR":   mf["odds_ratio"],
        "mechanism_fisher_p":    mf["p_fisher"],
    }
    for sasa in ("I", "N"):
        key = f"ss_fisher_{sasa}"
        if key in stats_results:
            summary[f"ss_fisher_{sasa}_OR"] = stats_results[key]["odds_ratio"]
            summary[f"ss_fisher_{sasa}_p"]  = stats_results[key]["p_fisher"]
    for sasa in ("I", "N"):
        key = f"bb_shift_{sasa}"
        if key in stats_results:
            summary[f"bb_shift_{sasa}_p"] = stats_results[key]["p_MWU"]
            summary[f"bb_shift_{sasa}_median_shifted"] = stats_results[key]["median_shifted"]
            summary[f"bb_shift_{sasa}_median_stable"]  = stats_results[key]["median_stable"]

    pd.Series(summary).to_csv(
        output_dir / "statistical_summary_final.csv", header=["value"])
    logging.info("Saved per_residue_logodds.csv, per_aa_stats_final.csv, "
                 "statistical_summary_final.csv")

    # Console summary
    print("\n" + "=" * 70)
    print("BACKBONE LOG-ODDS ANALYSIS — KEY RESULTS")
    print("=" * 70)
    print(f"\n  Valid residues:    {len(df_valid):,}  "
          f"(excluded off-library: {n_excluded:,})")
    print(f"\n  Global (residue-level):")
    print(f"    Interface    median={gt['median_I']:.4f}  mean={gt['mean_I']:.4f}  n={gt['n_I']}")
    print(f"    Non-interf.  median={gt['median_N']:.4f}  mean={gt['mean_N']:.4f}  n={gt['n_N']}")
    print(f"    Mann-Whitney p = {gt['p_residue_level']:.3e}")
    print(f"\n  PDB-level:")
    print(f"    Interface    median={pt['median_I']:.4f}  n_pdbs={pt['n_pdbs_I']}")
    print(f"    Non-interf.  median={pt['median_N']:.4f}  n_pdbs={pt['n_pdbs_N']}")
    print(f"    Mann-Whitney p = {pt['p_pdb_level']:.3e}")
    print(f"\n  Mechanism (χ1-shifted residues):")
    print(f"    Interface    passive={mf['passive_pct_I']:.1f}%  active={mf['active_pct_I']:.1f}%")
    print(f"    Non-interf.  passive={mf['passive_pct_N']:.1f}%  active={mf['active_pct_N']:.1f}%")
    print(f"    Fisher OR={mf['odds_ratio']:.3f}  p={mf['p_fisher']:.3e}")

    aa_df_sig = aa_df[aa_df.get("significant_BH", pd.Series(dtype=bool))] \
        if not aa_df.empty else pd.DataFrame()
    if len(aa_df_sig) > 0:
        print(f"\n  Per-AA significant (BH-adj p < 0.05):")
        for _, r in aa_df_sig.iterrows():
            print(f"    {r['AA']:3s} ({r['SASA']}): "
                  f"median={r['median_lo']:.3f}  "
                  f"p_adj={r['p_adj_BH']:.2e}  n={r['n']}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backbone-Conditioned Rotamer Log-Odds Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument(
        "--torsion", type=Path, required=True,
        help="Path to torsion angle JSON (preferred) or TSV/CSV file")
    parser.add_argument(
        "--bbd-dir", type=Path, required=True,
        help="Directory containing the four BBD JSON library files")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("logodds_output"),
        help="Output directory for CSVs and figures (default: ./logodds_output)")
    parser.add_argument(
        "--bb-sep", type=int, default=30,
        help="Backbone grid resolution in degrees (default: 30; must match library build)")
    args = parser.parse_args()

    # ── Load torsion data ──────────────────────────────────────────────────
    torsion_path = args.torsion
    if not torsion_path.exists():
        alt_path = Path(__file__).resolve().parent.parent / "input_files" / args.torsion.name
        if alt_path.exists():
            torsion_path = alt_path

    logging.info(f"Loading torsion data: {torsion_path}")
    suffix = torsion_path.suffix.lower()
    if suffix == ".json":
        df = pd.read_json(torsion_path)
    else:
        for sep in (",", "\t"):
            df = pd.read_csv(torsion_path, sep=sep)
            if "LABEL" in df.columns or "AA" in df.columns:
                break
    logging.info(f"Loaded {len(df):,} rows")

    # ── Load BBD libraries ─────────────────────────────────────────────────
    libs = load_bbd_libraries(args.bbd_dir)

    # ── Compute log-odds ───────────────────────────────────────────────────
    logging.info("Computing per-residue backbone-conditioned log-odds...")
    df_lo = compute_logodds(df, libs, bb_sep=args.bb_sep)
    logging.info(f"Computed {len(df_lo):,} rows "
                 f"(I={len(df_lo[df_lo['SASA']=='I']):,}, "
                 f"N={len(df_lo[df_lo['SASA']=='N']):,})")

    if df_lo.empty:
        logging.error("No log-odds computed. "
                      "Check that --torsion and --bbd-dir paths are correct.")
        return

    # ── Filter pseudocount-only rows ───────────────────────────────────────
    df_valid, df_excl = filter_valid(df_lo)
    n_excluded = len(df_excl)
    logging.info(f"Valid: {len(df_valid):,}  |  "
                 f"Excluded (both off-library): {n_excluded:,}")

    # ── Statistics ─────────────────────────────────────────────────────────
    logging.info("Running statistical tests...")
    stats_results = run_statistics(df_valid)

    # ── Export and plot ────────────────────────────────────────────────────
    export_all(df_lo, df_valid, n_excluded, stats_results, args.output_dir)
    plot_all(df_valid, n_excluded, stats_results, args.output_dir)
    logging.info(f"All outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
