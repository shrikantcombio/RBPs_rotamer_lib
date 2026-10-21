#!/usr/bin/env python3
"""
rotamer_definitions.py

Rotamer state definitions shared by the backbone-dependent and backbone-independent library
builders.

sp3-sp3 chis (chi1 of every residue, chi2 of LEU/ILE/LYS/ARG/MET/GLU/GLN, ...) have three staggered
minima and are binned into p (0, 120], t (120, 240], m (-120, 0].

The terminal chi of ASP, GLU, PHE, TYR, ASN, GLN, HIS and TRP rotates a planar (sp2) group. Its
distribution has no staggered minima, so the three 120-degree bins cut through populated modes:
folding PHE/TYR chi2 into [-90, 90) put the boundary on the ring's most populated orientation
(~ +-90), and the bin edge at 0 split the single ASP chi2 / GLU chi3 peak in two. Following
Dunbrack (2010, 2011) and Lovell et al. (2000), these chis are binned by residue-specific ranges
whose edges lie at density minima (see TERMINAL_SP2_CHI), and residues with a two-fold symmetric
group (ASP, GLU, PHE, TYR) use 180-degree periodic statistics.
"""

import bisect

import numpy as np
from scipy.stats import vonmises

# Number of chi angles per standard amino acid (excluding ALA and GLY)
RESIDUE_CHI_COUNTS = {
    "ARG": 4, "LYS": 4, "GLU": 3, "GLN": 3, "MET": 3,
    "ASP": 2, "ASN": 2, "HIS": 2, "LEU": 2, "ILE": 2,
    "PHE": 2, "PRO": 2, "TRP": 2, "TYR": 2,
    "CYS": 1, "SER": 1, "THR": 1, "VAL": 1,
}

# Terminal sp2 chis: residue -> (0-based chi index, period, canonical range start, bin edges).
#   period 180: two-fold symmetric group (OD1/OD2, OE1/OE2, ring flip); angles are folded into
#               [start, start + 180) and statistics are computed on doubled angles.
#   bin edges : boundaries in (-180, 180]; an angle belongs to the bin of the next edge above it,
#               angles above the last edge wrap into the first bin. No edges = one bin.
# Edges follow the density minima of the 187-complex bound dataset (20-degree histograms):
#   ASP chi2, GLU chi3 : single peak near 0 when folded; minima at +-90
#   PHE chi2, TYR chi2 : single peak near 90 when folded (minimum near 30); reported in Dunbrack's
#                        range [-30, 150)
#   HIS chi2           : modes near -75, 75, 170; minima near 0 and +-120
#   TRP chi2           : modes near -100, 0, 90 (Lovell m-90, m0, m95); minima near -50, 50, 180
#   ASN chi2           : modes near -30 and 120 (Lovell m-20, m120); minima near -150 and 60
#   GLN chi3           : broad, no clear minimum; one bin
TERMINAL_SP2_CHI = {
    "ASP": (1, 180, -90.0, ()),
    "GLU": (2, 180, -90.0, ()),
    "PHE": (1, 180, -30.0, ()),
    "TYR": (1, 180, -30.0, ()),
    "HIS": (1, 360, -180.0, (-120.0, 0.0, 120.0)),
    "TRP": (1, 360, -180.0, (-50.0, 50.0, 180.0)),
    "ASN": (1, 360, -180.0, (-150.0, 60.0)),
    "GLN": (2, 360, -180.0, ()),
}

# Label of each terminal sp2 bin (index order of terminal_sp2_class): the approximate mode in
# degrees, '*' for a single bin without a defined mode
TERMINAL_SP2_LABELS = {
    "ASP": ("0",), "GLU": ("0",), "PHE": ("90",), "TYR": ("90",),
    "HIS": ("180", "-75", "75"),
    "TRP": ("-100", "0", "90"),
    "ASN": ("120", "-30"),
    "GLN": ("*",),
}


def rotameric_class(angle, n_splits=3):
    """Classifies an sp3 chi angle (degrees) into a rotameric bin index."""
    if n_splits == 3:
        if 0 < angle <= 120:
            return 0
        elif -120 < angle <= 0:
            return 2
        return 1

    elif n_splits == 4:
        if 45 < angle <= 135:
            return 0
        elif -123 < angle <= -45:
            return 2
        elif angle > 135 or angle <= -135:
            return 1
        return 3

    elif n_splits == 6:
        if -30 < angle <= 30:
            return 0
        elif 30 < angle <= 90:
            return 1
        elif 90 < angle <= 150:
            return 2
        elif angle > 150 or angle <= -150:
            return 3
        elif -150 < angle <= -90:
            return 4
        return 5

    # Fallback 3-split default
    return 0 if 0 < angle <= 120 else (2 if -120 < angle <= 0 else 1)


def terminal_sp2(aa, chi_idx):
    """(period, range start, edges) if chi `chi_idx` (0-based) of `aa` is a terminal sp2 chi, else None."""
    spec = TERMINAL_SP2_CHI.get(aa)
    if spec is None or spec[0] != chi_idx:
        return None
    return spec[1:]


def canonical_angle(angle, period, start):
    """Maps an angle into [start, start + period)."""
    return (angle - start) % period + start


def terminal_sp2_class(angle, edges):
    """Bin index of a terminal sp2 chi (already in canonical range) for the given edges."""
    if not edges:
        return 0
    k = bisect.bisect_left(edges, angle)
    return 0 if k == len(edges) else k


def terminal_sp2_label(aa, chi_idx, angle):
    """Mode label of a terminal sp2 chi (see TERMINAL_SP2_LABELS), or None for other chis."""
    sp2 = terminal_sp2(aa, chi_idx)
    if sp2 is None:
        return None
    period, start, edges = sp2
    return TERMINAL_SP2_LABELS[aa][terminal_sp2_class(canonical_angle(angle, period, start), edges)]


def chi_class(aa, chi_idx, angle, rot_bin_lib=None):
    """Rotamer bin of chi `chi_idx` (0-based) of residue `aa`."""
    sp2 = terminal_sp2(aa, chi_idx)
    if sp2 is not None:
        period, start, edges = sp2
        return terminal_sp2_class(canonical_angle(angle, period, start), edges)
    n_splits = (rot_bin_lib or {}).get((aa, chi_idx), 3)
    return rotameric_class(angle, n_splits)


def weighted_circmean(angles_rad, weights):
    """Weighted circular mean of angles (radians)."""
    weighted_sum = np.sum(np.asarray(weights) * np.exp(1j * np.asarray(angles_rad)))
    return float(np.angle(weighted_sum))


def weighted_circstd(angles_rad, weights, mean_angle_rad):
    """Weighted RMS circular deviation from `mean_angle_rad` (radians)."""
    weights_arr = np.asarray(weights)
    angle_diffs = np.angle(np.exp(1j * (np.asarray(angles_rad) - mean_angle_rad)))
    sum_w = np.sum(weights_arr)
    if sum_w <= 0:
        return 0.0
    weighted_variance = np.sum(weights_arr * (angle_diffs ** 2)) / sum_w
    return float(np.sqrt(max(0.0, weighted_variance)))


def chi_mean_and_sigma(aa, chi_idx, angles_deg):
    """
    von Mises-weighted circular mean and deviation (degrees) of one chi within a rotamer.

    For two-fold symmetric terminal chis the statistics are computed on doubled angles (so that
    x and x + 180 are the same conformation) and the mean is reported in the canonical range.
    """
    sp2 = terminal_sp2(aa, chi_idx)
    factor = 2.0 if sp2 is not None and sp2[0] == 180 else 1.0
    ang_rad = np.radians(np.asarray(angles_deg, dtype=float)) * factor

    try:
        kappa, loc, _ = vonmises.fit(ang_rad, fscale=1)
        weights = vonmises(loc=loc, kappa=kappa).pdf(ang_rad)
    except Exception:
        weights = np.ones_like(ang_rad)

    wm_rad = weighted_circmean(ang_rad, weights)
    wstd_rad = weighted_circstd(ang_rad, weights, wm_rad)
    mean_deg = np.degrees(wm_rad) / factor
    if sp2 is not None:
        mean_deg = canonical_angle(mean_deg, sp2[0], sp2[1])
    return float(mean_deg), float(np.degrees(wstd_rad) / factor)


def classify_chis(aa, chi_frame, chi_cols, rot_bin_lib=None):
    """Rotamer bins of every chi column, as an array of shape (n_chi, n_residues)."""
    return np.array([
        [chi_class(aa, n, a, rot_bin_lib) for a in chi_frame[col].values]
        for n, col in enumerate(chi_cols)
    ])
