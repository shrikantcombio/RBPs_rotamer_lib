#!/usr/bin/env python3
"""
backbone_independent_rotamer_lib.py

Calculates Backbone-Independent (BBI) Rotamer Libraries using von Mises
kernel-weighted circular statistics for protein structures.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import vonmises

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BackboneIndependentRotamerLibrary:
    """
    Class for calculating backbone-independent rotamer libraries.
    """

    RESIDUE_CHI_COUNTS = {
        "ARG": 4, "LYS": 4, "GLU": 3, "GLN": 3, "MET": 3,
        "ASP": 2, "ASN": 2, "HIS": 2, "LEU": 2, "ILE": 2,
        "PHE": 2, "PRO": 2, "TRP": 2, "TYR": 2,
        "CYS": 1, "SER": 1, "THR": 1, "VAL": 1
    }

    # Residues with symmetric terminal chi angles (180-degree periodicity).
    # Key: residue name -> 0-based index of the symmetric chi.
    SYMMETRIC_CHI = {
        "ASP": 1,
        "GLU": 2,
        "PHE": 1,
        "TYR": 1,
    }

    def __init__(self, rot_bin_lib=None, min_count=5):
        """
        Parameters:
        -----------
        rot_bin_lib : dict, optional
            Custom rotamer binning rules mapping (residue, chi_idx) to number of splits.
        min_count : int
            Minimum observations for a rotamer bin to be reported. Default 5.
        """
        self.rot_bin_lib = rot_bin_lib if rot_bin_lib is not None else {}
        self.min_count = min_count

    @staticmethod
    def fold_symmetric_chi(angle):
        """
        Canonicalize a symmetric terminal chi angle into [-90, 90).
        Applies to ASP chi2, GLU chi3, PHE chi2, TYR chi2.
        Matches Dunbrack 2011 and Richardson/MolProbity conventions.
        """
        if angle >= 90.0:
            return angle - 180.0
        if angle < -90.0:
            return angle + 180.0
        return angle

    @staticmethod
    def rotameric_class(angle, n_splits=3):
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
        return 0 if 0 < angle <= 120 else (2 if -120 < angle <= 0 else 1)

    @staticmethod
    def weighted_circmean(angles_rad, weights):
        angles_arr = np.array(angles_rad)
        weights_arr = np.array(weights)
        complex_numbers = np.exp(1j * angles_arr)
        weighted_sum = np.sum(weights_arr * complex_numbers)
        return float(np.angle(weighted_sum))

    @staticmethod
    def weighted_circstd(angles_rad, weights, mean_angle_rad):
        angles_arr = np.array(angles_rad)
        weights_arr = np.array(weights)
        angle_diffs = np.angle(np.exp(1j * (angles_arr - mean_angle_rad)))
        sum_w = np.sum(weights_arr)
        if sum_w <= 0:
            return 0.0
        weighted_variance = np.sum(weights_arr * (angle_diffs ** 2)) / sum_w
        return float(np.sqrt(max(0.0, weighted_variance)))

    def generate_library(self, df, state="B", sasa_filter=None, residues=None):
        if residues is None:
            residues = list(self.RESIDUE_CHI_COUNTS.keys())

        df_work = df.copy()
        if sasa_filter:
            if sasa_filter == "S":
                df_work = df_work[df_work["SASA"].astype(str).isin(["I", "N", "S"])]
            else:
                df_work = df_work[df_work["SASA"].astype(str).str.startswith(sasa_filter)]

        lib_dict = {}
        for aa in residues:
            nchi = self.RESIDUE_CHI_COUNTS[aa]
            chi_cols = [f"{state}_CHI{n+1}" if f"{state}_CHI{n+1}" in df_work.columns else f"CHI{n+1}" for n in range(nchi)]

            res_data = df_work[df_work["LABEL"].str[:3] == aa]
            if len(res_data) == 0:
                continue

            # Build per-depth valid masks: cumulative non-null across chi_1..chi_n.
            # A residue with missing chi_k contributes to depths 1..(k-1) only.
            # For rotamer-state classification and probability, use only residues
            # where ALL chi are non-null (deepest valid population).
            depth_valid_mask = np.ones(len(res_data), dtype=bool)
            for n in range(nchi):
                depth_valid_mask = depth_valid_mask & res_data[chi_cols[n]].notna().values
            res_data_full = res_data[depth_valid_mask]

            if len(res_data_full) == 0:
                continue

            sym_chi_idx = self.SYMMETRIC_CHI.get(aa, None)
            chi_classes = []
            for n in range(nchi):
                col = chi_cols[n]
                n_splits = self.rot_bin_lib.get((aa, n), 3)
                if sym_chi_idx is not None and n == sym_chi_idx:
                    angles = res_data_full[col].apply(self.fold_symmetric_chi)
                else:
                    angles = res_data_full[col]
                chi_classes.append(angles.apply(lambda x: self.rotameric_class(x, n_splits)).values)

            effect_chi_classes = np.array(chi_classes)
            if effect_chi_classes.size == 0:
                continue

            unique_rot_states = np.unique(effect_chi_classes.T, axis=0)
            total_count = len(res_data_full)

            results = []
            for c in unique_rot_states:
                mask = np.ones(total_count, dtype=bool)
                for i, l in enumerate(c):
                    mask &= (effect_chi_classes[i] == l)

                count_each = int(np.sum(mask))
                if count_each < self.min_count:
                    continue

                prob = float(count_each) / float(total_count)
                angles_sub = res_data_full[mask]

                line = np.zeros(2 * nchi + 3)
                line[0] = count_each
                line[1] = total_count
                line[2] = prob

                for n in range(nchi):
                    chi_col_name = chi_cols[n]
                    raw_angles = angles_sub[chi_col_name].values
                    # Fold symmetric terminal chi before computing circular statistics.
                    if sym_chi_idx is not None and n == sym_chi_idx:
                        raw_angles = np.array([self.fold_symmetric_chi(a) for a in raw_angles])
                    ang_rad = np.radians(raw_angles)

                    try:
                        kappa, loc, _ = vonmises.fit(ang_rad, fscale=1)
                        weights = vonmises(loc=loc, kappa=kappa).pdf(ang_rad)
                    except Exception:
                        weights = np.ones_like(ang_rad)

                    wm_rad = self.weighted_circmean(ang_rad, weights)
                    wstd_rad = self.weighted_circstd(ang_rad, weights, wm_rad)

                    line[3 + n] = np.degrees(wm_rad)
                    line[3 + n + nchi] = np.degrees(wstd_rad)

                results.append(line)

            if results:
                columns = ["_COUNT", "COUNT", "PROB"]
                for i in range(1, nchi + 1):
                    columns.append(f"CHI{i}")
                for i in range(1, nchi + 1):
                    columns.append(f"CHI{i}Sig")

                res_df = pd.DataFrame(results, columns=columns)
                res_df.insert(0, "AA", aa)
                res_df["_COUNT"] = res_df["_COUNT"].astype(int)
                res_df["COUNT"] = res_df["COUNT"].astype(int)
                lib_dict[aa] = res_df

        return lib_dict
