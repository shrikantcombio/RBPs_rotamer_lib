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

import rotamer_definitions as rotdef

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BackboneIndependentRotamerLibrary:
    """
    Class for calculating backbone-independent rotamer libraries.
    """

    RESIDUE_CHI_COUNTS = rotdef.RESIDUE_CHI_COUNTS
    # Terminal sp2 chis use residue-specific bins and, for two-fold symmetric groups,
    # 180-degree periodic statistics: see rotamer_definitions.py
    TERMINAL_SP2_CHI = rotdef.TERMINAL_SP2_CHI

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

    rotameric_class = staticmethod(rotdef.rotameric_class)
    weighted_circmean = staticmethod(rotdef.weighted_circmean)
    weighted_circstd = staticmethod(rotdef.weighted_circstd)

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

            # Working population: residues with every chi present (truncated side chains have NaN chis)
            full_valid_mask = np.ones(len(res_data), dtype=bool)
            for col in chi_cols:
                full_valid_mask &= res_data[col].notna().values
            res_data_full = res_data[full_valid_mask]

            if len(res_data_full) == 0:
                continue

            # Rotamer bin of every chi (terminal sp2 chis use residue-specific bins)
            effect_chi_classes = rotdef.classify_chis(aa, res_data_full, chi_cols, self.rot_bin_lib)
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

                # Denominator includes rotamers dropped for count < min_count
                prob = float(count_each) / float(total_count)
                angles_sub = res_data_full[mask]

                line = np.zeros(2 * nchi + 3)
                line[0] = count_each
                line[1] = total_count
                line[2] = prob

                for n in range(nchi):
                    mean_deg, sigma_deg = rotdef.chi_mean_and_sigma(aa, n, angles_sub[chi_cols[n]].values)
                    line[3 + n] = mean_deg
                    line[3 + n + nchi] = sigma_deg

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
