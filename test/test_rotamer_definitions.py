#!/usr/bin/env python3
"""
test_rotamer_definitions.py
---------------------------
Regression tests for the v2 rotamer definitions (October 2026). Each test fails on the v1 code:

  - PHE/TYR chi2 was folded into [-90, 90), splitting the ring orientation near +-90 into two rotamers
  - the 0-degree bin edge split the single ASP chi2 peak in two
  - the TRP chi2 mode near 0 was split by the same edge
  - circular means of folded symmetric chis were taken on a 360-degree circle (89 and -89 -> 0)
  - a backbone angle of exactly 165 was labelled 180 instead of -180, and ties were rounded to even

Run:  python test/test_rotamer_definitions.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "script"
sys.path.insert(0, str(SCRIPT_DIR))

import rotamer_definitions as rotdef
from backbone_dependent_rotamer_lib import BackboneDependentRotamerLibrary
from backbone_independent_rotamer_lib import BackboneIndependentRotamerLibrary


class TerminalSp2BinTests(unittest.TestCase):
    def test_phe_ring_flip_and_near_90_share_a_bin(self):
        classes = {rotdef.chi_class("PHE", 1, a) for a in (89.0, 91.0, -89.0, -91.0, 100.0, -80.0)}
        self.assertEqual(classes, {0})

    def test_asp_chi2_peak_near_zero_is_one_bin(self):
        self.assertEqual(rotdef.chi_class("ASP", 1, -20.0), rotdef.chi_class("ASP", 1, 25.0))
        self.assertEqual(rotdef.chi_class("ASP", 1, 160.0), rotdef.chi_class("ASP", 1, -20.0))

    def test_trp_chi2_has_three_modes_with_zero_intact(self):
        self.assertEqual(rotdef.chi_class("TRP", 1, -10.0), rotdef.chi_class("TRP", 1, 10.0))
        self.assertEqual(len({rotdef.chi_class("TRP", 1, a) for a in (-100.0, 0.0, 90.0)}), 3)

    def test_his_keeps_standard_bins(self):
        angles = (-170.0, -125.0, -70.0, -5.0, 5.0, 75.0, 119.0, 125.0, 170.0)
        for a in angles:
            for b in angles:
                self.assertEqual(rotdef.chi_class("HIS", 1, a) == rotdef.chi_class("HIS", 1, b),
                                 rotdef.rotameric_class(a) == rotdef.rotameric_class(b))

    def test_sp3_chis_unchanged(self):
        for a in (-179.0, -60.0, 0.0, 60.0, 120.0, 175.0):
            self.assertEqual(rotdef.chi_class("PHE", 0, a), rotdef.rotameric_class(a))
            self.assertEqual(rotdef.chi_class("LEU", 1, a), rotdef.rotameric_class(a))


class ProlineTests(unittest.TestCase):
    def test_pucker_from_chi1_sign(self):
        self.assertEqual(rotdef.proline_pucker(28.0), "endo")
        self.assertEqual(rotdef.proline_pucker(-27.0), "exo")
        self.assertEqual(rotdef.proline_pucker(-35.0), "exo")  # 6-state 'm', not a different rotamer


class StatisticsTests(unittest.TestCase):
    def test_symmetric_mean_uses_180_periodicity(self):
        mean, sigma = rotdef.chi_mean_and_sigma("PHE", 1, [89.0, -89.0, 91.0, -91.0])
        self.assertAlmostEqual(mean, 90.0, places=3)
        self.assertLess(sigma, 2.0)

    def test_symmetric_mean_in_canonical_range(self):
        mean, _ = rotdef.chi_mean_and_sigma("ASP", 1, [170.0, -170.0, 175.0])
        self.assertTrue(-90.0 <= mean < 90.0)
        self.assertLess(abs(mean), 10.0)  # 170 / -170 / 175 are -10 / 10 / -5 after the ring flip


class BackboneBinTests(unittest.TestCase):
    def test_bin_centred_on_180_is_minus_180(self):
        f = BackboneDependentRotamerLibrary.get_closest_angle
        self.assertEqual(f(165.0), -180)
        self.assertEqual(f(179.9), -180)
        self.assertEqual(f(-180.0), -180)
        self.assertEqual(f(164.99), 150)

    def test_ties_go_to_upper_bin(self):
        f = BackboneDependentRotamerLibrary.get_closest_angle
        self.assertEqual([f(15.0), f(45.0), f(-45.0), f(-15.0)], [30, 60, -30, 0])


class LibraryTests(unittest.TestCase):
    def _phe_frame(self):
        rng = np.random.default_rng(0)
        chi2 = np.concatenate([rng.normal(95, 8, 60), rng.normal(-85, 8, 60)])  # one mode, both labellings
        n = len(chi2)
        return pd.DataFrame({"LABEL": ["PHE A   1"] * n, "SASA": ["N"] * n,
                             "B_PHI": [-65.0] * n, "B_PSI": [-40.0] * n,
                             "B_CHI1": rng.normal(-65, 8, n), "B_CHI2": chi2})

    def test_bbi_phe_ring_is_one_rotamer(self):
        lib = BackboneIndependentRotamerLibrary().generate_library(self._phe_frame(), state="B")
        self.assertEqual(len(lib["PHE"]), 1)
        self.assertAlmostEqual(lib["PHE"]["CHI2"].iloc[0], 95.0, delta=3.0)

    def test_bbd_phe_ring_is_one_rotamer(self):
        lib = BackboneDependentRotamerLibrary().generate_library(self._phe_frame(), state="B")
        self.assertEqual(len(lib["PHE"]), 1)
        self.assertAlmostEqual(lib["PHE"]["PROB"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
