"""
utils/top8000.py
----------------
Reader for the MolProbity Top8000 percentile-contour grids (Richardson lab reference_data,
CC-BY-4.0; location: paths key "molprobity_reference_dir"). Cite: Williams et al. (2018)
Protein Sci 27:293 (MolProbity); Hintze et al. (2016) Proteins 84:1177 (Top8000).

Each .data file is an N-dimensional table of bin-centre values with a header such as
    #   x1: -180.0 180.0 180 true        (lower bound, upper bound, number of bins, wrapping)
The value at a point is linearly interpolated between bin centres, wrapping periodic axes, as in
MolProbity's NDimTable.
"""

import itertools

import numpy as np

from utils.paths import get_path

TOP8000_DIR = get_path("molprobity_reference_dir") / "Top8000"


class Top8000Grid:
    def __init__(self, path):
        self.lower, self.upper, self.nbins, self.wrap = [], [], [], []
        rows = []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    parts = line[1:].split()
                    if len(parts) == 5 and parts[0].startswith("x") and parts[0].endswith(":"):
                        self.lower.append(float(parts[1]))
                        self.upper.append(float(parts[2]))
                        self.nbins.append(int(parts[3]))
                        self.wrap.append(parts[4].lower() == "true")
                elif line.strip():
                    rows.append([float(v) for v in line.split()])
        self.lower, self.upper = np.array(self.lower), np.array(self.upper)
        self.nbins = np.array(self.nbins)
        self.width = (self.upper - self.lower) / self.nbins
        rows = np.array(rows)
        idx = np.floor((rows[:, :-1] - self.lower) / self.width).astype(int)
        self.values = np.zeros(tuple(self.nbins))
        self.values[tuple(idx.T)] = rows[:, -1]

    @property
    def ndim(self):
        return len(self.nbins)

    def centres(self, axis):
        return self.lower[axis] + (np.arange(self.nbins[axis]) + 0.5) * self.width[axis]

    def __call__(self, *coords):
        """Interpolated value at points; coords are arrays of angles (degrees), one per axis."""
        pts = np.stack([np.asarray(c, float) for c in coords], axis=-1)
        span = self.upper - self.lower
        pts = np.where(self.wrap, (pts - self.lower) % span + self.lower, pts)
        frac = (pts - self.lower) / self.width - 0.5
        i0 = np.floor(frac).astype(int)
        t = frac - i0
        out = np.zeros(pts.shape[:-1])
        for corner in itertools.product((0, 1), repeat=self.ndim):
            corner = np.array(corner)
            idx = i0 + corner
            idx = np.where(self.wrap, idx % self.nbins, np.clip(idx, 0, self.nbins - 1))
            weight = np.prod(np.where(corner, t, 1 - t), axis=-1)
            out += weight * self.values[tuple(np.moveaxis(idx, -1, 0))]
        return np.where(np.isnan(pts).any(axis=-1), np.nan, out)
