"""Integration checks for the public inputs, exports and synthetic experiment."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ihc_demo.analysis import recovery
from ihc_demo.measurement import quantify
from ihc_demo.phantom import build_phantom
from ihc_demo.stress import mask_stress


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb, cls.labels, cls.truth = build_phantom(seed=17)

    def test_generator_is_reproducible(self):
        rgb, labels, truth = build_phantom(seed=17)
        np.testing.assert_array_equal(rgb, self.rgb)
        np.testing.assert_array_equal(labels, self.labels)
        pd.testing.assert_frame_equal(truth, self.truth)

    def test_default_demo_exposes_boundary_bias(self):
        stress = mask_stress(self.rgb, self.labels, self.truth)
        rings = stress[stress.compartment == "ring"].set_index("mask_condition")
        self.assertEqual(rings.cells_measured.tolist(), [len(self.truth)] * 3)
        self.assertLess(rings.loc["reference", "mean_absolute_error_od"], 0.025)
        self.assertGreater(rings.loc["eroded_2px", "mean_absolute_error_od"],
                           4 * rings.loc["reference", "mean_absolute_error_od"])
        self.assertLess(rings.loc["eroded_2px", "rank_correlation"],
                        rings.loc["reference", "rank_correlation"])

    def test_measure_command_preserves_labels_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            Image.fromarray(self.rgb).save(folder / "field.png")
            Image.fromarray(self.labels).save(folder / "nuclei.tif")
            out = folder / "measurement"
            result = subprocess.run([
                sys.executable, "-m", "ihc_demo", "measure", "--rgb", str(folder / "field.png"),
                "--labels", str(folder / "nuclei.tif"), "--out", str(out),
            ], capture_output=True, text=True, check=True)
            self.assertIn(f"Measured {len(self.truth)} cells", result.stdout)
            exported = pd.read_csv(out / "cells.csv")
            cells, rings, dab = quantify(self.rgb, self.labels)
            np.testing.assert_array_equal(exported.cell_id, cells.cell_id)
            np.testing.assert_allclose(exported.ring_mean_dab_od, cells.ring_mean_dab_od, rtol=1e-6)
            with Image.open(out / "rings.tif") as image:
                np.testing.assert_array_equal(np.array(image), rings)
            with Image.open(out / "dab_od.tif") as image:
                np.testing.assert_allclose(np.array(image), dab, rtol=1e-6)

    def test_missing_measurements_have_explicit_denominators(self):
        cells, _, _ = quantify(self.rgb, self.labels, outer_scale=1)
        scores = recovery(cells, self.truth).set_index("compartment")
        self.assertEqual(scores.loc["ring", "cells_total"], len(self.truth))
        self.assertEqual(scores.loc["ring", "cells_measured"], 0)
        self.assertTrue(np.isnan(scores.loc["ring", "rank_correlation"]))


if __name__ == "__main__":
    unittest.main()
