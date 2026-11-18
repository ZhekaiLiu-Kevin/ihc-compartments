import unittest

import numpy as np
import pandas as pd

from ihc_demo.measurement import HDAB_BASIS
from ihc_demo.stress import _perturb_instances, mask_stress


class MaskStressTests(unittest.TestCase):
    def test_touching_instances_shrink_at_shared_boundary(self):
        labels = np.zeros((13, 20), dtype=np.int32)
        labels[2:11, 2:10] = 3
        labels[2:11, 10:18] = 9
        eroded, dilated = _perturb_instances(labels)
        np.testing.assert_array_equal(eroded[6, 7:13], [3, 0, 0, 0, 0, 9])
        np.testing.assert_array_equal(dilated[labels > 0], labels[labels > 0])

    def test_dilation_uses_euclidean_distance_and_resolves_exact_ties(self):
        labels = np.zeros((9, 9), dtype=np.int32)
        labels[4, 2], labels[4, 6] = 17, 5
        eroded, dilated = _perturb_instances(labels)
        self.assertFalse(np.any(eroded))
        self.assertEqual(dilated[4, 4], 5)
        self.assertEqual(dilated[4, 3], 17)
        self.assertEqual(dilated[2, 3], 0)  # sqrt(5) from the nearest input

    def test_stress_uses_fixed_targets_and_retains_disappearing_instances(self):
        labels = np.zeros((22, 22), dtype=np.int32)
        labels[5:14, 5:14] = 1
        labels[18, 18] = 2
        concentration = (labels > 0) * 0.8
        rgb = np.round(256 * np.exp(-concentration[..., None] * HDAB_BASIS[:, 1]) - 1)
        rgb = rgb.astype(np.uint8)
        truth = pd.DataFrame({
            "cell_id": [1, 2], "target_nuclear_dab_od": [0.8, 0.8],
            "target_ring_dab_od": [0.0, 0.0],
        })
        original = labels.copy()
        result = mask_stress(rgb, labels, truth).set_index(["mask_condition", "compartment"])
        self.assertEqual(len(result), 6)
        self.assertTrue((result.cells_total == 2).all())
        self.assertEqual(result.loc[("eroded_2px", "nuclear"), "cells_measured"], 1)
        self.assertGreater(result.loc[("eroded_2px", "ring"), "mean_signed_error_od"], 0.1)
        self.assertLess(result.loc[("dilated_2px", "nuclear"), "mean_signed_error_od"], -0.1)
        np.testing.assert_array_equal(labels, original)
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            mask_stress(rgb, labels, truth.iloc[:1])
