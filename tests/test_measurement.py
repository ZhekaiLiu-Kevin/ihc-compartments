"""Tests based on analytic color mixtures and hand-drawn geometries."""

import unittest

import numpy as np
from scipy.optimize import nnls

from ihc_demo.measurement import HDAB_BASIS, deconvolve_dab, quantify


def render_stains(hematoxylin, dab):
    concentrations = np.stack(np.broadcast_arrays(hematoxylin, dab), axis=-1)
    optical_density = concentrations @ HDAB_BASIS.T
    return np.clip(np.rint(256 * np.exp(-optical_density) - 1), 0, 255).astype(np.uint8)


def image(shape):
    return np.full((*shape, 3), 128, dtype=np.uint8)


class ColorTests(unittest.TestCase):
    def test_recover_known_mixture_through_rgb_quantization(self):
        expected = np.array([[0.00, 0.16, 0.53, 1.05]])
        rgb = render_stains(np.array([[0.7, 0.3, 1.0, 0.1]]), expected)
        np.testing.assert_allclose(deconvolve_dab(rgb), expected, atol=0.015)

    def test_matches_independent_nnls_including_boundaries(self):
        rgb = np.array(
            [[[75, 240, 190], [248, 120, 51], [255, 255, 255], [30, 65, 39]]],
            dtype=np.uint8,
        )
        od = -np.log((rgb.astype(float) + 1) / 256)
        reference = [nnls(HDAB_BASIS, pixel)[0][1] for pixel in od[0]]
        np.testing.assert_allclose(deconvolve_dab(rgb)[0], reference, atol=1e-12)

    def test_known_per_cell_signals_and_ratio(self):
        labels = np.zeros((31, 31), dtype=np.int32)
        labels[10:21, 10:21] = 1
        dab = np.where(labels > 0, 0.28, 0.56)
        rgb = render_stains(np.full_like(dab, 0.3), dab)
        row = quantify(rgb, labels)[0].iloc[0]
        self.assertAlmostEqual(row.nuclear_mean_dab_od, 0.28, delta=0.01)
        self.assertAlmostEqual(row.ring_mean_dab_od, 0.56, delta=0.01)
        self.assertAlmostEqual(row.ring_to_nucleus_ratio, 2.0, delta=0.08)


class RingTests(unittest.TestCase):
    def test_radius_from_area_excludes_all_nuclear_pixels(self):
        labels = np.zeros((17, 17), dtype=np.int32)
        labels[6:10, 6:10] = 4321  # R≈2.26, width≈1.13 at scale 1.5
        cells, rings, _ = quantify(image(labels.shape), labels)
        expected = np.zeros_like(labels)
        expected[5, 6:10] = expected[10, 6:10] = 4321
        expected[6:10, 5] = expected[6:10, 10] = 4321
        np.testing.assert_array_equal(rings, expected)
        row = cells.iloc[0]
        self.assertEqual(rings.dtype, np.int32)
        self.assertEqual(row.nuclear_area_px, 16)
        self.assertEqual(row.ring_area_px, 16)
        self.assertEqual(row.expected_ring_area_px, 20)
        self.assertAlmostEqual(row.ring_area_fraction, 0.8)
        self.assertIn("small_ring", row.qc_flags)

    def test_crowded_tie_belongs_to_smaller_id(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[7:11, 3:7] = 9100
        labels[7:11, 8:12] = 24
        cells, rings, _ = quantify(image(labels.shape), labels, outer_scale=2)
        np.testing.assert_array_equal(rings[7:11, 7], [24] * 4)
        self.assertTrue(np.all(rings[labels > 0] == 0))
        self.assertTrue(all("crowded" in flags for flags in cells.qc_flags))
        self.assertEqual(cells.ring_area_px.sum(), np.count_nonzero(rings))

    def test_edge_and_no_ring(self):
        labels = np.full((7, 7), 100, dtype=np.int32)
        cells, rings, _ = quantify(image(labels.shape), labels)
        row = cells.iloc[0]
        self.assertEqual(row.ring_area_px, 0)
        self.assertEqual(np.count_nonzero(rings), 0)
        self.assertTrue(np.isnan(row.ring_mean_dab_od))
        self.assertTrue(np.isnan(row.ring_to_nucleus_ratio))
        self.assertIn("edge", row.qc_flags)
        self.assertIn("empty_ring", row.qc_flags)
        self.assertIn("low_ring_coverage", row.qc_flags)

    def test_white_image_has_undefined_ratio_and_zero_stain(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:10, 5:10] = 1
        rgb = np.full((*labels.shape, 3), 255, np.uint8)
        row = quantify(rgb, labels)[0].iloc[0]
        self.assertEqual(row.nuclear_mean_dab_od, 0)
        self.assertEqual(row.ring_mean_dab_od, 0)
        self.assertTrue(np.isnan(row.ring_to_nucleus_ratio))
        self.assertIn("low_nuclear_signal", row.qc_flags)

    def test_empty_mask_yields_empty_table_and_valid_dab(self):
        labels = np.zeros((8, 9), dtype=np.int32)
        cells, rings, dab = quantify(image(labels.shape), labels)
        self.assertTrue(cells.empty)
        self.assertIn("cell_id", cells.columns)
        self.assertFalse(rings.any())
        self.assertTrue(np.all(np.isfinite(dab)))

    def test_tissue_mask_trims_ring_and_preserves_full_nucleus(self):
        labels = np.zeros((30, 30), dtype=np.int32)
        labels[10:20, 10:20] = 19
        tissue = np.zeros(labels.shape, bool)
        tissue[8:10, 10:20] = True
        cells, rings, _ = quantify(image(labels.shape), labels, tissue_mask=tissue)
        row = cells.iloc[0]
        self.assertEqual(row.nuclear_area_px, 100)
        self.assertEqual(row.ring_area_px, 20)
        self.assertTrue(np.all(rings[~tissue] == 0))
        self.assertIn("tissue_clipped", row.qc_flags)
        self.assertIn("low_ring_coverage", row.qc_flags)
        self.assertNotIn("crowded", row.qc_flags)
        self.assertTrue(np.isfinite(row.ring_mean_dab_od))

    def test_ring_stops_at_nearest_owner_even_with_unequal_radii(self):
        labels = np.zeros((25, 25), dtype=np.uint16)
        labels[5:15, 2:13] = 5
        labels[10, 15] = 2
        _, rings, _ = quantify(image(labels.shape), labels)
        # Large nucleus reaches 2.96 px. Pixel (10, 14) is nearest to the
        # small nucleus, whose ring width is below one pixel, so stays unowned.
        self.assertEqual(rings[10, 14], 0)
        self.assertEqual(rings[10, 13], 5)

    def test_unit_scale_has_no_ring_and_no_defined_coverage(self):
        labels = np.zeros((12, 12), dtype=np.int32)
        labels[4:8, 4:8] = 2
        cells, rings, _ = quantify(image(labels.shape), labels, outer_scale=1)
        self.assertFalse(rings.any())
        self.assertTrue(np.isnan(cells.iloc[0].ring_area_fraction))

    def test_invalid_inputs(self):
        labels = np.zeros((4, 4), np.int32)
        rgb = image(labels.shape)
        cases = [
            (rgb, -np.ones_like(labels), {}),
            (rgb, labels.astype(float), {}),
            (rgb.astype(float), labels, {}),
            (rgb[:-1], labels, {}),
            (rgb, labels, {"outer_scale": 0.5}),
            (rgb, labels, {"outer_scale": np.nan}),
            (rgb, labels, {"tissue_mask": np.ones((3, 3), bool)}),
            (rgb, labels, {"tissue_mask": np.ones((4, 4), np.uint8)}),
        ]
        for color, mask, kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                quantify(color, mask, **kwargs)


if __name__ == "__main__":
    unittest.main()
