"""Generate a synthetic brightfield IHC field with known compartment signals.

Coordinates, cell masks, textures and stain concentrations are generated here.
The generator uses analytic ellipses and does not call the measurement code.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def _texture(rng, shape, sigma):
    """Return a smooth, zero-centered field of approximately unit variance."""
    values = gaussian_filter(rng.normal(size=shape), sigma=sigma)
    return (values - values.mean()) / max(values.std(), 1e-8)


def build_phantom(seed=17):
    """Return RGB uint8, integer nucleus labels and per-cell nominal DAB OD.

    Three gland-like arrangements, peripheral cells and two crowded pairs give
    the estimator open space, clipped objects and competing neighborhoods.
    Every phenotype occurs in each gland. Each nucleus has elliptical contours
    and a perinuclear stain band extending to 1.6 times its axes. Where cells
    compete, the smallest normalized elliptical distance owns the pixel.

    Targets describe the generating concentrations inside the two compartments.
    Texture, antialiasing, 8-bit encoding, and geometric mismatch with estimated
    rings make the recovered means approximate. These are arbitrary synthetic
    stain units and are not calibrated against tissue or patient scores.
    """
    rng = np.random.default_rng(seed)
    height, width = 470, 560
    shape = (height, width)
    y, x = np.indices(shape, dtype=float)
    glands = [
        # Center x/y, major/minor arrangement axes, rotation and cell count.
        (136, 135, 87, 49, -0.33, 14),
        (398, 131, 84, 51, 0.28, 14),
        (275, 334, 105, 65, -0.16, 17),
    ]
    geometries = []
    lumen = np.zeros(shape)
    for cx, cy, a, b, angle, count in glands:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        u = (x - cx) * cos_a + (y - cy) * sin_a
        v = -(x - cx) * sin_a + (y - cy) * cos_a
        radial_distance = np.hypot(u / a, v / b)
        # A soft luminal boundary lets the faint stromal texture show through.
        lumen = np.maximum(lumen, np.clip((0.75 - radial_distance) / 0.23, 0, 1))
        for i in range(count):
            t = (2 * np.pi * i / count) + rng.uniform(-0.033, 0.033)
            u_cell = a * np.cos(t) + rng.normal(0, 2)
            v_cell = b * np.sin(t) + rng.normal(0, 2)
            center_x = cx + u_cell * cos_a - v_cell * sin_a
            center_y = cy + u_cell * sin_a + v_cell * cos_a
            orientation = angle + np.arctan2(np.sin(t) / b, np.cos(t) / a)
            geometries.append((center_x, center_y, orientation + rng.normal(0, 0.2)))

    # The first and last of these cells cross an image boundary. Pairs near
    # (65, 282) and (472, 337) deliberately compete for some ring pixels.
    geometries.extend([
        (3, 271, 0.61), (55, 279, 1.05), (74, 286, 1.25),
        (470, 332, -0.75), (490, 340, -0.90), (515, 252, 1.51),
        (215, 220, -0.60), (335, 224, 0.83), (368, 422, -0.87),
        (120, 420, 1.00), (32, 63, 0.53), (518, 403, -0.24),
        (444, 465, 1.35),
    ])
    phenotypes = ("nuclear-enriched", "balanced", "ring-enriched")
    # Repeated short blocks give each field locally coherent, mixed patterns.
    target_pairs = ((0.72, 0.12), (0.32, 0.32), (0.12, 0.70))
    distance = np.full(shape, np.inf)
    owner = np.zeros(shape, dtype=np.int32)
    nuclear_targets = [0.0]
    ring_targets = [0.0]
    truth_rows = []

    for index, (cx, cy, angle) in enumerate(geometries):
        cell_id = index + 1
        major = rng.uniform(13.5, 18.0)
        minor = rng.uniform(8.0, 11.5)
        dx, dy = x - cx, y - cy
        u = dx * np.cos(angle) + dy * np.sin(angle)
        v = -dx * np.sin(angle) + dy * np.cos(angle)
        radius = np.hypot(u / major, v / minor)
        nearer = radius < distance
        distance[nearer] = radius[nearer]
        owner[nearer] = cell_id
        group = (index // 2) % len(phenotypes)
        scale = rng.uniform(0.86, 1.16)
        nuclear, ring = np.asarray(target_pairs[group]) * scale
        nuclear_targets.append(nuclear)
        ring_targets.append(ring)
        truth_rows.append((cell_id, phenotypes[group], nuclear, ring))

    labels = np.where(distance <= 1.0, owner, 0).astype(np.int32)
    nuclei = np.clip((1.035 - distance) / 0.07, 0, 1)
    cytoplasm = np.clip((1.64 - distance) / 0.08, 0, 1)
    rings = cytoplasm - nuclei
    granular = _texture(rng, shape, 0.85)
    mottled = _texture(rng, shape, 2.8)
    fibers = _texture(rng, shape, (1.9, 12.0))
    fibers += 0.55 * np.sin(0.18 * y + 0.025 * x + 2 * np.sin(x / 41))
    stroma = 1 - 0.85 * lumen

    hema = stroma * (0.093 + 0.029 * fibers + 0.013 * mottled)
    hema += nuclei * (0.58 + 0.078 * granular + 0.11 * mottled)
    hema += rings * (0.074 + 0.022 * mottled)
    ambient_dab = stroma * (0.038 + 0.010 * fibers)
    local_grain = np.clip(1 + 0.056 * granular + 0.05 * mottled, 0.70, 1.30)
    dab = ambient_dab * (1 - cytoplasm)
    dab += nuclei * np.asarray(nuclear_targets)[owner] * local_grain
    dab += rings * np.asarray(ring_targets)[owner] * local_grain

    # An explicit two-stain forward Beer–Lambert model renders the RGB image.
    # Keeping these reference vectors local makes the phantom independent of
    # adjustments to the estimator. Each vector has unit Euclidean norm.
    basis = np.array([[0.650, 0.268], [0.704, 0.570], [0.286, 0.776]])
    basis /= np.linalg.norm(basis, axis=0)
    concentrations = np.stack([np.maximum(hema, 0), np.maximum(dab, 0)], axis=-1)
    rgb = np.rint(256 * np.exp(-(concentrations @ basis.T)) - 1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    truth = pd.DataFrame(truth_rows, columns=[
        "cell_id", "phenotype", "target_nuclear_dab_od", "target_ring_dab_od",
    ])
    return rgb, labels, truth
