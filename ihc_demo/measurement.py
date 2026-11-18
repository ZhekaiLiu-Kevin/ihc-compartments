"""Measure hematoxylin–DAB images using labeled nuclei and exclusive outer rings.

The API operates on one RGB field of view. Distances are in pixels and assume
square pixels. A positive label identifies one pre-segmented nucleus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage


# Columns are hematoxylin and DAB absorbance vectors in RGB optical-density space.
HDAB_BASIS = np.array([[0.650, 0.268], [0.704, 0.570], [0.286, 0.776]])
HDAB_BASIS /= np.linalg.norm(HDAB_BASIS, axis=0)
HDAB_BASIS.setflags(write=False)
_STAIN_INVERSE = np.linalg.pinv(HDAB_BASIS)
RATIO_FLOOR_OD = 0.01
SMALL_RING_PX = 20
MIN_RING_FRACTION = 0.25


def deconvolve_dab(rgb: np.ndarray) -> np.ndarray:
    """Return the DAB coefficient of a nonnegative two-stain least-squares fit.

    White is the reference intensity. OD = -ln((RGB + 1) / 256). In the cone
    spanned by the two unit stain vectors, the unconstrained solution is valid.
    Outside the cone the nearest fit lies on one of its two boundary rays.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an H × W × 3 uint8 image")
    if not rgb.shape[0] or not rgb.shape[1]:
        raise ValueError("rgb must contain at least one pixel")

    optical_density = -np.log((rgb.astype(np.float64) + 1) / 256)
    coefficients = optical_density @ _STAIN_INVERSE.T
    hematoxylin, dab = coefficients[..., 0], coefficients[..., 1]
    dab_only = optical_density @ HDAB_BASIS[:, 1]
    # If the hematoxylin weight would be negative, refit along the DAB ray.
    # If the DAB weight would be negative, the DAB coefficient is zero.
    return np.where(hematoxylin < 0, dab_only, np.maximum(0.0, dab))


def _ring_geometry(labels, cell_ids, outer_scale):
    """Build a disjoint ring map and per-cell QC counts without all-pairs arrays."""
    height, width = labels.shape
    foreground = labels > 0
    compact = np.zeros(labels.shape, dtype=np.int32)
    compact[foreground] = np.searchsorted(cell_ids, labels[foreground]) + 1
    areas = np.bincount(compact.ravel(), minlength=len(cell_ids) + 1)[1:]
    widths = (outer_scale - 1) * np.sqrt(areas / np.pi)

    nearest_distance = ndimage.distance_transform_edt(~foreground)
    rings = np.zeros_like(compact)
    ideal_ring_areas = np.zeros(len(cell_ids), dtype=np.int64)
    edge = np.zeros(len(cell_ids), dtype=bool)
    # Sorted labels supply a reproducible tie break for equally close nuclei.
    for index, (bounds, thickness) in enumerate(
        zip(ndimage.find_objects(compact), widths)
    ):
        nucleus_id = index + 1
        rows, cols = bounds
        margin = min(rows.start, cols.start, height - rows.stop, width - cols.stop)
        edge[index] = margin == 0 or thickness >= margin + 1
        pad = int(np.ceil(min(max(height, width), thickness)))
        roi = (
            slice(max(0, rows.start - pad), min(height, rows.stop + pad)),
            slice(max(0, cols.start - pad), min(width, cols.stop + pad)),
        )
        cell_mask = compact[roi] == nucleus_id
        distance = ndimage.distance_transform_edt(~cell_mask)
        ideal_ring = (~cell_mask) & (distance <= thickness)
        ideal_ring_areas[index] = np.count_nonzero(ideal_ring)
        local_rings = rings[roi]
        owned = ideal_ring & (distance == nearest_distance[roi]) & (local_rings == 0)
        local_rings[owned] = nucleus_id

    return compact, rings, areas, ideal_ring_areas, edge


def quantify(
    rgb: np.ndarray,
    labels: np.ndarray,
    *,
    outer_scale: float = 1.5,
    tissue_mask: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return cell measurements, a labeled ring image, and the DAB OD image.

    For a nucleus of area A, the ring reaches (outer_scale − 1) × sqrt(A/π)
    beyond its nearest nuclear pixel (center-to-center Euclidean distance).
    All nuclei are excluded. An outside pixel belongs only to a globally
    nearest nucleus within that nucleus's radius. If multiple nearest nuclei
    qualify, the smallest original label wins. Zero in the ring image means
    unassigned. Input labels may be sparse and must fit a signed int32.

    QC flags retain measurements. 'edge' indicates the nucleus touches the
    border or its ideal ring extends past it; 'crowded' indicates in-frame
    ideal ring pixels lost to neighbors; 'empty_ring' means no pixels remain.
    Ratios are missing for empty rings or nuclear mean DAB ≤ 0.01 OD. This
    denominator floor is a numerical reporting choice for the demonstration.

    An optional Boolean tissue mask further restricts ring pixels, while
    nuclear means use each complete supplied nucleus. 'tissue_clipped' marks
    any loss at this mask. 'small_ring' marks fewer than 20 ring pixels and
    'low_ring_coverage' marks less than 25% of the theoretical *circular* ring
    area A × (outer_scale² − 1). This proxy can exceed one for irregular nuclei;
    it is undefined at scale 1. The cutoffs are illustrative QC settings.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a 2D integer nucleus-label image")
    if labels.shape != np.shape(rgb)[:2]:
        raise ValueError("labels and RGB image must have the same spatial shape")
    if np.any(labels < 0) or np.any(labels > np.iinfo(np.int32).max):
        raise ValueError("labels must be nonnegative and fit in int32")
    if not np.isfinite(outer_scale) or outer_scale < 1:
        raise ValueError("outer_scale must be finite and at least 1")

    if tissue_mask is not None:
        tissue_mask = np.asarray(tissue_mask)
        if tissue_mask.shape != labels.shape or tissue_mask.dtype != bool:
            raise ValueError("tissue_mask must be Boolean and match labels.shape")

    dab_od = deconvolve_dab(rgb)
    cell_ids = np.unique(labels[labels > 0]).astype(np.int32)
    if len(cell_ids):
        compact, ring_ids, areas, ideal_areas, edges = _ring_geometry(
            labels, cell_ids, outer_scale
        )
    else:
        compact = np.zeros(labels.shape, dtype=np.int32)
        ring_ids = np.zeros_like(compact)
        areas = np.empty(0, dtype=np.int64)
        ideal_areas = np.empty(0, dtype=np.int64)
        edges = np.empty(0, dtype=bool)

    length = len(cell_ids) + 1
    owned_counts = np.bincount(ring_ids.ravel(), minlength=length)[1:]
    if tissue_mask is not None:
        ring_ids[~tissue_mask] = 0
    counts = np.bincount(ring_ids.ravel(), minlength=length)[1:]
    expected_area = areas * (outer_scale**2 - 1)
    fractions = np.divide(
        counts, expected_area, out=np.full(len(cell_ids), np.nan),
        where=expected_area > 0,
    )

    def summed_signal(compartments):
        return np.bincount(
            compartments.ravel(), weights=dab_od.ravel(), minlength=length
        )[1:]

    nuclear_means = summed_signal(compact) / areas
    ring_means = np.divide(
        summed_signal(ring_ids), counts, out=np.full(len(cell_ids), np.nan),
        where=counts > 0,
    )
    ratios = np.divide(
        ring_means, nuclear_means, out=np.full(len(cell_ids), np.nan),
        where=(counts > 0) & (nuclear_means > RATIO_FLOOR_OD),
    )
    flags = []
    for i in range(len(cell_ids)):
        reasons = []
        if edges[i]:
            reasons.append("edge")
        if owned_counts[i] < ideal_areas[i]:
            reasons.append("crowded")
        if counts[i] < owned_counts[i]:
            reasons.append("tissue_clipped")
        if counts[i] == 0:
            reasons.append("empty_ring")
        if counts[i] < SMALL_RING_PX:
            reasons.append("small_ring")
        if fractions[i] < MIN_RING_FRACTION:
            reasons.append("low_ring_coverage")
        if nuclear_means[i] <= RATIO_FLOOR_OD:
            reasons.append("low_nuclear_signal")
        flags.append("|".join(reasons) if reasons else "ok")

    cells = pd.DataFrame({
        "cell_id": cell_ids,
        "nuclear_area_px": areas,
        "ring_area_px": counts,
        "expected_ring_area_px": expected_area,
        "ring_area_fraction": fractions,
        "nuclear_mean_dab_od": nuclear_means,
        "ring_mean_dab_od": ring_means,
        "ring_to_nucleus_ratio": ratios,
        "qc_flags": pd.Series(flags, dtype="object"),
    })
    lookup = np.concatenate((np.zeros(1, dtype=np.int32), cell_ids))
    return cells, lookup[ring_ids], dab_od
