"""Sensitivity of compartment recovery to a controlled error in nuclear boundaries."""

import numpy as np
import pandas as pd
from scipy import ndimage

from .analysis import recovery
from .measurement import quantify


def _perturb_instances(labels):
    """Shrink or expand every instance by a disk of radius two pixel centers.

    Erosion treats adjacent labels and the exterior of the field as outside each
    nucleus. Dilation preserves foreground, assigning each new pixel to its
    nearest original nucleus. Exact distance ties go to the smaller label.
    """
    dy, dx = np.ogrid[-2:3, -2:3]
    squared_distance = dy**2 + dx**2
    disk = squared_distance <= 4
    local_min = ndimage.minimum_filter(labels, footprint=disk, mode="constant", cval=0)
    local_max = ndimage.maximum_filter(labels, footprint=disk, mode="constant", cval=0)
    eroded = np.where(local_min == local_max, labels, 0)

    # All positive input labels fit int32. Use a larger integer for absent
    # candidates so the minimum on each distance shell resolves ties by ID.
    absent = np.int64(np.iinfo(np.int32).max) + 1
    candidates = np.where(labels > 0, labels.astype(np.int64), absent)
    dilated = labels.copy()
    for distance_sq in (1, 2, 4):
        closest = ndimage.minimum_filter(
            candidates, footprint=squared_distance == distance_sq,
            mode="constant", cval=absent,
        )
        add = (dilated == 0) & (closest < absent)
        dilated[add] = closest[add]
    return eroded, dilated


def mask_stress(rgb, labels, truth):
    """Return synthetic recovery for three masks with identical RGB and targets.

    All target IDs remain in each denominator, including instances erased by
    erosion. Positive signed error means the measured concentration exceeds its
    synthetic target. This experiment introduces a known boundary perturbation;
    it does not estimate the accuracy of a segmentation model.
    """
    reference, _, _ = quantify(rgb, labels)
    target_ids = truth[["cell_id"]]
    ids = pd.Index(target_ids["cell_id"])
    if not ids.is_unique or set(ids) != set(reference["cell_id"]):
        raise ValueError("truth must contain exactly one row per input nucleus ID")

    eroded, dilated = _perturb_instances(np.asarray(labels))
    rows = []
    for condition, perturbed in (
        ("reference", None), ("eroded_2px", eroded), ("dilated_2px", dilated),
    ):
        cells = reference if perturbed is None else quantify(rgb, perturbed)[0]
        paired = target_ids.merge(cells, on="cell_id", how="left", validate="one_to_one")
        summary = recovery(paired, truth)
        summary.insert(0, "mask_condition", condition)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)
