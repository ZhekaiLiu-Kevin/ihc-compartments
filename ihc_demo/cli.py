"""Generate the synthetic study or measure an RGB field with instance labels."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .analysis import DEFAULT_SEEDS, recovery, ring_sweep
from .measurement import quantify
from .phantom import build_phantom


def save_measurements(out, cells, rings, dab_od):
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "cells.csv", index=False, float_format="%.8g")
    Image.fromarray(rings.astype(np.int32)).save(out / "rings.tif", compression="tiff_deflate")
    Image.fromarray(dab_od.astype(np.float32)).save(out / "dab_od.tif", compression="tiff_deflate")


def run_demo(out, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .explorer import write_explorer
    from .figures import overview_figure, qc_figure

    rgb, labels, truth = build_phantom(seed)
    cells, rings, dab_od = quantify(rgb, labels)
    sweep = ring_sweep(rgb, labels, truth, cells)
    save_measurements(out, cells, rings, dab_od)
    Image.fromarray(rgb).save(out / "field.png")
    Image.fromarray(labels.astype(np.int32)).save(out / "nuclei.tif", compression="tiff_deflate")
    truth.to_csv(out / "targets.csv", index=False, float_format="%.8g")
    sweep.to_csv(out / "sensitivity.csv", index=False, float_format="%.8g")
    recovery(cells, truth).to_csv(out / "recovery.csv", index=False, float_format="%.8g")
    for name, fig in (
        ("overview", overview_figure(rgb, labels, rings, dab_od, cells)),
        ("validation", qc_figure(cells, truth, sweep)),
    ):
        fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    write_explorer(out / "explorer.html", rgb, labels, rings, dab_od, cells, truth)
    print(f"Measured {len(cells)} synthetic cells. Open {out / 'explorer.html'}")


def run_benchmark(out):
    out.mkdir(parents=True, exist_ok=True)
    from .stress import mask_stress
    records = []
    stress_records = []
    for seed in DEFAULT_SEEDS:
        rgb, labels, truth = build_phantom(seed)
        cells, _, _ = quantify(rgb, labels)
        records.append(recovery(cells, truth).assign(seed=seed))
        stress_records.append(mask_stress(rgb, labels, truth).assign(seed=seed))
    result = pd.concat(records, ignore_index=True)
    pd.concat(stress_records, ignore_index=True).to_csv(
        out / "mask_stress.csv", index=False, float_format="%.8g"
    )
    result.to_csv(out / "benchmark.csv", index=False, float_format="%.8g")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def read_array(path):
    with Image.open(path) as image:
        return np.array(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Build a synthetic field, figures and offline explorer.")
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--out", type=Path, default=Path("runs/demo"))
    benchmark = commands.add_parser("benchmark", help="Run the fixed five-seed recovery benchmark.")
    benchmark.add_argument("--out", type=Path, default=Path("runs/benchmark"))
    measure = commands.add_parser("measure", help="Measure an 8-bit RGB image and aligned integer label mask.")
    measure.add_argument("--rgb", type=Path, required=True)
    measure.add_argument("--labels", type=Path, required=True)
    measure.add_argument("--tissue-mask", type=Path, help="Optional aligned single-channel mask, nonzero=tissue.")
    measure.add_argument("--outer-scale", type=float, default=1.5)
    measure.add_argument("--out", type=Path, default=Path("runs/measure"))
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.out, args.seed)
    elif args.command == "benchmark":
        run_benchmark(args.out)
    else:
        try:
            tissue = read_array(args.tissue_mask) != 0 if args.tissue_mask else None
            arrays = quantify(read_array(args.rgb), read_array(args.labels),
                              outer_scale=args.outer_scale, tissue_mask=tissue)
        except (OSError, ValueError, TypeError) as err:
            parser.error(str(err))
        save_measurements(args.out, *arrays)
        print(f"Measured {len(arrays[0])} cells. Saved results in {args.out}")


if __name__ == "__main__":
    main()
