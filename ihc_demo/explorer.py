"""Export an offline, clickable view of a generated synthetic field."""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .measurement import quantify
from .stress import _perturb_instances


def _png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _runs(values):
    """Pack contiguous nonzero runs of cell IDs for browser hit-testing."""
    flat = values.ravel()
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    ends = np.r_[starts[1:], flat.size]
    keep = flat[starts] != 0
    return np.column_stack((starts[keep], (ends - starts)[keep], flat[starts][keep])).tolist()


def _inner_boundary(labels):
    padded = np.pad(labels, 1)
    return (labels > 0) & (
        (labels != padded[:-2, 1:-1])
        | (labels != padded[2:, 1:-1])
        | (labels != padded[1:-1, :-2])
        | (labels != padded[1:-1, 2:])
    )


def _finite_or_none(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _condition(labels, rings, cells):
    """One geometry plus its measured cell table. RGB and targets are shared."""
    owner = np.where(labels > 0, labels, rings)
    boundary = np.zeros((*labels.shape, 4), dtype=np.uint8)
    boundary[_inner_boundary(owner) & (rings > 0)] = (8, 127, 129, 255)
    boundary[_inner_boundary(labels)] = (229, 126, 14, 255)
    records = {}
    for row in cells.itertuples():
        cell_id = int(row.cell_id)
        ys, xs = np.nonzero(owner == cell_id)
        if not xs.size:
            raise ValueError(f"No labeled pixels for measured cell {cell_id}.")
        records[cell_id] = {
            "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "nuclear_area_px": int(row.nuclear_area_px),
            "ring_area_px": int(row.ring_area_px),
            "qc_flags": "" if pd.isna(row.qc_flags) else str(row.qc_flags),
            "nuclear_mean_dab_od": _finite_or_none(row.nuclear_mean_dab_od),
            "ring_mean_dab_od": _finite_or_none(row.ring_mean_dab_od),
            "ring_to_nucleus_ratio": _finite_or_none(row.ring_to_nucleus_ratio),
        }
    return {"edges": _png(boundary), "runs": _runs(owner), "cells": records}


def write_explorer(path, rgb, labels, rings, dab_od, cells, truth):
    """Write a standalone HTML view with reference, eroded, and dilated masks.

    Reference rings and measurements must use the default quantification settings.
    The experimental conditions use the same 2-pixel perturbations as mask_stress.
    """
    rgb, labels, rings, dab_od = map(np.asarray, (rgb, labels, rings, dab_od))
    if rgb.shape != (*labels.shape, 3) or rings.shape != labels.shape or dab_od.shape != labels.shape:
        raise ValueError("All field arrays must share the same spatial dimensions.")
    if cells.empty:
        raise ValueError("The synthetic explorer expects at least one measured cell.")
    if set(cells.cell_id) != set(truth.cell_id) or not truth.cell_id.is_unique:
        raise ValueError("Provide one target row per reference cell.")

    # White corresponds to clear background and darker gray to higher DAB OD.
    finite = dab_od[np.isfinite(dab_od)]
    upper = float(np.quantile(finite, 0.995)) if finite.size else 0.0
    upper = upper if upper > 0 else 1.0
    scaled = np.clip(np.nan_to_num(dab_od) / upper, 0, 1)
    dab_view = np.rint((1 - scaled) * 255).astype(np.uint8)

    conditions = {"reference": _condition(labels, rings, cells)}
    eroded, dilated = _perturb_instances(labels)
    for key, alternate in (("eroded_2px", eroded), ("dilated_2px", dilated)):
        measured, new_rings, _ = quantify(rgb, alternate)
        conditions[key] = _condition(alternate, new_rings, measured)

    targets = [
        {
            "id": int(row.cell_id),
            "phenotype": None if pd.isna(row.phenotype) else str(row.phenotype),
            "target_nuclear_dab_od": _finite_or_none(row.target_nuclear_dab_od),
            "target_ring_dab_od": _finite_or_none(row.target_ring_dab_od),
        }
        for row in truth.sort_values("cell_id").itertuples()
    ]
    height, width = labels.shape
    reference = conditions["reference"]["cells"]

    def center_distance(cell_id):
        x0, y0, x1, y1 = reference[cell_id]["box"]
        return ((x0 + x1) / 2 - width / 2) ** 2 + ((y0 + y1) / 2 - height / 2) ** 2

    payload = {
        "width": width, "height": height, "upper": upper,
        "first": min(reference, key=center_distance),
        "rgb": _png(rgb.astype(np.uint8)), "dab": _png(dab_view),
        "conditions": conditions, "targets": targets,
    }
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    encoded = encoded.replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PAGE.replace("__PAYLOAD__", encoded), encoding="utf-8")


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic IHC compartment explorer</title>
<style>
:root { color-scheme:light; --ink:#182d32; --quiet:#495e63; --line:#ccd6d5; --teal:#005d63;
  --paper:#f4f6f4; --panel:white; font-family:Inter,Arial,Helvetica,sans-serif; }
* {box-sizing:border-box} body {margin:0;color:var(--ink);background:var(--paper)}
main {max-width:1140px;margin:auto;padding:clamp(16px,3.2vw,40px)}
h1,h2,p,dl,dd {margin:0} h1 {font-size:clamp(1.8rem,3.2vw,2.9rem);font-weight:650;letter-spacing:-.05em}
header {margin-bottom:22px} header p,.hint {color:var(--quiet);line-height:1.5} header p {margin-top:8px}
.layout {display:grid;grid-template-columns:minmax(0,1.7fr) minmax(275px,1fr);gap:18px;align-items:start}
.panel {background:var(--panel);padding:clamp(15px,2vw,23px);border:1px solid var(--line);border-radius:12px}
h2 {font-size:1.15rem;letter-spacing:-.02em} .toolbar {display:flex;flex-wrap:wrap;gap:4px;margin:13px 0;
  padding:4px;background:#edf1ef;border-radius:8px} .mode {position:relative;flex:1;min-width:max-content;text-align:center}
.mode input {position:absolute;opacity:0} .mode span {display:block;border-radius:5px;padding:7px 9px;font-size:.87rem;
  line-height:1.3;cursor:pointer;color:#405256} .mode input:checked+span {background:var(--teal);color:white}
.mode input:focus-visible+span,select:focus-visible,button:focus-visible {outline:3px solid #ad6400;outline-offset:2px}
.field {position:relative;line-height:0;background:#edeae3;overflow:hidden;border:1px solid var(--line);border-radius:6px}
canvas {width:100%;height:auto;cursor:crosshair;touch-action:manipulation}
.legend {display:flex;align-items:center;flex-wrap:wrap;gap:8px 16px;margin-top:13px;color:var(--quiet);font-size:.8rem;min-height:20px}
.key {display:inline-flex;align-items:center;gap:6px}.swatch {display:inline-block;width:19px;border-top:3px solid}
.nuclear {border-color:#e57e0e}.ring {border-color:#087f81}.focus-key {width:11px;height:11px;border:2px solid #172e36}
.grad {width:56px;height:8px;background:linear-gradient(to right,white,#111);border:1px solid #526266}
.hint {margin-top:10px;font-size:.85rem} .field-head {display:flex;justify-content:space-between;align-items:baseline;gap:10px}
#dimensions {font-size:.79rem;color:var(--quiet);font-variant-numeric:tabular-nums}
.mask-control {margin:16px 0 0}.mask-control .toolbar {margin:5px 0 0}.mask-label {font-size:.83rem;font-weight:600}
.cell-toolbar {display:flex;align-items:center;gap:7px;margin:13px 0 5px}
button,select {font:inherit;color:var(--ink);border:1px solid #adbbba;border-radius:6px;background:white;height:38px}
select {flex:1;min-width:0;padding:0 8px;cursor:pointer} button {width:38px;cursor:pointer;font-size:1.1rem}
button:hover,select:hover {background:#edf4f2} button:disabled {opacity:.4;cursor:default}
.detail {margin:19px 0 0} dt {color:var(--quiet);font-size:.81rem;line-height:1.3}
dd {font-variant-numeric:tabular-nums;font-size:1rem;font-weight:600;line-height:1.3}
.measure {margin:18px 0;padding:14px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.measure #ratio {font-size:2.15rem;letter-spacing:-.04em;color:var(--teal)}
.comparison {display:flex;align-items:baseline;gap:8px;margin-top:4px}.comparison dd {font-size:.87rem}
.row {display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-top:10px}
.row dd {font-size:.93rem;text-align:right}.row dt {max-width:70%}
details {margin-top:20px;border-top:1px solid var(--line);padding-top:13px}
summary {cursor:pointer;font-size:.9rem;font-weight:600}.subtle {color:var(--quiet);font-size:.82rem;line-height:1.45;margin-top:9px}
.sr {position:absolute;height:1px;width:1px;overflow:hidden;clip-path:inset(50%)}[hidden] {display:none!important}
@media(max-width:740px){main {max-width:610px}.layout {grid-template-columns:1fr}.panel {padding:16px}.legend {gap:7px 12px}}
</style></head><body><main>
<header><h1>Synthetic IHC compartment explorer</h1>
<p>Select a cell, then change its nuclear mask to see how a two-pixel boundary shift affects the staining measurement.</p></header>
<noscript>Enable JavaScript to use cell selection and image layers.</noscript>
<div class="layout"><section class="panel" aria-labelledby="field-title">
<div class="field-head"><h2 id="field-title">Image field</h2><span id="dimensions"></span></div>
<div class="toolbar" role="radiogroup" aria-label="Image layer">
<label class="mode"><input type="radio" name="layer" value="rgb"><span>Stained image</span></label>
<label class="mode"><input type="radio" name="layer" value="outline" checked><span>Compartments</span></label>
<label class="mode"><input type="radio" name="layer" value="dab"><span>DAB signal</span></label></div>
<div class="field"><canvas id="canvas" aria-label="Generated tissue field. Select a cell using the cell menu or by clicking its image."></canvas></div>
<div class="legend"><span class="key" id="nuc-key"><i class="swatch nuclear"></i>Nucleus</span>
<span class="key" id="ring-key"><i class="swatch ring"></i>Perinuclear ring</span>
<span class="key" id="scale-key" hidden><i class="grad"></i><span id="scale-text"></span></span>
<span class="key"><i class="focus-key"></i>Selected cell</span></div>
</section><section class="panel" aria-labelledby="cell-title"><h2 id="cell-title">Cell measurements</h2>
<div class="cell-toolbar"><button id="prev" aria-label="Previous cell" title="Previous cell">&#8592;</button>
<label class="sr" for="cell-select">Select cell</label><select id="cell-select"></select>
<button id="next" aria-label="Next cell" title="Next cell">&#8594;</button></div>
<div class="mask-control"><span class="mask-label" id="mask-label">Nuclear mask</span>
<div class="toolbar" role="radiogroup" aria-labelledby="mask-label">
<label class="mode"><input type="radio" name="mask" value="reference" checked><span>Reference</span></label>
<label class="mode"><input type="radio" name="mask" value="eroded_2px"><span>Erode 2 px</span></label>
<label class="mode"><input type="radio" name="mask" value="dilated_2px"><span>Dilate 2 px</span></label></div></div>
<dl class="measure"><div><dt>Perinuclear / nuclear mean DAB</dt><dd id="ratio">—</dd></div>
<div class="comparison"><dt>Change from reference</dt><dd id="delta">—</dd></div></dl>
<dl class="detail"><div class="row"><dt>Nuclear mean DAB (OD)</dt><dd id="nuc">—</dd></div>
<div class="row"><dt>Perinuclear mean DAB (OD)</dt><dd id="ring">—</dd></div>
<div class="row"><dt>Nuclear area</dt><dd id="na">—</dd></div>
<div class="row"><dt>Ring area</dt><dd id="ra">—</dd></div>
<div class="row"><dt>Quality flags</dt><dd id="flags">—</dd></div></dl>
<details open><summary>Generator targets</summary><dl>
<div class="row"><dt>Designed phenotype</dt><dd id="pheno">—</dd></div>
<div class="row"><dt>Nuclear DAB (OD)</dt><dd id="target-n">—</dd></div>
<div class="row"><dt>Perinuclear DAB (OD)</dt><dd id="target-r">—</dd></div></dl>
<p class="subtle">Targets are assigned before rendering. Stain texture, neighboring cells, and compartment geometry can shift measured values.</p></details>
<p class="hint" id="missing" hidden>A dash marks an unavailable measurement.</p>
<span id="status" class="sr" role="status" aria-live="polite"></span>
</section></div></main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const data = JSON.parse(document.getElementById("payload").textContent);
const get = (id) => document.getElementById(id);
const canvas = get("canvas"), ctx = canvas.getContext("2d"), select = get("cell-select");
const width = data.width, height = data.height;
canvas.width = width;
canvas.height = height;
const owner = new Uint32Array(width * height);
const byId = new Map(data.targets.map((cell, index) => [cell.id, index]));
for (const cell of data.targets) select.add(new Option("Cell " + cell.id, String(cell.id)));
get("dimensions").textContent = data.targets.length + " cells · " + width + " × " + height + " px";
get("scale-text").textContent = "0 to " + data.upper.toFixed(2) + " OD (99.5th percentile)";
let layer = "outline", mask = "reference", current = byId.get(data.first), images;
function updateHitMap() {
  owner.fill(0);
  for (const [start, size, id] of data.conditions[mask].runs) owner.fill(id, start, start + size);
}
function redraw() {
  if (!images) return;
  ctx.drawImage(layer === "dab" ? images.dab : images.rgb, 0, 0);
  if (layer === "outline") ctx.drawImage(images[mask], 0, 0);
  const cell = data.conditions[mask].cells[data.targets[current].id];
  if (!cell) return;
  const [x0, y0, x1, y1] = cell.box;
  const x = Math.max(2, x0 - 4), y = Math.max(2, y0 - 4);
  const right = Math.min(width - 2, x1 + 4), bottom = Math.min(height - 2, y1 + 4);
  const len = Math.min(9, (right - x) / 3, (bottom - y) / 3);
  ctx.beginPath();
  for (const [cx, cy, dx, dy] of [[x,y,1,1],[right,y,-1,1],[right,bottom,-1,-1],[x,bottom,1,-1]]) {
    ctx.moveTo(cx + dx * len, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * len);
  }
  ctx.lineWidth = 4; ctx.strokeStyle = "white"; ctx.stroke();
  ctx.lineWidth = 2; ctx.strokeStyle = "#172e36"; ctx.stroke();
}
function number(value, places) { return value == null ? "—" : value.toFixed(places); }
function deltaText(value, reference) {
  if (value == null || reference == null) return "—";
  const delta = Number((value - reference).toFixed(3));
  return (delta > 0 ? "+" : "") + delta.toFixed(3);
}
function showCell(index) {
  current = index;
  const target = data.targets[index];
  const cell = data.conditions[mask].cells[target.id];
  const reference = data.conditions.reference.cells[target.id];
  select.value = String(target.id);
  get("prev").disabled = index === 0;
  get("next").disabled = index === data.targets.length - 1;
  get("ratio").textContent = number(cell?.ring_to_nucleus_ratio, 2);
  get("delta").textContent = deltaText(cell?.ring_to_nucleus_ratio, reference.ring_to_nucleus_ratio);
  get("nuc").textContent = number(cell?.nuclear_mean_dab_od, 3);
  get("ring").textContent = number(cell?.ring_mean_dab_od, 3);
  get("na").textContent = cell ? cell.nuclear_area_px + " px" : "—";
  get("ra").textContent = cell ? cell.ring_area_px + " px" : "—";
  const flag = cell ? cell.qc_flags.trim() : "Cell absent from mask";
  get("flags").textContent = /^(ok|pass|none)?$/i.test(flag) ? "None" : flag.replace(/_/g, " ").split("|").join(", ");
  get("pheno").textContent = target.phenotype || "—";
  get("target-n").textContent = number(target.target_nuclear_dab_od, 3);
  get("target-r").textContent = number(target.target_ring_dab_od, 3);
  get("missing").hidden = cell?.ring_to_nucleus_ratio != null && cell?.ring_mean_dab_od != null;
  get("status").textContent = "Cell " + target.id + ". Ratio " + get("ratio").textContent + ". Change from reference " + get("delta").textContent + ".";
  redraw();
}
select.addEventListener("change", () => showCell(byId.get(Number(select.value))));
get("prev").addEventListener("click", () => { if (current > 0) showCell(current - 1); });
get("next").addEventListener("click", () => { if (current + 1 < data.targets.length) showCell(current + 1); });
for (const radio of document.querySelectorAll('input[name="layer"]')) {
  radio.addEventListener("change", () => {
    layer = radio.value;
    get("nuc-key").hidden = get("ring-key").hidden = layer !== "outline";
    get("scale-key").hidden = layer !== "dab";
    redraw();
  });
}
for (const radio of document.querySelectorAll('input[name="mask"]')) {
  radio.addEventListener("change", () => {
    mask = radio.value;
    updateHitMap();
    showCell(current);
  });
}
canvas.addEventListener("click", event => {
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((event.clientX - rect.left) * width / rect.width);
  const y = Math.floor((event.clientY - rect.top) * height / rect.height);
  if (x < 0 || x >= width || y < 0 || y >= height) return;
  const next = byId.get(owner[y * width + x]);
  if (next !== undefined) showCell(next);
});
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src;
  });
}
updateHitMap();
showCell(current);
const sources = {rgb:data.rgb, dab:data.dab};
for (const [name, item] of Object.entries(data.conditions)) sources[name] = item.edges;
Promise.all(Object.entries(sources).map(async ([name, source]) => [name, await loadImage(source)]))
.then(result => {
  images = Object.fromEntries(result); redraw();
}).catch(() => { get("dimensions").textContent = "Image rendering failed. Try another browser."; });
</script></body></html>
"""
