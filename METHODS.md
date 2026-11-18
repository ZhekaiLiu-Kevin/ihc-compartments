# How the measurements are made

## Image and controls

The synthetic field uses oriented ellipses, three gland-like arrangements, clipped cells and crowded neighbors. Each nucleus has a positive instance ID. Three intended signal patterns cover nuclear enrichment, perinuclear enrichment and balanced staining. The generator draws a band whose outer ellipse has 1.6 times the nuclear axes. It adds spatial texture, smooths the transition at compartment boundaries and encodes the mixture as an 8-bit stained RGB image. Each row in `targets.csv` contains the nominal concentration assigned to one cell before rendering.

The generator and estimator use different ring geometry and the same two stain vectors. The target concentrations are hidden from the estimator. This design tests arithmetic, pixel ownership and measurement bias under a known image formation model. It is an intentionally favorable test of color separation. Changes in illumination, scanner calibration and stain composition remain untested.

## From color to a stain estimate

For an 8-bit channel intensity $I$, the optical-density transform is $-\ln((I+1)/256)$. The offset keeps black pixels finite and maps white to zero. The RGB absorbance vectors for hematoxylin and DAB are $(0.650,0.704,0.286)$ and $(0.268,0.570,0.776)$, respectively. Each vector is normalized to unit length. A nonnegative two-component least-squares fit yields a DAB coefficient for each pixel. Units depend on the basis and white reference. Color deconvolution follows the principle described by [Ruifrok and Johnston (2001)](https://pubmed.ncbi.nlm.nih.gov/11531144/).

## Assigning a ring to a nucleus

For a nucleus of area $A$ pixels, let $R=\sqrt{A/\pi}$. With outer scale $f$, the external band reaches $(f-1)R$ from the nearest nuclear pixel. At the default $f=1.5$, the band is half an equivalent nuclear radius wide. For an irregular nucleus the band follows its outline. The full supplied nucleus mask defines the nuclear compartment in this implementation.

An exterior pixel must be within the allowed width of a globally nearest nucleus. When several equally near nuclei qualify, the smallest original ID wins. Every nuclear pixel is excluded from the rings. With a supplied tissue mask, only eligible exterior pixels contribute to a ring. The perinuclear compartment is a geometric sampling region. The image contains no measured cell-membrane boundaries.

The numerator and denominator of the ring-to-nucleus ratio are mean DAB coefficients, each averaged over its own pixels. This makes the numerator and denominator comparable when the sampled areas differ. Ratios with an empty ring or a nuclear mean at or below 0.01 are undefined and exported as blank CSV values.

## Quality flags

All measured cells remain in the table so the reviewer can choose an exclusion policy. A cell can receive several flags. Measurements on partially supported rings may still be useful to inspect.

| Flag | What to check |
|---|---|
| `edge` | The nucleus touches the field border or the ideal ring extends beyond the field. |
| `crowded` | Neighboring nuclei or their competing rings reduce the in-frame ring. |
| `tissue_clipped` | The supplied tissue mask removes ring pixels. |
| `empty_ring` | No eligible ring pixels remain. The ring mean and ratio are missing. |
| `small_ring` | Fewer than 20 ring pixels remain. |
| `low_ring_coverage` | Observed ring area is less than 25% of the circular approximation. |
| `low_nuclear_signal` | Nuclear mean DAB is at or below 0.01 OD. The ratio is missing. |

`ring_area_fraction` divides observed ring area by $A(f^2-1)$, the expected area of a circular annulus. Irregular nuclei and pixel discretization can produce values above one. A flag describes the measurement geometry or numerical support and does not identify a cell as tumor or background.

## Two separate sensitivity experiments

1. **Ring width.** Keep the RGB field and nucleus mask constant. Measure each cell with outer scales of 1.25, 1.50, 1.75 and 2.00. Group summaries use the median and interquartile range of defined ratios. This isolates a choice made by the analyst.
2. **Nuclear boundary error.** Keep the RGB field and assigned targets constant. Erode or dilate each instance by a two-pixel Euclidean disk, then rebuild the ring at scale 1.50. Every original target stays in the denominator. Report finite measurements, mean absolute error, mean signed error and rank correlation. A positive signed error means the measured coefficient exceeds the target. This isolates an error in the supplied segmentation.

The five seeds vary geometry, texture and stain levels within the same simulation design. They are repeated software checks, not independent biological specimens. The benchmark estimates neither diagnostic accuracy nor the fraction of tumor cells staining positive.
