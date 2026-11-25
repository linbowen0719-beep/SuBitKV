# Manuscript figures

These figures are reused from the author's
[ICLR2027_COMPRESS manuscript repository](https://github.com/linbowen0719-beep/ICLR2027_COMPRESS/tree/4c10265a526b76656654b0371958fdab00fc8a76),
commit `4c10265a526b76656654b0371958fdab00fc8a76`.

| Figure | Manuscript source | README asset |
| --- | --- | --- |
| Persistent base and truncatable refinement | `figs/method_overview.pdf` | [PNG](method_overview.png) · [Original PDF](method_overview.pdf) |
| Wan2.2 quality versus KV compression | `figs/lpips_pareto_v2.pdf` | [PNG](wan_quality_compression.png) · [Original PDF](wan_quality_compression.pdf) |
| Wan2.2 post-boundary temporal comparison | `figs/fig2.pdf` | [PNG](wan_temporal_comparison.png) · [Original PDF](wan_temporal_comparison.pdf) |

PDFs are copied without modification. PNGs render the complete first page at
1,800 pixels wide on an opaque background using PyMuPDF. No labels, plotted
values, or video frames have been changed. [provenance.json](provenance.json)
records the source revision, renderer version, and asset hashes.

The README additionally displays [Figure 2(a), PNG](wan_visual_comparison.png)
([cropped PDF](wan_visual_comparison.pdf)), derived from
[`figs/fig2.pdf` at commit 57916dc](https://github.com/linbowen0719-beep/ICLR2027_COMPRESS/blob/57916dcf21c1a5117dfa730b811c2c0090f5ddc8/figs/fig2.pdf).
This crop omits the right-hand panel (b), retaining all of panel (a), including
the video frames, error maps, color scale, labels, and caption. It is rendered
at 2,400 pixels wide. The `derived_figures` entry in
[provenance.json](provenance.json) records its crop bounds and hashes.

The README's selected Wan2.2 results are transcribed from
[`tables/5_table1.tex`](https://github.com/linbowen0719-beep/ICLR2027_COMPRESS/blob/4c10265a526b76656654b0371958fdab00fc8a76/tables/5_table1.tex).
That table reports quality over 973 samples and TWE over 10 videos per setting.
Its compression ratios include shared codebooks, metadata, and cache-related
workspace. These measurements belong to the manuscript's frozen evaluation
protocol. The five-stage example performs fresh calibration and records its own
results and accounting.

In the figures, SuBitKV's `Q` or “0.2-bit” label denotes the nominal residual
budget. The physical KV representation also includes the persistent base and
other storage costs. At `Q=0.2`, the reported 21.42× compression corresponds to
approximately `16 / 21.42 = 0.747` effective bits per original KV element.
The base remains available at `Q=0`.
