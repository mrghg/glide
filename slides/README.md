# Slides

`glide-lightning.qmd` is a short talk on GLIDE, written in
[Quarto](https://quarto.org) as a reveal.js deck with its own theme
(`glide.scss`). Render it to one self-contained HTML file with:

```bash
cd slides
quarto render glide-lightning.qmd
```

The rendered `glide-lightning.html` is not committed.

The schematics in `img/` are drawn by the scripts in `tools/`, run from the
repository root:

| Image | Script |
| --- | --- |
| `img/footprint-concept.svg` | `tools/footprint_schematic.py` (a toy particle model, not GLIDE output) |
| `img/gpu-step.svg` | `tools/gpu_step_schematic.py` (numbers from `docs/architecture.md` §5) |
| `img/zarr-store-slide.svg` | `tools/zarr_store_schematic.py --no-pipeline` |

`img/pytorch-step.svg` and `img/zarr-chunks.svg` are hand-edited SVGs; the
first is a relabelled crop of the architecture diagram. The footprint video and
title map come from `scripts/animate_footprints.py` and `scripts/plot_feature.py`
run on the January 2024 ICOS run, then reduced in size to stay under the
repository's 1 MB file limit.
