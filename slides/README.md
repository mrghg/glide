# Slides

Two Quarto (reveal.js) decks that share one theme and title template.

| File | What it is |
| --- | --- |
| `glide-overview.qmd` | The full overview talk |
| `glide-lightning.qmd` | A short talk: motivation, PyTorch and Zarr, current state, validation plan |
| `custom.scss` | The shared theme |
| `title-slide.html` | The shared title slide; its byline comes from `byline:` in each deck's header |

## Building

```bash
quarto render slides/glide-overview.qmd
quarto render slides/glide-lightning.qmd
```

The lightning deck renders to a single self-contained HTML file. The rendered
files are not committed.

## Figures and animations

The decks read their figures from `slides/img/`, which is not in git. A deck
still builds without them, with gaps where the figures go. The files needed:

| File | Used by | Source |
| --- | --- | --- |
| `glide_feature.png` | both title slides, overview | a GLIDE multi-site footprint map |
| `lpdm_schematic.svg` | overview | hand-drawn schematic |
| `architecture.svg` | overview | hand-drawn schematic |
| `hiding-io.svg` | lightning | the lower half of `architecture.svg` (same file, cropped `viewBox`) |
| `zarr-chunks.svg` | lightning | schematic of reading Zarr chunks |
| `footprints-terrain-small.mp4` | overview | `scripts/animate_footprints.py`, re-encoded at 640 pixels high |
| `footprints-terrain-20s.mp4` | lightning | the first 20 seconds of the same animation |

To shrink an animation for a slide (ffmpeg ships with the `imageio-ffmpeg`
package in the project environment):

```bash
ffmpeg -i in.mp4 -vf "scale=-2:640" -an -c:v libx264 -preset veryslow \
  -crf 32 -pix_fmt yuv420p -movflags +faststart out.mp4
```
