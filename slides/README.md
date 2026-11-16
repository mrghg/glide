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

The figures are in `slides/img/`. The animations are not in git, because they
are large and can be rebuilt. A deck still builds without them, with a gap
where the animation goes.

| File | In git | Used by | Source |
| --- | --- | --- | --- |
| `glide_feature.png` | yes | both title slides, overview | a GLIDE multi-site footprint map |
| `lpdm_schematic.svg` | yes | overview | hand-drawn schematic |
| `architecture.svg` | yes | overview | hand-drawn schematic |
| `hiding-io.svg` | yes | lightning | the lower half of `architecture.svg` (same file, cropped `viewBox`) |
| `zarr-chunks.svg` | yes | lightning | schematic of reading Zarr chunks |
| `footprints-terrain-small.mp4` | no | overview | `scripts/animate_footprints.py`, re-encoded at 640 pixels high |
| `footprints-terrain-20s.mp4` | no | lightning | the first 20 seconds of the same animation |

To shrink an animation for a slide (ffmpeg ships with the `imageio-ffmpeg`
package in the project environment):

```bash
ffmpeg -i in.mp4 -vf "scale=-2:640" -an -c:v libx264 -preset veryslow \
  -crf 32 -pix_fmt yuv420p -movflags +faststart out.mp4
```
