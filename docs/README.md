# GLIDE documentation

Technical documentation for GLIDE. Start with the [project README](../README.md)
for installation and run instructions; these pages cover the physics and the
engineering in depth. For current project status see [../STATUS.md](../STATUS.md);
for the *why* behind the major design/physics choices see
[../dev/decisions/](../dev/decisions/).

## Contents

- **[architecture.md](architecture.md)** — how the code is built to get the most
  out of the hardware: the met-streaming pipeline, the device-gated per-step
  execution paths, CUDA-graph capture, and memory management.
- **[LPDM_physics_spec.md](LPDM_physics_spec.md)** — the backward-in-time LPDM
  core: governing equations, coordinate conventions, and verification spec.
- **[turbulence.md](turbulence.md)** — turbulence parameterisation: the modular
  scheme interface and the Hanna (1982) implementation.
- **[convection.md](convection.md)** — deep-convection scheme (reduced Emanuel).
- **[met_schema.md](met_schema.md)** — the meteorology input contract: variable
  names, units, coordinate conventions, and the pressure- vs model-level modes.
  Read this to prepare met from a non-ERA5 source.
- **[VALIDATION.md](VALIDATION.md)** — validation suite: scope, test tolerances
  and seeds, and which metrics are still pending external validation.

These are plain Markdown, rendered by GitHub as-is. They are structured so a
static documentation site (e.g. MkDocs) can be pointed at this directory later
without reorganisation.
