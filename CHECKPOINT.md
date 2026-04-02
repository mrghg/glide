# GLIDE checkpoint (2026-04-02)

- Scaffold complete: Dockerfile, deploy.sh, module stubs, README updates.
- Runtime device selection added: cuda -> mps -> cpu.
- Release classes implemented with device-aware tensor output in src/lpdm/release_generator.py.
- GPU engine utilities implemented in src/lpdm/gpu_engine.py (RK2, Langevin update, reflection, periodic diffusion helpers).
- Physics tests in test_physics.py now pass when run with project venv python.
- Visualization helpers added in src/lpdm/visualize.py.
- Notebook scaffold created: notebooks/visualization_starter.ipynb.

## Next recommended task
- Implement main ERA5 integration loop in src/lpdm/main.py using met_reader + gpu_engine + footprint_gridder.

## Known gotchas
- Running bare pytest may use wrong interpreter; prefer .venv python invocation.
- uv command was initially missing in shell; after install, ensure shell PATH is refreshed.
