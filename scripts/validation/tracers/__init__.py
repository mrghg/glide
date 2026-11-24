"""Tracer-experiment readers and scoring (queue task B10).

Readers for the DATEM ``emit``/``meas`` files and the JRC ETEX files, a
writer for DATEM-format predictions, a re-implementation of DATEM's
``statmain`` statistics and ranking score, the split into the published
per-release experiments, and the ATMES-II statistics for ETEX-1.

This package reads measurement files and other models' predictions only.
It never reads GLIDE output. See ``data/tracer-experiments.md`` §5.
"""
