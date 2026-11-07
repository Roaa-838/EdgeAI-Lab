# Research integrity statement

This repository is a benchmark framework. The included `smoke-test` creates
random images and checks data-processing and model execution functionality.
It is **not** training, a model evaluation, or a quantified performance result.

No measured top-1 accuracy, latency, energy efficiency, RAM or compression
improvement is published until a real checkpoint is evaluated. If CPU INT8 is
unsupported on a system, report it as an unavailable experiment rather than
interpolating numbers or comparing two different checkpoints.

`models/`, `data/`, and generated `results/` are excluded from Git commits.
Publish only dataset source/provenance, non-sensitive aggregated results,
configurations, and code. Verify rights and privacy before publishing examples.

If adopting an earlier model from a team project, clearly state your own
contributions and the original model authors' roles. A verified experiment is
more useful to a graduate admissions committee than unsupported speedup claims.
