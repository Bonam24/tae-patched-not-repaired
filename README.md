# Patched, Not Repaired — artifact

Probes, evaluation code, per-epoch traces and benchmark specification for

> **Patched, Not Repaired: A Low Benign False Positive Rate Certifies the Region,
> Not the Model.** Bonam Osene and Denis Mugendi. TAE (Trust-AI-Eval): Can We Trust
> AI Evaluation?, NeurIPS 2026 workshop, Sydney. Non-archival poster.

The paper's claim: a low benign false-positive rate after hard-negative correction
establishes that the evaluated manifestations of a failure have been suppressed, not
that the failure mode has been removed — the two are separated by the part of input
space the evaluation does not reach.

**What is here:** the probe implementations, the evaluation/scoring harness, the
per-epoch probe traces, and the benchmark specification with per-image SHA-256
checksums. **What is not:** the training corpus, the model checkpoints, and the
image pixels themselves (licensing and privacy; see `benchmark_spec/SPECIFICATION.md`
— the protocol is fully reproducible from the specification, the exact pixels are
not; texture families are available on request for non-commercial research use).

## Layout

```
probes/
  generate_noise_probe.py        deterministic generator for the 15 bandpass stimuli (§5, §7)
  noise_sweep_setup.py           frequency sweep, baseline + corrected (§5); defines the
                                 shared model/scoring helpers for the three scripts below
  noise_controls.py              blank grey / flat patch / white field / gradient controls,
                                 all at most 0.011 (§5)
  patch_size_control.py          the 80/160/320 px control that refutes the spectral
                                 account (Appendix E)
  erasure_localized_and_families.py
                                 localized erasure at the firing region vs a displaced
                                 control, four operators (§4, Table 4), and the
                                 cross-family erasure signature (Appendix C)
  noise_erasure_signature.py     global erasure on the noise probe, four models — the
                                 0.779 permutation drop (§7, Appendix C)
evaluation/
  benchmark_four_models.py       the harness behind Table 3: six-family 280-image frozen
                                 benchmark + the 49 later fur images + the 52 dry grass
                                 images, four models, both criteria (§6, §7.1, §9)
  cardd_six_checkpoints.py       CarDD zero-shot re-evaluation, six checkpoints, one code
                                 path, fur arm vs matched control (Appendix B, Table 5)
  probe_traces_per_epoch.py      the 59-image fur probe and the noise probe on every
                                 epoch of the reported lineage — the trace behind
                                 Table 2 and §8 (fur returning at epochs 5–6)
  fur_collections_settled.py     frozen-40 vs later-49 fur under both criteria, with a
                                 gravel anchor assertion — the §6 "qualification on the
                                 fur family" (97.5% vs 46.9%)
  table1_seed3_control_probe.py  the seed 3 control at epoch 4, probed after training
                                 (the parenthetical in §7)
  drygrass_seeded_eval.py        dry grass over the eight epoch-4 checkpoints; produced
                                 traces/drygrass_seeded_per_image.csv (§7.1)
  drygrass_bootstrap.py          recomputes the Table 6 caption CIs from the shipped
                                 per-image trace: 10,000 paired resamples; asserts the
                                 published intervals reproduce
  recalibration_fit.py           the monotone-recalibration test of Appendix G, with a
                                 hard ep8 benchmark anchor before any comparison
  reported_arm_epochs_benchmark.py
                                 the reported arm's epochs 1-3 on the benchmark — the
                                 "as reported" row of the Appendix H table
training/                        the intervention itself. These require the unreleased
                                 training corpus and are included for transparency of
                                 the recipe, not as runnable artifacts.
  fur_finetune.py                the fur intervention (§7)
  control_arm.py                 the matched control, sampler draw count pinned (§7)
  seed_replication.py            seeds 1-3 x {fur, control}, with the per-source
                                 seam-term logging behind Appendix D
  seam_ablation.py               the beta=0 / alpha=0 ablation arms (Appendix H)
  corpus_facts.py                prints every Appendix F corpus figure
traces/
  probe_traces_summary_expanded.csv   per-epoch summary: arm x epoch x probe
  drygrass_seeded_per_image.csv       52 dry grass images x 8 arms (4 runs x 2)
  drygrass_seeded_summary.json        the §7.1 dry grass result with the
                                      pre-registered decision rule and Wilson CIs
benchmark_spec/
  SPECIFICATION.md               family definitions, collection protocol, preprocessing,
                                 the two firing criteria
  make_checksums.py              writes / verifies checksums/benchmark_sha256.txt
```

## Reproducing

Environment: PyTorch 2.13, `transformers==5.13.0`, OpenCV, pandas, pycocotools for
CarDD; a single 16 GB GPU suffices. The scripts are published exactly as run
(Kaggle/Azure paths and notebook `!pip` lines included) rather than cosmetically
cleaned: edit the path constants at the top of each script to your data layout.
`noise_controls.py`, `patch_size_control.py` and `erasure_localized_and_families.py`
expect the helpers defined by `noise_sweep_setup.py` in the same session, mirroring
the notebook they were run in. Scripts carry anchor assertions (e.g. gravel at ep8
must reproduce 0.480 firing / 0.256 area at tau = 0.85) that fail loudly if the
harness drifts, so a silent wrong-number reproduction is not possible.

The two firing criteria are not interchangeable; see
`benchmark_spec/SPECIFICATION.md` before comparing any rate to the paper.

## Licence

Code and specification: MIT (see `LICENSE`). The image sets are **not** covered by
this licence; their status is in `SPECIFICATION.md`.
