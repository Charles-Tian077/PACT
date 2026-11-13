PACT: optional frozen-output verification materials

Scope
This directory is an optional companion to the supplementary PDF, not a
mandatory CVPR artifact. It verifies sealed numerical evidence without a
GPU, network access, fitting, model selection, or checkpoint downloads.

Run from this directory, using Python 3.11+ and the pinned NumPy version:
  python -m pip install -r requirements.txt
  python verify_results.py
Optional JSON output:
  python verify_results.py --output verification_report.json

What is checked
1. Every exported source/data SHA256.
2. Qwen: native valid-domain integer IoUs, all 19 arms and 57 seed metric
   rows, all arm means, and exact reconstruction of 54 nonbaseline
   policies from the frozen proposal/acceptor readouts and shared budget.
3. InternVL3.5-4B: 7 arms and 21 saved seed rows for 13 directly computable
   metric fields. This compact export lacks the integer intersections/
   unions needed to recompute InternVL cIoU, and lacks its complete
   candidate-feature tensors needed to reconstruct proposal selection.

Evidence boundaries
The current primary is wide_PACT. Do not select the best value in each
column from different arms, seeds or cohorts. Positive observed seed
means do not establish a population improvement. The original intervals
are retained; this quick verifier does not rerun the 2,000 bootstraps.
Qwen's primary held-out cohort and InternVL's consumed diagnostic cohort
are separate. Their SFT training budgets differ.

Online timing is retained as actual native execution measurements.
This CPU audit is not a timing benchmark and does not recreate those
measurements. Exhaustive capture and cached-array replay are not normal
deployment latency.

No raw photographs, full GT masks, native logits/mask tensor caches,
foundation checkpoints, private paths, remote credentials or author
identities are included. Numeric sample/image identifiers are SHA256
strings. Full native inference and end-to-end training require the
originally licensed datasets, backbones, adapter and execution sources;
this directory does not claim one-command end-to-end reproduction.

Review-use source export. No new open-source license is asserted here.
All original failed runs and execution revisions remain in the internal
research records; this export neither changes nor replaces them.


The public release additionally includes native integration sources and TRAIN
feature arrays. The CPU result verifier remains identical to the audited
reviewer export. `release_manifest.json` binds the additional release files.
