# PACT

**PACT: Deliverable-Value Learning for Credit Assignment in Segmentation Refinement**

Posterior-Acceptance Credit Transfer trains the proposer on utility retained
after acceptance or exact rollback. A frozen vision-language segmenter supplies
the anchor; PACT selects one trial, decodes it once, and evaluates signed added
and removed regions before delivery. Image-disjoint cross-fitting separates
acceptor fitting from proposal-target construction. Accepted losses retain
negative credit, and rollback receives zero delivery credit while still charging
the attempted decode.

This is the research code and numerical-data release. It does not claim a CVPR
acceptance or provide a pretrained foundation-model download.

## Quick start: reproduce the reported numerical results

Python 3.11+; CPU only, no model downloads or network access during verification.

```bash
git clone https://github.com/Charles-Tian077/PACT.git
cd PACT
python -m pip install -r requirements.txt
python experiments/reproduction/verify_results.py
python scripts/verify_training.py
python scripts/export_results.py
```

The first command checks every artifact SHA, recomputes all 57 Qwen seed rows
(19 arms), reconstructs 54 nonbaseline policies, and checks 21 InternVL seed rows
(7 arms). The training verifier checks the 16 saved fixed-feature heads and
their disjoint image roles without fitting or changing them. The export command
writes all means and seeds to `results/`; it never chooses the best seed.

## Main comparison

On the final 512-image / 1,024-query held-out Qwen cohort, PACT reduces the rate
of IoU losses exceeding 0.02 from **7.63% to 0.37%** compared with forced acceptance
of the **same trials**, and improves mean hard IoU by **2.41 percentage points**
in that matched comparison. Relative to the frozen SFT baseline, its mean
hard-IoU change is **+0.04 percentage points**, with presence decisions unchanged.
Image-paired intervals for the small overall gains include zero. These are
distinct comparisons; the result is not a claim of across-the-board SOTA.

The measured online benchmark uses 32 input-selected queries, three repetitions
and one H200 per arm: 10.548 s/query for SFT and 10.673 s/query for PACT (+1.2%).
Cached numerical replay and exhaustive candidate capture are not deployment
latency. InternVL uses a separate diagnostic cohort and a different SFT training
budget; compare increments within each backbone.

## Contents

| Path | Contents |
| --- | --- |
| `pact/` | Exact 39-D feature maps, pre-trial features, frozen readouts, budget allocation, ridge fitting and native one-trial execution |
| `backbones/qwen/` | Qwen/SAM model, datasets and original SFT training/evaluation entry points |
| `data/` | TRAIN feature arrays and the frozen matched heads (192 images / 384 queries) |
| `experiments/reproduction/` | Hashed Qwen/InternVL numerical evaluation data, frozen candidates, predictions and verifier |
| `configs/` | Frozen evaluation and native timing protocols |
| `results/` | Complete arm means and seed tables, generated from released results |
| `docs/` | Data setup, native integration, method and evidence boundaries |

`wide_PACT` is the adopted final primary. Historical status flags in the original
protocol record whether formal adoption had occurred **when that record was
sealed**; they are preserved for provenance. No released test outcomes are used
to fit or select the controller.

## Native inference and training

See [native execution](docs/NATIVE_EXECUTION.md) and [data](docs/DATA.md).
Full native reproduction additionally requires the exact SFT adapter/checkpoint,
backbone, SAM weights, processor and licensed images. Large checkpoints and raw
photographs are not bundled. The portable CPU path is verified; this packaging
change has not rerun the GPU experiments. Original model code also contains
historical policy variants; the final PACT path is `pact/online.py` with the
released `wide` changed-region experts, not those historical defaults.

## Reuse and provenance

See [third-party notices](THIRD_PARTY_NOTICES.md) and [reproduction scope](docs/REPRODUCIBILITY.md).
No blanket open-source license is asserted over third-party code, datasets or
weights. Public availability alone does not change their original terms.
The repository contains no remote access credentials or private server paths.
