# Native execution

Install `backbones/qwen/requirements.txt` in a separate GPU environment. Add
`backbones/qwen` and the repository root to `PYTHONPATH`. Load the matching
frozen SFT checkpoint using `runtime_utils.load_bundle` and its saved model
configuration; do not substitute a newly trained adapter and call it the same
paper result. The architecture supports Qwen3-VL and SAM; the model config
default may be 8B, while the reported primary uses the saved 4B configuration.

The measured execution functions are extracted unchanged from the original
online worker except for portable imports. `pact.online.streamed_block` takes
the loaded model, processor, image/question rows and the frozen experts. Rows
contain `sample_key`, `image_path`, and `question`. Each identity must be unique;
the block length is divisible by four. The recorded benchmark fixes policy seed
42; use the frozen-array verifier for the separately recorded seeds 42, 43, 44.

```python
import json
from pact.online import streamed_block
frozen = json.load(open("experiments/reproduction/data/qwen_frozen_candidates.json"))
experts = frozen["candidates"]["wide"]["models"]["changed"]["experts"]
# model, processor and rows are supplied from the licensed local assets.
result = streamed_block(model, processor, rows, experts,
                        target="delivery", rep="changed", arm="wide_PACT")
```

The interface does not accept ground-truth masks or outcomes. The segmenter
performs one prefill, at most one additional trial decode, and exact fallback.
This is a component entry point, not a checkpoint-free runnable demo.

`backbones/qwen/train_sesame_qwen3vl.py --help` and the matching evaluation
entry point expose the original backbone/SFT workflows. They also retain
historical policies for provenance. Running their defaults does not recreate
the final PACT model. The final auxiliary fit uses the released TRAIN features,
image roles, support rule and `pact.fitting.fit`; `scripts/verify_training.py`
checks the resulting saved heads without initiating a new fit.
