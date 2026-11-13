# Data and model assets

The repository distributes numerical research arrays, not the source photographs,
full ground-truth masks or foundation checkpoints. Image and query identifiers
are SHA256 strings. The released TRAIN features have 384 queries / 192 images;
Qwen final evaluation has 1,024 queries / 512 images, 384 negatives, 640 positives
and 638 valid positive masks. InternVL uses its own 512/512 diagnostic balance.

Obtain the original assets from their official projects and follow their terms:

- SESAME / false-premise referring segmentation: https://github.com/see-say-segment/sesame
- ReasonSeg / LISA: https://github.com/dvlab-research/LISA
- COCO images and annotations: https://cocodataset.org/
- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- Segment Anything: https://github.com/facebookresearch/segment-anything
- InternVL: https://github.com/OpenGVLab/InternVL

COCO photographs retain their original rights. Dataset terms and weight licenses
must be checked with the providers; this release does not relicense them.
The exact experiment SFT checkpoints and complete native caches are not included.
Numerical results are reproducible from the included arrays. Complete neural
training and raw-image inference additionally require those original assets.
