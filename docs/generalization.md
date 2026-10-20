# Generalization evaluation

## WHU-OPT-SAR unseen dataset

```bash
CHECKPOINT=./checkpoints/remotesam_plusplus.pth CUDA_VISIBLE_DEVICES=0 \
bash tasks/SEG.sh --modal sar --dataset whu --data-root /prepared/SegEarthOV2
```

This wraps the existing `tasks/code/_adapters/SEG_SAR.py` with:

- paper backbone/text encoder, resolution 896, 20 tokens, LoRA stages 0/1;
- SAR routing;
- the original prepared WHU crops and dataset configuration;
- per-class foreground probabilities followed by argmax;
- `--include_background`: all seven valid evaluation classes included;
- raw prepared label 0 ignored through the existing reduction;
- no class-score scaling/bias, smoothing, mode filtering or new prompts.

The seventh class is named `background` internally but corresponds to the valid others class, prompted as `others`. This is distinct from the ignored raw label 0. Inference size 896 and native evaluation-label size are separate settings; the existing `SEG_SAR.py` dataset `eval_size=224` is retained.

One-sample command for infrastructure smoke checks:

```bash
CHECKPOINT=./checkpoints/remotesam_plusplus.pth CUDA_VISIBLE_DEVICES=0 \
bash tasks/SEG.sh --modal sar --dataset whu --data-root /prepared/SegEarthOV2 \
  -- --limit_samples 1
```

One-sample results are not comparable to the manuscript's full seven-class mIoU. Retain the exact original crop/split artifacts for full evaluation. The repository does not contain a verified recreation of the original tiling pipeline.

## RGB unseen datasets and SATIN

The original Vaihingen, DeepGlobe and four-subset SATIN evaluators were recovered from dishimin and integrated behind the task-level SEG/MLC entry points. SATIN uses constituent train splits and micro label accuracy. Vaihingen predicts six classes and reports both a five-class mean excluding clutter and a six-class mean. DeepGlobe evaluates six classes on the labeled train folder, ignoring unknown pixels. These exact split and averaging conventions must be checked against the paper before claiming reproduction. See [commands](evaluation.md) and [source audit](recovered_sources.md).

See `docs/paper_alignment.md` for the remaining data, prompt, metric and split requirements.
