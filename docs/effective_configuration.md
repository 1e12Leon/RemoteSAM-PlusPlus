# Effective training and evaluation configuration

These defaults refer to the public review helpers, not arbitrary historical shell scripts. Native options forwarded after the helper arguments can deliberately override them. Always inspect `--dry-run` before a run.

## Shared model

`configs/paper_model.json` selects `lavt_one`, DINOv3 ConvNeXt-B and GTE-ModernBERT-Base, input 896×896, 20 tokens, modality-specific LoRA enabled in stages 0/1 on `pointwise_conv1,pointwise_conv2`, rank 4, alpha 8, dropout .05. Residual text adapter is enabled with dropout .1. Local pretrained directories are `./dinov3-convnext-B` and `./gte-modernbert-base`.

## Training

`tools/train_paper.py` invokes unchanged `train.py` via torch.distributed.run. It requires explicit image/ref/instance paths; it does not select a dataset from a private server path. No fine-tuned checkpoint is resumed unless `--resume` is passed.

| Setting | Default |
|---|---|
| Processes / GPUs | 1; use `--nproc-per-node 8` for eight visible GPUs |
| Batch size | 2 per process; effective batch 16 for eight processes; no gradient accumulation |
| Epochs / workers | 40 / 8 per process |
| Optimizer / schedule | AdamW, AMSGrad off; per-iteration polynomial decay `(1-step/total)^0.9` |
| Base LR / visual backbone LR | 3e-5 / 1e-5 |
| Fusion / decoder / text adapter / LoRA LR | 5e-5 each |
| Text encoder LR | 5e-6; original selection of first 10 text layers plus optional output projection is retained; embeddings are not added to the optimizer by this selection |
| Weight decay | .01 on decay groups; 0 on original no-decay groups and LoRA |
| Loss | .9 × weighted CE (background .9, foreground 1.1) + .1 × original Dice |
| Seed | 2401, original train.py seeding |
| Validation | Original `test` split every epoch, batch 1; original best-mIoU/best-oIoU saving |
| Output | `./checkpoints/paper` |
| Modal routing | Per-reference `modal` metadata in original training data loader; missing values follow its RGB default |

This is not LoRA-only training: original trainable backbone/fusion/decoder/text groups are retained. The training helper does not create a toy subset or the 20%-validation subset from dishimin's separate ablation script. The actual training sample set is determined by your supplied refs/instances.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train_paper.py \
  --nproc-per-node 8 --dry-run \
  --imageFolder /path/to/images --ref_file_path '/path/to/refs(unc).p' \
  --instances_path /path/to/instances.json
```

Remove `--dry-run` after checking. Native `train.py` remains available but bypasses this explicit configuration helper.

## Testing

Public files are `tasks/code/eval/{REF,SEG,VG,DET,MLC,MCC,CAP,CNT}.py`. Required: modality, supported dataset/input format, checkpoint, and relevant data paths. There is no implicit final checkpoint selection. Defaults: one process, logical `cuda:0`, 4 workers, `test` split, output `./result/TASK_MODAL/dataset` (also SATIN statistics directory unless overridden), EPOC off unless `--epoc` is requested for DET/CNT/CAP. Existing EPOC thresholds remain DET .25, CNT .15; no thresholds are unified by this cleanup.

Routing differs by source: older evaluators use explicit modality; recovered SATIN/MSAR/SARDet-VG and RGB unseen SEG use the original image-statistics gate. See `recovered_sources.md`. Split overrides are only honored where the underlying source supports them: SATIN reads constituent train parquet files; DeepGlobe reads the labeled folder you supply.

WHU defaults retain seven valid classes including `others`, ignore raw label 0, merge foreground class probabilities with argmax, no calibration/smoothing, inference size 896 and native label evaluation size 224. The source-specific configuration is not replaced by a newly invented generic metric.

```bash
CUDA_VISIBLE_DEVICES=4 python tasks/code/eval/SEG.py --modal sar --dataset whu \
  --resume ./checkpoints/remotesam_plusplus.pth --data-root /prepared/SegEarthOV2 \
  --device cuda:0 --dry-run
```

Here `cuda:0` means physical GPU 4. Remove `--dry-run` to evaluate after checking prepared data. See `evaluation.md` for other tasks and SATIN summary commands.

## Checkpoint verification scope

The supplied lora01 best-oIoU checkpoint has model-owned parameter count 283,677,853, no missing/unexpected keys under strict loading, and finite CPU RGB/SAR forward outputs of shape 1×2×896×896. These are infrastructure checks, not reruns of paper scores. No GPU training or full benchmark run was started by this cleanup.
