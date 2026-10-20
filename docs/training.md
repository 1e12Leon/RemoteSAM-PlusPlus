# Training and model configuration

`train.py` is unchanged from the server snapshot. Dataset sampling, CE/Dice loss, optimizer parameter groups, learning rates, text adapter, schedule, checkpoint selection, and RGB/SAR routing are preserved.

The only parser default changed is `--modal_lora_stages`: `2,3` becomes `0,1` to match the current manuscript and the inspected final checkpoint. All public launchers also pass this value explicitly. The underlying backbone constructors retain their original implementation; they receive the explicit paper arguments.

## Paper model

| Setting | Value |
|---|---|
| Model | `lavt_one` |
| Visual encoder | `dinov3_convnext_b` |
| Text encoder | `modernbert`, local GTE-ModernBERT |
| Resolution | 896×896 |
| Token length | 20 |
| Modal LoRA | enabled; separate RGB/SAR residuals |
| LoRA stages | 0 and 1 |
| LoRA target layers | pointwise_conv1, pointwise_conv2 |
| Rank / alpha / dropout | 4 / 8 / .05 |
| Text feature adapter | enabled; original residual adapter retained |

These are encoded once in `configs/paper_model.json`. The observed server checkpoint was saved at epoch index 38 and its state-dict LoRA keys are in stages 0/1. This review does not distribute that checkpoint.

## Commands

```bash
python tools/train_paper.py --dry-run --nproc-per-node 8 \
  --imageFolder /path/to/images --ref_file_path '/path/to/refs(unc).p' \
  --instances_path /path/to/instances.json --output-dir ./checkpoints/paper
```

Remove `--dry-run` after verifying the paths. The helper invokes the original `train.py` through PyTorch's distributed launcher with `--epochs 40`. Extra native options are forwarded, so deliberate ablation overrides remain possible and should not be reported as the default paper configuration.

Single-process testing remains available through `test.py`. `test_rgb_sar.py` retains its original functions and explicitly loads the root `test.py` to avoid module-name collisions. SAR source-specific REF helpers are internal adapters rather than a top-level `test/` package.

## Preservation verification

```bash
python tools/check_release.py --reference-source /path/to/server_snapshot \
  --report ./review_artifacts/source_audit.json
```

This compares normalized source contents of protected implementations. It does not equate syntax checks with full benchmark reproduction. No model forward function, loss, task conversion, benchmark metric calculation, threshold or prompt in the inherited evaluators has been changed.

The automatic image-statistics gate was recovered from dishimin and retained for those evaluators. Original training and older evaluators are not converted to automatic routing. See `recovered_sources.md` for this source-version distinction and `effective_configuration.md` for all effective defaults.
