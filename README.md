# RemoteSAM RGB Referring Segmentation

This directory is a compact RGB-only refactor of the RemoteSAM training and
testing code. The model architecture follows the original RemoteSAM/LAVT design:
language-guided PWAM fusion, CIM feature interaction, and the original
lightweight segmentation decoder.

By default the visual/text encoders are `DINOv3-ConvNeXt-B` and `ModernBERT`,
loaded from the local `dinov3-convnext-B/` and `gte-modernbert-base/` folders.
The Swin-B + BERT path remains available with
`--vision_backbone swin_b --text_encoder bert`.

## Layout

```text
arc/                 adaptive rotated convolution decoder layers
data/all_dataset.py  RGB COCO-style referring segmentation dataset
lib/                 RemoteSAM model, DINOv3/Swin backbones, text encoders
loss/loss.py         CE + Dice segmentation loss
train.py             distributed or single-process training/evaluation
test.py              single-process checkpoint evaluation
args.py              shared CLI arguments
```

## Data Format

Pass the following paths at runtime:

```bash
--imageFolder /path/to/images \
--ref_file_path /path/to/refs\(unc\).p \
--instances_path /path/to/instances.json
```

The refs pickle should contain RemoteSAM-style entries with `split`, `ann_id`,
`file_name`, and `sentences[0]["sent"]`. The instances file should be COCO-style
JSON with annotation ids matching `ann_id`.

## Train

```bash
torchrun --nproc_per_node=1 train.py \
  --imageFolder /path/to/images \
  --ref_file_path /path/to/refs\(unc\).p \
  --instances_path /path/to/instances.json
```

## Test

```bash
python test.py \
  --resume /path/to/model_best_RemoteSAM.pth \
  --split val \
  --imageFolder /path/to/images \
  --ref_file_path /path/to/refs\(unc\).p \
  --instances_path /path/to/instances.json
```
