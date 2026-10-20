<div align="center">

# RemoteSAM++: A Mask-Centered Foundation Model for Multi-modal Earth Observation

Liang Yao · Fan Liu · Junjie Zhou · Chuanyi Zhang · Ziyun Chen · Shimin Di

[Code](https://github.com/1e12Leon/RemoteSAM-PlusPlus) · [Conference predecessor](https://github.com/1e12Leon/RemoteSAM) · [Paper alignment](docs/paper_alignment.md)

</div>

> Author-review version. Core model and task implementations are preserved from the server project. Some manuscript-specific evaluation sources are missing; these are listed explicitly below. This draft does not claim complete benchmark reproduction.

## Introduction

RemoteSAM++ extends RemoteSAM to a shared RGB–SAR framework. DINOv3 ConvNeXt-B and GTE-ModernBERT-Base encode images and referring expressions. Modality-specific LoRA branches adapt **pointwise_conv1** and **pointwise_conv2** in visual stages **0 and 1**; the vision–language fusion and mask decoder remain shared.

One language-conditioned mask interface supports eight tasks for either modality: referring expression segmentation, semantic segmentation, visual grounding, object detection, multi-label classification, image classification, image captioning, and object counting. The task transitions introduce no trainable task-specific heads.

RemoteSAM-520K contains 520K image–text–mask triplets spanning 358 categories and 25 attribute types, combining RGB and SAR supervision under one annotation format.

**Implementation note:** the original training server uses reference annotations or explicit modality selection. The recovered dishimin evaluators include the manuscript's channel-statistics automatic gate. Both source behaviors are preserved and their scope is documented in [source audit](docs/recovered_sources.md); they have not been silently made numerically equivalent.

## Setting Up

Use Linux with Python 3.11. [Observed server versions](configs/environment_observed.json) are recorded for reproducibility. The following matches its main packages; a fresh-environment install still needs separate validation.

```bash
conda create -n remotesam_plusplus python=3.11
conda activate remotesam_plusplus
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-eval.txt
```

SwanLab logging is optional (`pip install swanlab==0.7.16`). EPOC requires the original DirectSAM assets. Existing CIDEr evaluation uses Java and the Stanford tokenizer JAR; its tokenization is unchanged.

### Model Preparation

Retain the original encoder locations:

```text
RemoteSAM-PlusPlus/
├── dinov3-convnext-B/       # model config + pretrained weights
├── gte-modernbert-base/    # text model + tokenizer
├── checkpoints/
│   └── remotesam_plusplus.pth
└── chendelong/
    └── DirectSAM-EntitySeg-1024px-0501/  # optional EPOC
```

Weights and datasets are external assets. Final download links must be confirmed by the authors; the predecessor's weights are not interchangeable with RemoteSAM++.

New launchers use [configs/paper_model.json](configs/paper_model.json): 896×896 input, 20 tokens, LoRA stages `0,1`, rank `4`, alpha `8`, dropout `0.05`. These match the server's `lora01` checkpoint metadata and LoRA state-dict stages. Its residual text feature adapter is retained.

## Data Preparation

Training retains the existing reference/COCO format:

```text
datasets/RemoteSAM-520K/
├── images/
├── refs(unc).p
└── instances.json
```

Each reference contains `split`, `ann_id`, `file_name`, `sentences[0]["sent"]`, and `modal` (`rgb` or `sar`). Annotation IDs match the COCO annotation IDs. Images and masks use the existing transforms.

See [data preparation](docs/data_preparation.md) for SAR layouts and task annotations. Preserve the original splits and modality labels.

## Training

The original `train.py`, losses, optimizer groups, schedule, sampling, validation split and checkpoint format remain unchanged. This launcher only supplies the paper model configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train_paper.py \
  --nproc-per-node 8 \
  --imageFolder ./datasets/RemoteSAM-520K/images \
  --ref_file_path './datasets/RemoteSAM-520K/refs(unc).p' \
  --instances_path ./datasets/RemoteSAM-520K/instances.json \
  --output-dir ./checkpoints/paper
```

Use `--dry-run` to inspect the command. Existing arguments, including `--resume`, learning rates and optional SwanLab logging, remain available. See [training](docs/training.md).

## Getting Started

The example uses the existing evaluation loader and task conversion class:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/demo.py \
  --checkpoint ./checkpoints/remotesam_plusplus.pth \
  --image /path/to/image.png --modal sar --task REF \
  --sentence 'ship in the image' --output-dir ./result/demo
```

Use `--modal rgb` or `--modal sar` with modality-appropriate queries:

| Task | Option | Query |
|---|---|---|
| Referring expression segmentation | `--task REF` | `--sentence 'the airplane on the right'` |
| Semantic segmentation | `--task SEG` | `--classes airplane vehicle` |
| Visual grounding | `--task VG` | `--sentence 'the airplane on the right'` |
| Object detection | `--task DET` | `--classes airplane vehicle` |
| Multi-label classification | `--task MLC` | `--classes airplane vehicle` |
| Image classification | `--task MCC` | `--classes airplane vehicle` |
| Image captioning | `--task CAP` | `--classes airplane vehicle` |
| Object counting | `--task CNT` | `--classes airplane vehicle` |

Add `--epoc` when the experiment requires the existing EPOC path. Query vocabularies are benchmark-specific. A task demo is not a substitute for benchmark evaluation.

## Evaluation

Eight task-named Python evaluators are the public interface for both RGB and SAR. Benchmark-specific formats are handled by internal adapters; task conversion and metric calculations are retained from the source projects:

```text
tasks/code/eval/{REF,SEG,VG,DET,MLC,MCC,CAP,CNT}.py
tasks/{REF,SEG,VG,DET,MLC,MCC,CAP,CNT}.sh  # optional convenience launchers
```

List all cases and source availability:

```bash
python tasks/run.py --list
```

RGB referring segmentation:

```bash
CHECKPOINT=./checkpoints/remotesam_plusplus.pth CUDA_VISIBLE_DEVICES=0 \
python tasks/code/eval/REF.py --modal rgb --dataset rrsisd \
  --images /path/to/JPEGImages --refs '/path/to/refs(unc).p' \
  --instances /path/to/instances.json
```

WHU-OPT-SAR unseen semantic segmentation:

```bash
CHECKPOINT=./checkpoints/remotesam_plusplus.pth CUDA_VISIBLE_DEVICES=0 \
python tasks/code/eval/SEG.py --modal sar --dataset whu --data-root /path/to/SegEarthOV2
```

`--dry-run` prints commands without inference. `PYTHON_BIN` selects Python; `CUDA_VISIBLE_DEVICES` selects physical GPUs. No new launcher hardcodes a user's absolute path or GPU ID.

| Task | RGB evaluation sources | SAR evaluation sources |
|---|---|---|
| REF | RRSISD, RisBench | SARDet-100K, OpenEarthMap-SAR, FUSAR-Map, PIE-RGB-SAR, YESeg-OPT-SAR, DDHRNet |
| SEG | Potsdam, iSAID, Vaihingen; labeled DeepGlobe folder | WHU-OPT-SAR; supplementary source datasets |
| VG | RSVG | SARDet-100K refs/instances; SARVG supplementary |
| DET | DIOR, DOTAv2 export | SARDet-100K task support |
| MLC | DIOR, DOTAv2, SATIN four subsets | MSAR-1.0 accuracy/mAP |
| MCC | UC-Merced, RESISC45, PatternNet | FUSAR-SHIP, MSTAR, SAR-ACD, SAR-VSA supplementary |
| CAP | UCM CIDEr; paper's Sydney/RSIEval/UCM ReconScore pending | SARLANG-1M-Cap CIDEr |
| CNT | DIOR, DOTAv2 | SARDet-100K |

See [evaluation](docs/evaluation.md) for exact commands and metric caveats. Missing paper cases fail explicitly. An `available` case means source present, not that the full published number has been rerun.

## Generalization

WHU-OPT-SAR retains the original seven-class convention, probability-argmax merge and ignore-label policy. Recovered SATIN, Vaihingen and DeepGlobe evaluators are integrated through `MLC.py` and `SEG.py`. SATIN uses the constituent train splits and micro accuracy over labels; the recovered DeepGlobe folder protocol uses labeled training images, not an official held-out test split. These distinctions must accompany reported results. See [evaluation](docs/evaluation.md), [generalization](docs/generalization.md) and [source audit](docs/recovered_sources.md).

## Repository Layout

```text
arc/                  existing adaptive convolution components
lib/                  model, LoRA, fusion, text adapter and decoder
loss/                 unchanged training loss
data/                 server dataset adapters
test.py               original reference-format evaluation
test_rgb_sar.py        original RGB/SAR evaluation; import ambiguity fixed
tasks/code/eval/      eight task-named public Python entry points
tasks/code/_adapters/ internal benchmark-format/protocol implementations
tasks/code/metric/    inherited numerical metrics and tokenizer assets
tasks/code/model.py   original explicit-modality task conversion
tasks/code/model_auto.py recovered automatic-modality task conversion
tasks/*.sh            eight optional task launchers and common.sh
configs/              paper configuration and coverage registry
examples/demo.py      eight-task example using the existing model interface
tools/                preparation, training, SATIN summary and validation
docs/                 preparation, evaluation and paper alignment
```

## Review and Validation

Run `python tools/check_release.py` for syntax and launcher checks. Add `--reference-source /path/to/server_snapshot` to verify preserved implementation files. See [paper alignment](docs/paper_alignment.md) and [review changes](docs/review_changes.md) before release. Full benchmark reproduction and a clean installation are separate validation steps.

See [effective configuration](docs/effective_configuration.md) for the exact training/test defaults, source-dependent modality routing and commands to inspect them. The original training implementation is unchanged. Recovered evaluators keep their original automatic gate; older evaluators keep explicit routing. This source-version distinction has not been hidden by forcing one routing rule across all experiments.

## Acknowledgements

This project extends [RemoteSAM](https://github.com/1e12Leon/RemoteSAM) and its RMSIN/LAVT-based framework. It uses DINOv3, GTE-ModernBERT, LoRA and the existing EPOC/DirectSAM components. Third-party sources and notices are retained. License selection and asset redistribution rights need author confirmation; this draft does not assign a new license.

## Citation and Contact

Current manuscript: *RemoteSAM++: A Mask-Centered Foundation Model for Multi-modal Earth Observation*, by Liang Yao, Fan Liu, Junjie Zhou, Chuanyi Zhang, Ziyun Chen and Shimin Di. The final publication identifier and BibTeX entry have not been supplied for this review. See the [RemoteSAM citation](https://github.com/1e12Leon/RemoteSAM#cite) for the conference predecessor. Corresponding author in the supplied manuscript: Fan Liu, fanliu@hhu.edu.cn.
