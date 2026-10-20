# Manuscript-to-code audit

This review uses the supplied `main.txt` on 2026-10-20 and the server source snapshot from `/home/zhoujunjie/zhoujunjie/test/remotesam`. The paper is not modified. A source-present entry is not a claim of numerical reproduction.

## Architecture and checkpoint

| Manuscript item | Supplied implementation | Review treatment |
|---|---|---|
| Shared DINOv3 ConvNeXt-B | `lib/dinov3_convnext.py` | Unchanged |
| GTE-ModernBERT-Base | `lib/text_encoders.py` | Unchanged |
| Shared PWAM/CIM and mask decoder | `lib/dinov3_convnext.py`, `lib/_utils.py`, `lib/mask_predictor.py` | Unchanged |
| LoRA in stages 0/1, pointwise_conv1/2 | Confirmed in `lora01` metadata and actual state-dict keys | `args.py` default and all new paper launchers use `0,1` |
| LoRA rank 4, alpha 8, dropout .05 | Confirmed in checkpoint args | Explicit paper config |
| Input 896, maximum tokens 20 | Confirmed in checkpoint args | Explicit paper config |
| Residual text feature adapter | Present in source and checkpoint (`no_text_adapter=False`) | Retained; manuscript architecture description should disclose it |
| Automatic channel-statistics modality gate | Found in recovered dishimin `lib/modal_inference.py` and task wrapper | Recovered evaluations retain it; original training-server behavior remains explicit/metadata-based |
| 283M parameters | Strict-load verification counted 283,677,853 model parameters | Includes model-owned text encoder; not optimizer/trainable-only count |

The source-version routing difference requires author review before claiming one identical inference protocol across every paper experiment. The recovered gate is not invented here. The existing LoRA forward also computes both residual expressions before selecting by a mask; the selected output is modality-specific, but execution is not branch-skipping. Existing behavior is preserved.

## Eight tasks × two modalities

| Task | RGB | SAR |
|---|---|---|
| REF | RRSISD/RisBench evaluators present | Six source-specific evaluators present |
| SEG | Potsdam/iSAID sources present | WHU-OPT-SAR source present |
| VG | RSVG source present | SARDet-100K refs/instances source recovered; SARVG supplementary |
| DET | DIOR evaluation + DOTA export present | SARDet-100K evaluator present; not a separate current-paper table |
| MLC | DIOR/DOTA and SATIN sources present | MSAR-1.0 four-class adapter and ranked mAP recovered |
| MCC | UCM/RESISC45/PatternNet sources present | Four supplemental SAR sources present; no corresponding paper table |
| CAP | Existing UCM CIDEr available | SARLANG CIDEr source present |
| CNT | DIOR/DOTA source present | SARDet-100K source present |

Eight task operations are available in the existing `RemoteSAM` wrapper for both modalities. This does not mean all sixteen paper benchmark protocols are present.

## Exact paper sources still needed

1. RGB captioning: Sydney/RSIEval adapters and ReconScore evaluation for Sydney, RSIEval and UCM remain missing in the inspected source locations. Existing caption evaluation computes CIDEr and is not a substitute.
2. Confirm the recovered SATIN/Vaihingen/DeepGlobe/MSAR/SARDet protocols, split artifacts and checkpoint against the exact paper runs. Source recovery is not numerical reproduction; see `recovered_sources.md`.
3. Confirm that SARLANG's current `captioning` call and tokenization are exactly those used for the reported concise-caption result. No new caption-generation rule is introduced here.

Registry entries for missing cases stop with an explanatory error. This is deliberate: a placeholder that silently evaluates a different dataset or metric would give a false impression of completeness.

## Other protocol details to verify

- Original `train.py` validates `split='test'` each epoch; this is preserved and must be explained against the published split protocol.
- `All_Dataset` returns `exist=True`; negative examples still use their binary masks in the unchanged CE/Dice path. Do not describe an additional absence-classification loss.
- Detection expects the existing class-index category IDs. Validate converted COCO categories before reporting AP.
- Default distributed samplers in some inherited evaluators may pad samples; this review does not change their aggregation logic.
- Rule-based captioning contains random template selection. Do not claim bit-exact caption metrics without confirming the original seed/protocol.
- Main-paper commands must specify whether EPOC was enabled. The native DET and CNT EPOC thresholds differ (.25 and .15); both are retained.

## Approval boundary

This is a review candidate, not a full paper-reproduction release. Missing ReconScore sources and source-version protocol differences need author review before removing the notice. No metric or automatic gate was invented to fill a gap. The author has authorized updating the existing GitHub repository with this snapshot; that update is not a claim of full numerical reproduction.
