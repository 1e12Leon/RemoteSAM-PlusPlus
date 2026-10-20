# Evaluation entry points

The eight public files `tasks/code/eval/{REF,SEG,VG,DET,MLC,MCC,CAP,CNT}.py` accept `--modal rgb|sar` and select supported formats with `--dataset`. The eight same-named shell launchers are optional. Only these task-level files are public evaluation entry points. Native input-format/protocol implementations live in `_adapters/`, with their original task conversion, metrics and prompts preserved. This layout keeps protocol differences explicit rather than claiming one numerical protocol fits every benchmark.

Status meanings:

- `available`: a matching evaluator is present; full paper metrics are not revalidated by this label.
- `supplementary`: existing evaluator for an additional benchmark or a different metric.
- `needs_source`: a paper-specific implementation/protocol is missing; invocation stops before model loading.

Use `python tasks/run.py --list`. Add `--dry-run` to any command below to inspect the invocation without data or GPU use. Set `CHECKPOINT` or pass `--resume` explicitly. All evaluations use one process by default; native distributed entry points remain available for reproducing the original multi-GPU protocol.

```bash
export CHECKPOINT=./checkpoints/remotesam_plusplus.pth
export CUDA_VISIBLE_DEVICES=0
```

## Referring expression segmentation

```bash
bash tasks/REF.sh --modal rgb --dataset rrsisd --images /rgb/images \
  --refs '/rgb/refs(unc).p' --instances /rgb/instances.json
bash tasks/REF.sh --modal rgb --dataset risbench --images /risbench/images \
  --refs '/risbench/refs(unc).p' --instances /risbench/instances.json
bash tasks/REF.sh --modal sar --dataset sardet100k --data-root /prepared/SARDet_100K
```

The other SAR case names are `openearthmap`, `fusarmap`, `pie`, `yeseg`, `ddhrnet`. Native REF mean object IoU, overall IoU and precision thresholds are retained.

## Semantic segmentation

```bash
bash tasks/SEG.sh --modal rgb --dataset potsdam --images /potsdam/images --annotations /potsdam/labels
bash tasks/SEG.sh --modal rgb --dataset isaid --images /isaid/images --annotations /isaid/masks
bash tasks/SEG.sh --modal sar --dataset whu --data-root /prepared/SegEarthOV2
```

Potsdam's clutter convention and native mF1/mIoU calculations are unchanged. iSAID uses the inherited color palette and filename rules. SAR evaluation preserves each native dataset configuration. No class calibration or smoothing is enabled by the WHU paper launcher. Extra evaluator-specific flags can be forwarded after `--`.

## Visual grounding

```bash
bash tasks/VG.sh --modal rgb --dataset rsvg --images /rsvg/JPEGImages --annotations /rsvg/Annotations
bash tasks/VG.sh --modal sar --dataset sarvg --images /sarvg/JPEGImages --annotations /sarvg/Annotations
```

SARVG is supplementary. The recovered SARDet-100K protocol uses reference expressions and instance boxes, not XML and not an oracle scoring script:

```bash
python tasks/code/eval/VG.py --modal sar --dataset sardet100k \
  --resume "$CHECKPOINT" --images /sardet/test/images \
  --refs /sardet/test/refs.p --instances /sardet/test/instances.json
```

## Object detection

```bash
bash tasks/DET.sh --modal rgb --dataset dior --images /dior/images --annotations /dior/test.json \
  --classes airplane ship --epoc
bash tasks/DET.sh --modal rgb --dataset dotav2 --images /dota/images --annotations /dota/annfiles \
  --classes airplane ship --epoc
bash tasks/DET.sh --modal sar --dataset sardet100k --images /sardet/Images/test \
  --annotations /sardet/Annotations/test.json \
  --classes ship aircraft car tank bridge harbor --epoc
```

The RGB examples above evaluate/export only two classes for brevity. Specify every native dataset class for total results. DOTA exports native Task2 HBB text; official patch merging and benchmark evaluation remain a separate step. COCO detection uses existing class-index IDs, so converted annotation IDs must match the evaluator. The launcher calls the unchanged summary helper after native COCO per-class evaluation. Do not report a selected-class AP as full-dataset AP.

## Multi-label classification

```bash
bash tasks/MLC.sh --modal rgb --dataset dior --images /dior/images --annotations /dior/test.json
bash tasks/MLC.sh --modal rgb --dataset dotav2 --images /dota/images --annotations /dota/annfiles
```

DIOR/DOTA retain the original label-accuracy evaluator. The recovered MSAR evaluator additionally calculates native ranked mAP with four classes (`ship`, `aircraft`, `oil tank`, `bridge`). A no-ground-truth DOTA split only exports predictions; it cannot produce accuracy.

```bash
python tasks/code/eval/MLC.py --modal sar --dataset msar --resume "$CHECKPOINT" \
  --images /MSAR-1.0 --annotations /MSAR-1.0/Annotations_COCO/test.json
python tasks/code/eval/MLC.py --modal rgb --dataset satin --resume "$CHECKPOINT" \
  --data-root /SATIN -- --stats_dir ./result/satin_stats
python tools/summarize_satin.py --stats_dir ./result/satin_stats \
  --dataset UC_Merced_LandUse_MultiLabel --num_shards 1
```

Repeat the summary for `AID_MultiLabel`, `MultiScene`, `MLRSNet`. The single-GPU SATIN entry writes raw native statistics before summarization. It evaluates all four constituent train splits by default and reports micro accuracy over labels, separately from exact-match accuracy. These are not official held-out test results.

Recovered RGB generalization protocols:

```bash
python tasks/code/eval/SEG.py --modal rgb --dataset vaihingen --resume "$CHECKPOINT" \
  --images /Vaihingen/images --annotations /Vaihingen/labels
python tasks/code/eval/SEG.py --modal rgb --dataset deepglobe --resume "$CHECKPOINT" \
  --images /DeepGlobe/train --annotations /DeepGlobe/train
```

Vaihingen checks exactly 113 test labels and includes clutter as a sixth class. DeepGlobe uses six classes and ignores unknown pixels in the labeled train folder. See the source audit before treating these as exact numerical reproductions of the paper.

## Image classification

```bash
bash tasks/MCC.sh --modal rgb --dataset ucm --images /UCM/Images
bash tasks/MCC.sh --modal rgb --dataset resisc45 --images /NWPU-RESISC45/test
bash tasks/MCC.sh --modal rgb --dataset patternnet --images /PatternNet/images
bash tasks/MCC.sh --modal sar --dataset mstar --images /sar_classification_root
```

The SAR supplemental case names are `fusar-ship`, `mstar`, `sar-acd`, `sar-vsa`. The inherited MCC evaluators report Top-1/Top-5. Pass the class-folder root expected by each native adapter; the public launcher maps it to both image/annotation roots without altering the adapter.

## Image captioning

```bash
bash tasks/CAP.sh --modal rgb --dataset ucm-cider --images /ucm/imgs --annotations /ucm/dataset.json
bash tasks/CAP.sh --modal sar --dataset sarlang --images /sarlang/SARimages \
  --annotations /sarlang/Text/Caption/test/Caption_test.json
```

Both use existing CIDEr evaluators. The RGB command is supplementary because the current paper uses ReconScore. Sydney/RSIEval/UCM ReconScore requests stop until the original code is supplied. Native rule-based text generation, class vocabularies, aggregation and random template selection are preserved. Confirm the exact concise-caption setting for the SARLANG table before claiming a numerical reproduction.

## Object counting

```bash
python tools/convert_dior_coco_to_cnt.py --coco-json /sardet/test.json --output-json ./result/sardet_cnt.json
bash tasks/CNT.sh --modal rgb --dataset dior --images /dior/images --annotations /dior/counts.json --epoc
bash tasks/CNT.sh --modal rgb --dataset dotav2 --images /dota/images --annotations /dota/counts.json --epoc
bash tasks/CNT.sh --modal sar --dataset sardet100k --images /sardet/Images/test \
  --annotations ./result/sardet_cnt.json --epoc
```

Counting preserves exact-count accuracy, existing object-area priors and EPOC threshold .15. Detection preserves EPOC threshold .25. The launcher leaves EPOC explicit because changing its setting changes experiment results.

## Naming and compatibility

Task codes remain the original `REF/SEG/VG/DET/MLC/MCC/CAP/CNT`. Public filenames have no modality or dataset suffix; select both explicitly with arguments. The registry uses `_RGB`/`_SAR` keys internally. Same-named shell scripts forward to the public Python files, not an implicit RGB-only launcher. Native numerical implementations are internal adapters; model state keys remain unchanged. Obsolete contour/DOTA convenience shell scripts were removed from this review copy (their originals remain in the source snapshot/GitHub base); DOTA is selected through `DET.py --dataset dotav2`.
