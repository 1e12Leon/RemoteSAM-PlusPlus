# Data preparation

## Shared training format

The existing `data/all_dataset.py` loads images, `refs(unc).p`, and COCO `instances.json`. The reference fields are `split`, `ann_id`, `file_name`, `sentences[0]['sent']`, and `modal`. The image path is resolved relative to `--imageFolder`. The COCO annotation referenced by `ann_id` must contain the original RLE segmentation. Preserve mask values and the existing two-class conversion.

Training consumes the original `train` and `test` split names. This review does not resplit data. `modal='rgb'` routes to ID 0; values containing `sar` route to ID 1. An absent modality field defaults to RGB in the inherited loader, so SAR refs must be annotated explicitly.

## RGB task inputs

| Evaluation | Required input |
|---|---|
| RRSISD / RisBench REF | images, reference pickle, COCO instances |
| Potsdam SEG | matching RGB images and original color-coded masks |
| iSAID SEG | original color masks and inherited filename convention; `SEG.py` uses the first five stem characters for source image lookup |
| RSVG VG | existing XML/image layout expected by `RSVG_VG` |
| DIOR DET / MLC | converted COCO JSON; match the evaluator's category names and zero-based category-index convention |
| DOTA DET | image patches and DOTA `annfiles`; native evaluator exports Task2 HBB text |
| DOTA MLC | patches and actual class annotations; no-GT test sets produce predictions only |
| UCM / RESISC45 / PatternNet MCC | per-class image folders, with inherited class-name-to-folder mappings |
| UCM CIDEr CAP | existing Karpathy-style JSON (`images`, `split`, `filename`, `imgid`, `sentences[].raw`) |
| DIOR / DOTA CNT | existing count JSON, not a raw detection annotation file |

Preparation utilities retain their original behavior:

```bash
python tools/convert_dior_coco_to_cnt.py --help
python tools/convert_dota_annfiles_to_cnt.py --help
python tools/prepare_dotav2_patches.py --help
```

## SAR referring segmentation

The six original evaluators under `test/` share `common_sar_eval.py`. The prepared reference-data layout is:

```text
SOURCE/
└── test/
    ├── images/
    ├── refs.p
    └── instances.json
```

Use `--data-root /path/to/SOURCE` for the selected source. Valid launcher case names are `sardet100k`, `openearthmap`, `fusarmap`, `pie`, `yeseg`, `ddhrnet`. Do not swap these REF inputs with semantic segmentation labels without the original preprocessing.

## SAR semantic segmentation

`SEG_SAR.py` retains dataset-specific paths, classes, aliases, resize and ignore conventions. Pass the root containing the original prepared subdirectories using `--data-root` in the public launcher. It maps to the native `--seg_sar_root`.

WHU evaluation expects:

```text
SegEarthOV2/
└── whu-opt-sar/crop512/test/
    ├── sar_dir/*.tif
    └── ann_dir/*.tif
```

The server setup uses 20 original test scenes and 1,760 paired 512×512 crops. Original IDs `{0,10,...,70}` are divided by ten by the existing preparation. Native `reduce_zero_label=True` ignores prepared 0 and maps prepared IDs 1–7 to evaluation IDs 0–6. The seventh class is represented internally as `background` and prompted as `others`; it is included in the seven-class mean. Do not confuse this valid seventh class with the ignored raw label 0.

The review ships no replacement tiling algorithm. Preserve and verify the original crop assets; the existing verifier can be inspected with `python ablation/futher-exp/verify_whu_opt_sar.py --help` (see its original defaults before running).

## SAR downstream inputs

SARDet DET uses the existing COCO-style image/annotation set. SARDet CNT uses the JSON produced by `convert_dior_coco_to_cnt.py`. SARVG VG uses expression-bearing XML and split files and is supplementary, not the paper's SARDet-100K grounding protocol. SARLANG CAP uses the existing conversation-style caption JSON and referenced images. The four SAR MCC datasets use the native per-class image-folder layout.

MSAR-1.0 and SATIN source adapters have been recovered. MSAR expects native four-class COCO annotations (use `tools/convert_msar_to_coco.py` with explicit output directory); SATIN expects one directory per constituent dataset with native parquet image bytes/label IDs. See `recovered_sources.md` for class/split/metric details. RGB ReconScore adapters remain missing and no alternative metric is silently substituted.
