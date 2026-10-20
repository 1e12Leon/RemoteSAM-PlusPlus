# Recovered evaluation sources and protocol review

Source: dishimin's zxcpu14 (dsftrain0), `/data/dishimin/liangyao/remotesam`, inspected on 2026-10-20. The similarly named `RemoteSAM++` directories on both machines contain dataset resources, not a newer training codebase. Only selected project source files are included in this review; environments and datasets are not release contents.

| Task / case | Recovered source | Preserved protocol |
|---|---|---|
| MLC / SATIN | `MLC_SATIN.py`, merge helper | Four constituent train splits; native vocabularies; micro label accuracy and separate exact-match accuracy |
| MLC / MSAR | `MLC.py`, `data/datasets/MLC.py` | Four classes; label accuracy; native score-ranked AP averaged over classes with positives |
| VG / SARDet-100K | `VG.py`, `data/datasets/VG.py` | Reference expression and instance-box matching; normal predicted-box IoU evaluation |
| SEG / Vaihingen | `SEG_Vaihingen.py` | Exactly 113 labels; RGB palette; six predicted classes; means excluding clutter plus all-six means |
| SEG / DeepGlobe | `SEG_DeepGlobe_folder.py` | Labeled train folder; six land-cover classes; unknown ignored; native resizing/argmax |
| Automatic routing | `lib/modal_inference.py`, `tasks/code/model.py` | 2-D input → SAR; mean pairwise RGB-channel absolute difference ≤1 → SAR; otherwise RGB |

The native prediction loops, prompts, thresholds and numerical metrics were not rewritten. Files are relocated into `_adapters/`. For MSAR and SARDet grounding, only imports select recovered dataset classes and the recovered automatic-routing task wrapper (`model_auto.py`); the original dataset factory and `model.py` remain unchanged. A small dataset-selector module connects these recovered implementations without extending or altering the original training dataset factory.

The older evaluators continue using explicit `--modal` or annotation modality. The recovered MLC/VG wrapper infers modality from input content even when `--modal` is provided. Recovered RGB SEG similarly uses the content gate. `--modal` selects the supported evaluation family, but does not override an automatic-routing source's own scientific protocol. This difference is intentionally visible and must be reviewed before publication; no claim of numerical equivalence between source versions is made.

## Not included as main evaluation

- `VG_SARDet_oracle_det.py` uses ground-truth IoU as an oracle score for an additional AP analysis. It is not a valid replacement for normal grounding evaluation and is excluded.
- Road-specific DeepGlobe scripts are a different task/dataset variant and are not silently substituted for land-cover segmentation.
- Comparison-method VLM MSAR scripts under `others/` are not the RemoteSAM++ evaluator and are not bundled as such.
- RGB caption ReconScore implementations were not found in the inspected project task/model directories. The three current paper cases remain explicit errors.

## Verification boundary

`tools/check_release.py --reference-source ... --recovered-source ...` compares 70 original implementations (accounting for path relocation and the caption import path) and 11 recovered implementations (allowing only documented import integration). Syntax and CLI smoke tests are separate from full-data score reproduction. SATIN requires `pyarrow`; neither a parser check nor code availability proves paper numbers.
