# Review changes

The public evaluation layout now contains eight task-named Python entry points, shared by RGB and SAR. Dataset-specific protocols are internal adapters, not public entry files. See [public evaluation design](public_evaluation_design.md).

## Provenance

- GitHub base: `1e12Leon/RemoteSAM-PlusPlus`, main commit `1adff56491765218225302ff8d91fe1ad2f36b0e`.
- Layout/documentation reference: RemoteSAM commit `ebb7bc278c7343c29c8c16b035289f78f40f0f72`.
- Implementation source: server project `/home/zhoujunjie/zhoujunjie/test/remotesam`, code-only snapshot on 2026-10-20.
- Manuscript source: the current workspace `main.txt`; it is not included in the release copy.

This directory is independent of the server's working project. No original server files were overwritten. The initial review was prepared without committing or pushing; the author subsequently authorized updating the existing GitHub repository with this snapshot. Repository visibility is not changed by this update.

## Included

- The existing PlusPlus repository history and legacy sources are retained in the independent review checkout.
- Selected server model/data/task sources, six SAR REF tests and required CIDEr tokenizer assets are copied in.
- General preparation utilities are retained; manual single-example image edits, prompt sweeps, debug scripts, logs and large datasets/weights are excluded.
- Eight public task Python files, eight optional task shell launchers, one standard-library dispatcher, an eight-task demo, paper configuration and a coverage registry are added.
- Original evaluation implementations are relocated into `_adapters/`; inherited caption import paths follow the relocation. Dataset-specific files are not left in the public evaluation directory.
- Selected dishimin sources recover SATIN, Vaihingen, DeepGlobe, MSAR-1.0, SARDet-100K grounding and the automatic-modality wrapper. See `recovered_sources.md` for exact allowed import integration and numerical-protocol boundaries.
- README follows RemoteSAM's introduction → setup → model/data → training → getting started → evaluation → acknowledgement/citation organization.

## Intentional implementation-adjacent edits

1. `args.py`: change only the default LoRA stages and its help text to 0/1. This aligns model construction with the inspected final checkpoint.
2. `test_rgb_sar.py`: replace ambiguous `from test import ...` with explicit loading of the existing root `test.py`. Evaluation function bodies are unchanged.
3. Public task launchers select `--modal rgb|sar`; original evaluators move to internal adapters. Native numerical bodies and the original task conversion/model remain unchanged. Recovered evaluators select the separately preserved recovered automatic-routing wrapper and dataset classes by imports only.
4. Training defaults are made explicit in `configs/training_defaults.json`; the helper requires user-provided data paths and retains the native training implementation.

No model-forward, loss, optimizer, metric, prompt, decoding, EPOC threshold or checkpoint state-key change is part of this cleanup. Protocol gaps are exposed rather than filled with invented code.

## Validation artifacts

`tools/check_release.py` checks Python syntax, the sixteen registry groups, referenced scripts and preservation of protected implementation files against a provided snapshot. Reports are written under ignored `review_artifacts/` when requested. Runtime and benchmark checks must be distinguished from static checks.

`configs/environment_observed.json` records the actual server package versions, not an assertion that a clean environment has been recreated successfully.

## Before public release

- Resolve missing RGB ReconScore sources and review source-version routing/split differences in `paper_alignment.md`.
- Confirm final model/data links, paper identifier, license and redistribution notices.
- Run a fresh installation and the exact full-data paper protocols.
- Review the source diff and documented remaining gaps. Uploading this review snapshot does not establish full benchmark reproduction.
