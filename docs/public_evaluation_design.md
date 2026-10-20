# Public evaluation interface — required release layout

This requirement supersedes the preliminary sixteen task/modality launcher layout. The review now implements eight task-named public Python files under `tasks/code/eval/`, eight optional task shell launchers and internal source-preserved adapters under `tasks/code/_adapters/`.

The public Python evaluation interface must contain eight task-named entry points, following RemoteSAM's task-oriented presentation:

```text
REF.py  SEG.py  VG.py  DET.py  MLC.py  MCC.py  CAP.py  CNT.py
```

RGB and SAR are selected through an explicit modality argument rather than separate public Python entry points. Dataset selection, class vocabularies, annotation formats and split paths are configuration or internal adapter concerns. Existing task conversion and metric implementations must be preserved.

Dataset-specific experimental scripts are provenance sources, not the intended public interface. Adding a common launcher above an otherwise unchanged collection of dataset-specific public scripts is not sufficient. Shared task evaluation belongs in the task implementation; genuinely different input formats or benchmark protocols belong in internal adapters. Metrics with different definitions must remain distinguishable.

When a task cannot safely share one evaluator across benchmarks, publish one representative, established benchmark implementation under the task name and document its supported input format and metric. Do not advertise unverified dataset or modality support. Additional paper-reproduction scripts may remain in a separate, clearly identified experimental area only when needed and reviewed.

Review checks for the task-oriented layout:

1. Inspect recovered original evaluation sources and record their provenance and protocol.
2. Separate data-format adaptation from task prediction and metric computation without changing numerical behavior.
3. Verify each task entry point and its supported modality/format combinations.
4. Update README commands, layout, launchers, registry and audits together; distinguish exact paper reproduction from representative task demonstrations.

The initial cleanup was prepared without committing or pushing. The author subsequently authorized updating the existing GitHub repository; publishing this snapshot does not remove the scientific-protocol caveats.
