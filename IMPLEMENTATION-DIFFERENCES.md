# Implementation differences

The rewrite retains CCGL's AugSIM → GraphWave → contrastive BiGRU → fine-tuning
→ self-distillation pipeline. The original code remains in [legacy/tensorflow](legacy/tensorflow).

| Area | Original code | PyTorch code |
| --- | --- | --- |
| Storage | Path/time TXT files | Lossless Parquet input shared with CasFlow; model-specific `.pt` training caches. |
| Records | Parsing unlabeled/augmented files as labeled text dropped the final event. | Retain all events before the 100-event limit. |
| Views | Pre-training read the same extra unlabeled view twice. | Use independent, ID-aligned views. |
| AugSIM | Some degree calculations omitted the final edge; deletion modified the list being traversed. | Truncate first, compute the complete expanded graph and apply leaf removals together. Proportional strength 0.1 and arrival intervals are retained. |
| GraphWave | Omitted parents without their own adoption record; computed features per graph. | Retain edge endpoints, including Twitter's structural root `-1`; batch computation in float64 and store float32 features. |
| Heads | The 4-1 projection changed ReLU to linear after pre-training; count output was linear. | Retain projection ReLU and use softplus count output. |
| Unlabeled pool | Twitter's 7,396 candidates overlapped the labeled data. | Remove all 701 overlaps, retaining 6,695; labeled splits stay unchanged. |
| Training | Batch-averaged validation loss, repeated padding and fixed-teacher inference. | Sample-averaged validation loss, padding once and teacher predictions computed once. |

Historical labeled splits and subset selection, incremental targets,
log(1+x) training loss and base-2 reporting are retained. The BiGRU still
processes unmasked 100-step zero-padded sequences; repeated adopter IDs remain
separate events. Weibo has 19,431 extra unlabeled cascades; the paper's printed
19,413 is an author-confirmed typo.

Other historical augmentations and baselines remain in the archive.
