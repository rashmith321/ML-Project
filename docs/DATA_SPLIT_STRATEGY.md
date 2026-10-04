# Data split strategy (Phase 1)

Full rationale and code lives in `src/data/splits.py` — this file is the
human-readable summary the project brief asked for.

## Rule

**No two samples that are really the same underlying item are allowed to
land in different splits.** "Same underlying item" means: the same
Nutrition5k dish (its overhead RGB, its depth map, and any side-angle
frames), the same ECUSTFD physical dish (its multiple calibration-object
camera views), the same Recipe1M+ recipe (its multiple photos), etc.

## Per dataset

| Dataset | Split source | Grouping key |
|---|---|---|
| Nutrition5k | Official `dish_ids/splits/{rgb,depth}_{train,test}_ids.txt` | `dish_id` |
| Food-101 | Official `meta/{train,test}.txt` | n/a (one image = one sample, no multi-view) |
| Recipe1M+ | Official `partition` field in `layer1.json` | `recipe_id` |
| Vireo Food-172 | Official `SplitAndIngreLabel/{TR,VAL,TE}.txt` | n/a |
| UNIMIB2016 | **No official split exists.** Constructed: seeded grouped hash split, 80/10/10. | tray-photo id (each tray photo is already a self-contained sample) |
| ECUSTFD | **No official split exists.** Constructed: seeded grouped hash split, 80/10/10. | dish id (filename with the trailing `_<view>` suffix stripped) |
| MenuMatch | N/A — evaluation-only per `DATASET_PLAN.md`, never used for training. Every sample is assigned `split="eval_only"`. | n/a |

## How the constructed split works

For UNIMIB2016 and ECUSTFD, `src/data/splits.assign_grouped_split()`:
1. Takes each sample's `group_id` (or `instance_id` if the dataset has no
   natural grouping).
2. Hashes `f"{seed}:{group_id}"` with SHA-256 and maps the result
   deterministically into train/val/test at 80/10/10.
3. Every sample sharing a `group_id` gets the same hash input, so they
   always land in the same bucket. Re-running with the same seed
   (default `42`, set via `--split-seed` in `create_manifests.py`)
   reproduces an identical split.

## Verification

`scripts/validate_datasets.py` calls `src/data/splits.find_split_leakage()`
on every manifest after split assignment. It groups by the leakage key
(`group_id`, or `instance_id` when no group exists) and fails the run
(non-zero exit) if any group has members in more than one split. This was
verified to actually catch leakage, not just pass silently: a deliberate
test fixture with two views of the same ECUSTFD dish split across
train/test produced

```
Split leakage in ecustfd: group_id='apple001' appears in splits {'train': 1, 'test': 1}
```

and a non-zero exit code, before the fix.

## Unassigned samples

Any sample whose `split` is still `None` after adapter parsing + grouped-split
assignment (e.g. an official split file was expected but wasn't found, or a
row couldn't be matched into it) is left `split=None` rather than being
force-assigned. `create_manifests.py` prints this count per dataset; such
samples must not be used for training or evaluation until resolved.
