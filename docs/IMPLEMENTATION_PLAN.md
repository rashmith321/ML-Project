# IMPLEMENTATION_PLAN.md

## G. Training sequence

1. **Classification backbone** — pretrain/fine-tune EfficientNet on Food-101 (largest, cleanest labeled set → establishes a working transfer-learning backbone first).
2. **Segmentation** — prompt SAM with ground-truth-quality boxes on UNIMIB2016, fine-tune/evaluate the supervised head against UNIMIB2016 masks.
3. **Detection** — train YOLO on whatever boxes are available (UNIMIB2016 instance boxes derivable from its masks; ECUSTFD boxes once integrated) — deliberately sequenced *after* segmentation/classification are proven, since detection quality only needs to be "good enough to seed SAM prompts," not state-of-the-art on its own.
4. **Portion/mass regression** — train on Nutrition5k RGB(-D) + mass labels, once detection+segmentation can produce a masked crop to feed it.
5. **Multi-nutrient regression** — joint calories/protein/carbs/fat head on Nutrition5k, sharing input features with step 4 (or a shared trunk).
6. **Ingredient auxiliary head** — Recipe1M+/Vireo-172-driven, trained last since it doesn't gate the primary nutrition numbers.
7. **End-to-end assembly + aggregation logic + report generation.**

This order is a change from the Drive notebook's original "Stage 1 = segmentation first" plan only in that classification is pulled earlier — Food-101 is fully downloaded and ready right now, so it's the fastest path to a first working component; segmentation remains the very next step and picks up exactly where the paused Stage 1 notebook left off.

## H. Evaluation strategy

| Stage | Metric | Ground truth source |
|---|---|---|
| Detection | mAP@0.5 | UNIMIB2016 / ECUSTFD boxes |
| Segmentation | mean IoU, per-instance | UNIMIB2016 masks |
| Classification | top-1 / top-5 accuracy | Food-101, Vireo-172 |
| Mass/volume | MAE (g / cm³) | Nutrition5k, ECUSTFD |
| Nutrients | MAE + MAPE per nutrient | Nutrition5k |
| End-to-end meal report | Total-calorie MAE vs. MenuMatch entries (auxiliary, real-world check only — not used to tune the model) | MenuMatch |

Report every metric alongside which dataset produced its ground truth — never a single blended "accuracy" number across datasets with different label types.

## I. Final application architecture

- `app/` — single-image/meal upload → runs the full `src/` pipeline → renders the per-food + total report in the brief's exact format.
- Lightweight framework (Streamlit for a fast internal demo, or FastAPI + minimal frontend if a proper API is wanted) — pick based on whether the deliverable needs to be "a demo you click through" or "a service" — not yet decided, flagged for your input rather than assumed.
- Runs on CPU by default; the app should detect CUDA and use it opportunistically.

## J. Risks / Limitations

- **Nutrition5k is the single point of failure for every nutrient number.** Any bias/gap in its food coverage becomes a gap in the whole system's nutrient coverage — foods outside Nutrition5k's ~300 dishes fall back to "reference lookup" or "unavailable," per the schema, not a guess.
- **SAM inference cost** — heaviest component in the pipeline; CPU inference will be slow. Budget for a lighter supervised segmentation fallback if CPU-only deployment is a hard requirement.
- **Reference repo has effectively zero code reuse** (see `PROJECT_AUDIT.md` §C) — this is a from-scratch build guided by, not built on top of, the original project; attribution is still required per its license.
- **Khana exclusion reduces classification diversity** slightly versus the earlier Drive plan; Vireo-172 (approved, not yet integrated) is the natural replacement for auxiliary classification breadth and should be prioritized to fill that gap.
- **Timeline risk for a final-year project scope:** six model-training stages plus aggregation and an app is substantial. Recommend treating steps 1–2 (classification + segmentation) as the MVP milestone, with 4–5 (mass + nutrients) as the core research contribution, and the app/report as integration work at the end.

## Immediate next actions

1. Manually open the reference repo's main `.ipynb` once (this environment can't) to confirm the calibration-object assumption noted in `PROJECT_AUDIT.md` §B.5.
2. Download ECUSTFD, Recipe1M+, Vireo-172, MenuMatch into `food_seg_project` (currently missing, all approved).
3. Resume the paused Stage 1 SAM notebook — run it against real UNIMIB2016 data and report metrics, exactly as it was already waiting for.
4. Decide Streamlit vs. FastAPI for `app/` (see §I).
