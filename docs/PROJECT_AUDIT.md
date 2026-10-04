# PROJECT_AUDIT.md

## A. Existing Reference Repository

**Repo:** [chetan-jarande/Food-calorie-estimations-Using-Deep-Learning-And-Computer-Vision](https://github.com/chetan-jarande/Food-calorie-estimations-Using-Deep-Learning-And-Computer-Vision)
**Authors:** Chetan Jarande, Mukta Bhagwat, Vishakha Patil, Diya Ukirde
**License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) — non-commercial use only, attribution required.
**Stats (at audit time):** 55 stars, 15 forks, 17 commits, public.

> **Inspection limitation, stated plainly:** this environment has no outbound shell/git access, and GitHub blocks automated fetching of anything beyond a page that was explicitly surfaced via search — so the `.ipynb` main notebook, the YOLOv4 `.cfg`, and the `.names` file could not be opened and read line-by-line from here. This audit is built from the repository's root file listing, its README (which documents the pipeline, training notebooks, and dataset workflow in detail), and the file names/extensions present. Before any component below is reused verbatim, it should be opened once by hand (or by Claude Code / Claude in a networked environment) to confirm exact cell contents.

### Root file listing
```
Extra_components/                                            (dir)
Food_Calorie_Estimation_BE_Project_PPT.pptx
Food_calorie_estimations_Using_Deep_Learning_And_Computer_Vision.ipynb   (main notebook)
LICENSE
Project_Report_of_Food_calorie_estimations_Using_Deep_Learning_And_Computer_Vision.pdf
README.md
custom-yolov4-detector.cfg
darknet_yolov4_obj_names.names
```

`Extra_components/` contains, per the README:
- `Training_Custom_YOLOV4/Train_Custom_YOLOv4_Darknet_Roboflow.ipynb` — trains a custom YOLOv4 (Darknet) detector via Roboflow-hosted data.
- `Training_Custom_YOLOV5/CJ_Roboflow_Train_Custom_YOLOv5.ipynb` — trains a YOLOv5s model, described as "suitable for Android applications."
- `links.txt` — pointers to Google-Drive-hosted trained weights (YOLOv4 and YOLOv5), external to the repo itself.

## B. Existing Functionality (as documented)

1. **Detection only, two parallel implementations:** YOLOv4 (Darknet, config committed at repo root: `custom-yolov4-detector.cfg` + `darknet_yolov4_obj_names.names`) and YOLOv5 (via Ultralytics/Roboflow, weights external).
2. **Data source:** a custom Roboflow project ("fruits--and-thumb-detection"), not a standard published benchmark — small, custom-labelled, detector-only annotations (bounding boxes), no segmentation masks, no nutrition ground truth.
3. **No segmentation stage** — the pipeline is detection-only.
4. **No portion/mass/volume estimation stage** documented.
5. **No multi-nutrient regression** — the project name and report describe *calorie* estimation specifically (the "thumb" class in the dataset name suggests a thumb-as-calibration-object approach for scale, consistent with the classic single-reference-object calorie papers this space is built on, but this is inferred from the class name, not confirmed from code — flag for manual confirmation).
6. **No packaged inference app** in the repo itself (no `app/`, no API, no deployed demo linked from the README).
7. Two research papers and a project report/PPT accompany the code (JETIR-published), useful for citation/attribution text but not for reusable code.

## C. What Can Be Reused

| Component | Reuse value | Notes |
|---|---|---|
| YOLOv4 Darknet `.cfg` + `.names` | Low-medium | Useful as a *reference config* for anchor/layer conventions if a Darknet-family detector is ever wanted; not reusable for our class set (their classes are their custom fruit/thumb set, not our food/ingredient taxonomy). |
| YOLOv5 training notebook | Low-medium | Roboflow-driven training loop is a reasonable pattern reference for `training/` scripts, but the dataset and label map don't transfer. |
| Project report / papers | Medium | Legitimate source for the "prior work" section of our own report, and for the required attribution statement. |
| Overall pipeline concept (detect → estimate calories) | Conceptual only | This is exactly the stage our new project extends; the *idea* of a calibration-object-based scale reference is worth keeping as one candidate portion-estimation strategy (see `DATASET_PLAN.md`), even though the implementation isn't reusable. |

**Net assessment: very little code is directly reusable.** The repo is a detection-only, single-target-nutrient (calorie), custom-small-dataset project. It is valuable as the academic baseline we must cite and improve on, not as a codebase to build inside of.

## D. What Must Be Replaced / Built New

- Segmentation stage (doesn't exist in the reference repo) → SAM/SAM2 + supervised segmentation on UNIMIB2016.
- Classification stage (reference repo has none — it detects+estimates directly) → Food-101 transfer learning.
- Portion/mass/volume estimation (doesn't exist) → Nutrition5k-driven regression.
- Multi-nutrient regression, calories/protein/carbs/fat (reference repo does calories only, via a lookup table implied by their pipeline, not a learned regressor) → new Nutrition5k-trained regression head.
- Modular `src/` package, config-driven, cross-platform paths (reference repo is Colab-notebook-only with implicit paths) → new, see `ARCHITECTURE.md`.
- Multi-food aggregation / final report generation (doesn't exist) → new.

## E. Status of the Parallel Google Drive Work (`food_seg_project`)

Independently of the reference repo, a prior session already produced a Phase 0 audit notebook in the connected Google Drive project folder, executed today, which:
- Verified the Colab/T4 environment.
- Downloaded/verified Food-101 (full archive present) and set up Nutrition5k access (GCS bucket: metadata, overhead RGB, depth).
- Verified UNIMIB2016 is present as a Drive subfolder (segmentation eval).
- Attempted FoodX-251 and an Indian-food Kaggle dataset ("Khana") — **both failed due to Kaggle access gating** and were explicitly dropped as a project dataset. Khana's labels/taxonomy files (non-image metadata) were confirmed downloadable, but the image set itself requires manual, gated download — this is why it's listed as excluded below despite partial availability.
- Verified a SAM ViT-B checkpoint.
- Drafted (but has **not yet executed against real data**) a Stage 1 SAM-based detection/segmentation module.

**Decision carried forward per your instruction:** the dataset choices already made in that notebook are kept as-is rather than re-derived here (see `DATASET_PLAN.md` for the resulting, reconciled plan and the one explicit conflict this creates against the originally-approved dataset list).

## F. Risks / Limitations Carried Into This Audit

- Reference repo's actual notebook internals were not opened in this session (network restriction) — verify manually before assuming any cell-level detail beyond what's stated here.
- Reference repo's dataset (Roboflow fruit/thumb set) has no overlap with our food-nutrition taxonomy — expect **zero transfer** from its trained weights.
- `Khana` (Indian food) is in the Drive project's classification pool despite not being on the originally-approved dataset list, and despite the current brief's explicit "no unavailable Indian food datasets" rule — flagged, not silently resolved (see `DATASET_PLAN.md`).
- ECUSTFD, Recipe1M+, Vireo Food-172, and MenuMatch — all on the originally-approved list — have **not** been touched by the Drive work yet. They remain available for later stages (ingredient understanding, reference-object portion estimation, auxiliary evaluation) but are not yet integrated.
