# Detection Dataset Quality Audit

**Date:** 2026-09-30 22:35:09  
**Dataset Location:** `data/fixture_detection_p2/`  
**Splits:** 120 train / 40 validation / 40 test  

## 1. Summary Statistics

- **Total Bounding Boxes:** 385
- **Corrupt Images:** 0
- **Small Objects (<5% dimension):** 0 (0.0%)
- **Class Imbalance Ratio:** 2.05x
- **Split Leakage:** 0% (deterministic disjoint partitions)
- **Duplicates:** 0 (unique coordinate generator)

## 2. Class Distribution

| Class Name | ID | Instance Count |
|---|---|---|
| banana | 0 | 30 |
| apple | 1 | 43 |
| sandwich | 2 | 30 |
| orange | 3 | 25 |
| broccoli | 4 | 33 |
| carrot | 5 | 31 |
| hot_dog | 6 | 21 |
| pizza | 7 | 37 |
| donut | 8 | 35 |
| cake | 9 | 31 |
| cup | 10 | 29 |
| bowl | 11 | 40 |

## 3. Findings

No format errors, coordinate boundary violations, or corrupted files discovered.
