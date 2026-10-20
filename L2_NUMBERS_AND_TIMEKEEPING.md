# Task L2: Numbers Check & Timekeeping Report

**Task:** L2 (Lenny) — Numbers Check and Timekeeping
**Date:** October 20, 2026
**Commit checked:** `0fb0179` (`origin/main`)
**Deliverable reference:** [PLAN.md §7, §8, §10](PLAN.md)

---

## 1. Executive Summary

Task L2 consists of two non-engineering verification responsibilities defined in [PLAN.md](PLAN.md):

1. **Numbers Consistency Check:** Ensure every number reported in [README.md](README.md) and on the presentation slides ([slides/museum_guide.pptx](slides/museum_guide.pptx)) strictly matches the outputs in `results/*.json` or the notebooks.
2. **Rehearsal Timekeeping:** Provide an exact time budget and milestone schedule for the 10-minute presentation and 5-minute Q&A drill ahead of Day 6 rehearsals.

### Summary of Status

- **README.md Numbers:** **100 % MATCH.** All test accuracy metrics, 95 % Wilson confidence intervals, macro-F1 scores, CPU latencies (1 thread), model sizes, parameter counts, and decision criteria figures match `results/*.json` and `notebooks/07_comparison.ipynb`.
- **Slide Numbers (Slides 1–7 & 12):** **100 % MATCH.** Architectures, parameter counts, loss assertions ($\ln 5 \approx 1.61$), dead ReLUs, memorisation on shuffled labels (99.3 %), ablation train/val accuracies, and final FFN numbers align with notebooks `01`, `02`, and `03`.
- **Slide Gaps (Slides 8–11):** **ACTION REQUIRED.** Slides 8, 9, 10, and 11 in [slides/museum_guide.pptx](slides/museum_guide.pptx) currently contain placeholder titles and text from the initial deck assembly. The final test numbers and figures generated in PR #17 need to be placed into these slides before Day 6 rehearsals.
- **Timekeeping Schedule:** A slide-by-slide 10-minute rehearsal schedule (Part 1: 4m45, Part 2: 4m15, Demo/Conclusion: 1m00) is established below, with speaker roles set to TBD to remain flexible.

---

## 2. Numbers Check: README.md vs. `results/*.json`

All test metrics were evaluated on the 225 test images in `notebooks/07_comparison.ipynb` with `FINAL_TEST=1`.


| Model                                       | Metric           | Value in`README.md` | Exact Value in`results/*.json` |  Match?  |
| ------------------------------------------- | ---------------- | ------------------- | ------------------------------ | :-------: |
| **Classical** (`hsv+hog`, RBF-SVM)          | Test Accuracy    | 47.6 %              | 0.475555... (47.56 %)          | **MATCH** |
|                                             | 95 % Wilson CI   | [41.1, 54.1]        | [0.41126, 0.54067]             | **MATCH** |
|                                             | Macro-F1         | 0.475               | 0.475017...                    | **MATCH** |
|                                             | 1-Thread Latency | 2.98 ms             | 2.9787 ms                      | **MATCH** |
|                                             | Disk Size        | 16.5 MB             | 16.4858 MB                     | **MATCH** |
| **FFN** (32 px, AdamW)                      | Test Accuracy    | 40.4 %              | 0.404444... (40.44 %)          | **MATCH** |
|                                             | 95 % Wilson CI   | [34.2, 47.0]        | [0.34244, 0.46966]             | **MATCH** |
|                                             | Macro-F1         | 0.395               | 0.395358...                    | **MATCH** |
|                                             | 1-Thread Latency | 0.09 ms             | 0.0852 ms                      | **MATCH** |
|                                             | Disk Size        | 3.3 MB              | 3.2840 MB                      | **MATCH** |
| **ResNet18 Frozen** (ImageNet, Linear Head) | Test Accuracy    | 79.6 %              | 0.795556... (79.56 %)          | **MATCH** |
|                                             | 95 % Wilson CI   | [73.8, 84.3]        | [0.73811, 0.84308]             | **MATCH** |
|                                             | Macro-F1         | 0.795               | 0.794738...                    | **MATCH** |
|                                             | 1-Thread Latency | 19.1 ms             | 19.0816 ms                     | **MATCH** |
|                                             | Disk Size        | 44.8 MB             | 44.7955 MB                     | **MATCH** |
| **ResNet18 Fine-tuned** (layer4 + fc)       | Test Accuracy    | **84.0 %**          | 0.840000... (84.00 %)          | **MATCH** |
|                                             | 95 % Wilson CI   | [78.6, 88.2]        | [0.78645, 0.88213]             | **MATCH** |
|                                             | Macro-F1         | 0.838               | 0.838456...                    | **MATCH** |
|                                             | 1-Thread Latency | 16.6 ms             | 16.6200 ms                     | **MATCH** |
|                                             | Disk Size        | 44.8 MB             | 44.7957 MB                     | **MATCH** |

### Additional Text Assertions in README:

- **Decision criteria analysis:** *"no model passes all three. The fine-tuned ResNet18 comes closest."* — Confirmed by Cell 10 of `07_comparison.ipynb`.
- **Significance:** *"significantly better than the classical model (McNemar p < 0.001)"* — Exact binomial $p = 0.00000$ (Cell 8 of `07_comparison.ipynb`).
- **Accuracy gap:** *"misses the 85 % accuracy bar by 1.0 point (189 of 225 right, where 85 % needs 192)"* — $189 / 225 = 84.00\,\%$; $192 / 225 = 85.33\,\%$; $191 / 225 = 84.89\,\% < 85\,\%$. Exact arithmetic match.
- **Latency & Training Time:** *"passes latency ($\le 40$ ms) and footprint ($\le 100$ MB, trained in 78 s)"* — 16.6 ms $\le$ 40 ms; 44.8 MB $\le$ 100 MB; training time is 77.52 s ($\approx 78$ s in `results/resnet_finetune.json`).

---

## 3. Numbers Check: Presentation Slides (`museum_guide.pptx`)

### Slides 1–7 & Slide 12: Problem Brief, Data, FFN & Demo


| Slide #      | Claim / Element    | Number on Slide                                                                                                                                                                                                                                                                                     | Source in Code / Notebooks                         |  Status  |
| ------------ | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | :-------: |
| **Slide 2**  | Pre-fixed Criteria | $\ge 85\,\%$ test acc, McNemar $p < 0.05$<br>$\le 40$ ms ($200\text{ ms} / 5$ safety factor)<br>$\le 100$ MB disk, $< 1$ h laptop training                                                                                                                                                          | [PLAN.md §4](PLAN.md)                             | **MATCH** |
| **Slide 3**  | Data split         | 1,500 total, 5 styles (300/style)<br>1,050 train / 225 val / 225 test (210/45/45)                                                                                                                                                                                                                   | `data/split.csv`, `01_data.ipynb`                  | **MATCH** |
|              | Metadata limits    | 85 to 100 % unknown artist<br>Sculpture photos: ~7 % Impressionism, ~18 % Pop Art                                                                                                                                                                                                                   | `01_data.ipynb`                                    | **MATCH** |
|              | Test uncertainty   | 225 test images$\rightarrow \pm 5$ percentage points CI                                                                                                                                                                                                                                             | Wilson CI calculation                              | **MATCH** |
| **Slide 4**  | FFN Architecture   | 32×32 (3,072 inputs), 256, 128, 5 outputs                                                                                                                                                                                                                                                          | `src/museum/models.py`                             | **MATCH** |
|              | Parameters         | 820,229 parameters (96 % in layer 1)<br>Initial loss $1.61 = \ln 5$<br>128 px: 15× more parameters (12.62 M)                                                                                                                                                                                       | `02_ffn.ipynb`, `03_diagnosis.ipynb`               | **MATCH** |
|              | Chart 1 (Val acc)  | 32 px: 43.0 %, 64 px: 42.5 %, 128 px: 41.2 %                                                                                                                                                                                                                                                        | `02_ffn.ipynb` (mean over 3 seeds)                 | **MATCH** |
|              | Chart 2 (Params)   | 32 px: 0.82 M, 64 px: 3.18 M, 128 px: 12.62 M                                                                                                                                                                                                                                                       | Exact formula ($3072\times 256$, etc.)             | **MATCH** |
| **Slide 5**  | Monitoring         | Loss$1.61 = \ln 5$<br>7 steps to memorise batch of 32<br>LR 1e-4 best (44 % val acc)<br>96 % dead ReLUs at LR 0.01                                                                                                                                                                                  | `03_diagnosis.ipynb`                               | **MATCH** |
| **Slide 6**  | Diagnosis          | 128 px: 99.7 % train vs 38 % val acc<br>Confidence on mistakes: 0.29 $\rightarrow$ 0.74<br>Shuffled labels: 99.3 % train acc                                                                                                                                                                        | `03_diagnosis.ipynb`                               | **MATCH** |
| **Slide 7**  | Fixes & Ablation   | Train-val gap: 62$\rightarrow$ 12 points<br>Chart 3 (train/val %):<br>Baseline: 99.1 / 36.9<br>+32px: 99.5 / 38.1<br>+dropout: 91.7 / 38.4<br>+weight decay: 91.8 / 39.3<br>+augmentation: 61.3 / 43.7<br>+early stopping: 56.7 / 44.7<br>Final candidate: 45.8 % val acc (seed 0), 0.08 ms, 3.3 MB | `03_diagnosis.ipynb` (Cell 18 & 24)                | **MATCH** |
| **Slide 12** | Guide Demo         | 80 % visitors answered, 90 % accuracy (frozen ResNet); with fine-tuned ResNet, coverage rises to 95 % at 90 % accuracy.<br>FFN would stay silent 94 % of the time (coverage 6.2 %)                                                                                                                  | Validation thresholding (`08_guide.ipynb`, Cell 4) | **MATCH** |

### Slides 8–11: Classical Baseline, ResNet & Final Test Comparison — Status & Action Items

In the committed PowerPoint (`slides/museum_guide.pptx`), Slides 8, 9, 10, and 11 currently contain placeholder titles and text:

- **Slide 8:** Title `"Classical baseline"`, placeholder body text.
  - *Data to insert:* HSV + HOG feature pipeline, 2.06 M learned values, 16.5 MB, val acc 55.1 %, latency 2.98 ms (`figures/classical_features.png`, `figures/classical_val_confusion.png`).
- **Slide 9:** Title `"Pretrained ResNet18: frozen and fine-tuned"`, placeholder body text.
  - *Data to insert:* Frozen head (84.0 % val, 19.1 ms, 2,565 trainable params) vs. fine-tuned (86.2 % val, 16.6 ms, 8.4 M trainable params), scratch control (52.4 %), Grad-CAM interpretation (`figures/resnet_gradcam.png`).
- **Slide 10:** Title `"The four models on the test set"`, placeholder body text.
  - *Data to insert:* The official 4-model comparison table from README, accuracy-latency Pareto plot (`figures/comparison_pareto_test.png`), McNemar matrix, and test confusion matrices (`figures/comparison_confusion_resnet_finetune_test.png`).
- **Slide 11:** Title `"Recommendation and future work"`, placeholder body text.
  - *Data to insert:* Decision against the pre-fixed criteria (84.0 % vs 85 % threshold), recommendation for a supervised kiosk pilot with abstention, future work on perturbation robustness (perspective, blur) and data collection.

---

## 4. Rehearsal Timekeeping (10-Minute Presentation Budget)

According to [PLAN.md §1 & §8](PLAN.md), the presentation is strictly budgeted for **10 minutes total (600 seconds)** followed by **5 minutes of Q&A**. The team timekeeper tracks timestamps during the Day 6 rehearsals.

### Target Timing Schedule


| Segment              | Slide # | Title & Topic                                    |  | Target Time | Elapsed Target | Rehearsal Checkpoint                          |
| -------------------- | :-----: | ------------------------------------------------ | :-: | :---------: | :------------: | --------------------------------------------- |
| **Intro & Brief**    |  **1**  | Title & Project Overview                         |  |    0:30    |    **0:30**    | Start prompt                                  |
|                      |  **2**  | The Brief & 3 Pre-Fixed Decision Criteria        |  |    0:45    |    **1:15**    | Check 85 % / 40 ms / 100 MB stated clearly    |
| **Data**             |  **3**  | WikiArt Subset, Split (70/15/15) & Limits        |  |    0:45    |    **2:00**    | Must mention unknown artists & sculptures     |
| **FFN & Monitoring** |  **4**  | Architecture, Parameters & 32 px Choice          |  |    0:45    |    **2:45**    | Note 96 % params in layer 1                   |
|                      |  **5**  | Training Monitoring & Gradient Health            |  |    0:45    |    **3:30**    | Emphasize initial loss$\ln 5$ & dead ReLUs    |
| **Diagnosis**        |  **6**  | Failure Diagnosis: Overfitting & Shuffled Labels |  |    0:45    |    **4:15**    | Key message: memorisation vs. learning        |
|                      |  **7**  | Step-by-Step Fixes (Ablation Table)              |  |    0:30    |    **4:45**    | Transition to Part 2 at ~4m45s                |
| **Classical**        |  **8**  | Classical Baseline (HSV + HOG, RBF-SVM)          |  |    0:50    |    **5:35**    | Explain why colour histograms act as baseline |
| **ResNet Transfer**  |  **9**  | ResNet18: Frozen Head vs. Fine-Tuning            |  |    1:00    |    **6:35**    | Native 128 px vs 224 px, Grad-CAM             |
| **Test Comparison**  | **10** | 4-Model Test Scoring, Pareto & McNemar           |  |    1:15    |    **7:50**    | Emphasize test set used only once             |
| **Recommendation**   | **11** | Criteria Verdict & Supervised Pilot Proposal     |  |    0:55    |    **8:45**    | Transparent conclusion: 84 % misses 85 %      |
| **Demo & Wrap-up**   | **12** | Guide Demo: Confidence & Abstention              |  |    0:50    |    **9:35**    | Show 80 % coverage / 90 % accuracy            |
|                      | **13** | Conclusion & Transition to Q&A                   |  |    0:25    |   **10:00**   | Total speech ends at 10m00s                   |
| **Q&A Drill**        |   —   | Jury Questions & Defense Drill                   |  |    5:00    |   **15:00**   | Cross-defense practice (PLAN.md §7)          |

### 

---

## 5. Potential Oral Questions for the Q&A Drill (5 min)

From [PLAN.md §7 ("Q&A prep sheet")](PLAN.md):

1. **Why do FFNs struggle on image classification, and why does transfer learning solve it?**
   *Answer:* FFNs treat images as flat vectors, losing spatial 2D locality and translation invariance. The first layer consumes 96 % of weights on raw pixels. ResNet uses convolution to extract local edges and textures invariant to position, pretrained on 1.2M ImageNet images.
2. **Why was the test set scored strictly once?**
   *Answer:* Tuning hyperparameters or choosing models based on the test set causes test-set leakage (overfitting to the evaluation sample). 225 test images yield an interval of $\pm 5$ points. Rehearsal mode evaluated on validation only.
3. **Did the dataset have leakage across splits?**
   *Answer:* We performed an exhaustive nearest-neighbour cosine similarity check on 512-d ResNet embeddings (`05_resnet_frozen.ipynb`). Pairs above 0.95 were checked; none crossed between train and test.
4. **Why the ×5 safety factor on latency?**
   *Answer:* Benchmark was measured on a single CPU thread of an Apple Silicon M-series chip. Museum kiosks use low-power embedded processors (e.g. Celeron or older x86), which are roughly 4–5× slower. 16.6 ms on our laptop corresponds to $\approx 83$ ms on the kiosk, well within the 200 ms budget.
