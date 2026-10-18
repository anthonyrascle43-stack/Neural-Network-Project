# Museum Guide — Final Project Plan

**Team:** Anthony · Antoine · Lenny
**Repo:** https://github.com/anthonyrascle43-stack/Neural-Network-Project
**Deadline:** under a week; Day 1 = first working day, Day 6 = rehearsal

---

## 1. The brief in one paragraph

A (fictional) French fine-arts museum wants a guide on phones and gallery kiosks that recognises a painting's **style** and tells the story of that movement. The director also wants a clear recommendation: kiosks have modest CPUs, the team is small and the budget is limited. **Is a neural network worth it in production, or would a classical model do?**

Required deliverables:
1. A feed-forward network (FFN) in PyTorch for artwork classification.
2. Training monitoring and failure diagnosis.
3. A pretrained CNN (ResNet) for image classification.
4. Code on GitHub with a clear Git history, plus a 10-min PowerPoint and 5-min Q&A.

The course expects networks built from PyTorch primitives (`nn.Module`, `nn.Linear` and a hand-written training loop; no Lightning, skorch or timm), with every number measured and every claim backed by evidence.

**Out of scope:** the optional natural-language search (future work only), genre classification, a real mobile app, other architectures.

## 2. Data

- WikiArt subset: 5 styles × 300 images = 1,500 RGB JPEGs at 128×128.
- Built with `scripts/make_wikiart_subset.py` (seed 0, pinned dataset revision).
- Class indices: Baroque=0, Cubism=1, Impressionism=2, Pop_Art=3, Ukiyo_e=4.
- **Images are never committed.** Each of us generates them with the script, or Anthony shares a private 13 MB zip.

**Known limitations** (they go on the data slide):
- 85–100 % of images per class are "Unknown Artist" and 60–100 % are "Unknown Genre". So we can't split by artist, and the genre task isn't feasible.
- Impressionism and Pop_Art each come from only 1–2 source files, so there may be near-duplicates across splits. Antoine checks this (task N3).
- 225 test images give accuracy intervals of about ±5 points.

## 3. Rules every model follows

- **Split:** stratified 70/15/15 → 1,050 / 225 / 225 (210/45/45 per class), seed 0, written once to `data/split.csv`.
- **All tuning and model selection on validation.** Each family freezes **one** candidate on validation: classical, FFN, frozen ResNet, fine-tuned ResNet.
- **The 4 candidates are scored on the test set once, together, in `07_comparison.ipynb`.**
- **Reporting:** test accuracy with Wilson 95 % CI, macro-F1, confusion matrix, pairwise McNemar test. Neural-net ablations run 3 seeds and report mean ± std on validation.
- **Each candidate writes `results/<model>.json`:** config, val acc, test acc, macro-F1, parameter count, size (MB), CPU latency (1 and 2 threads, median + p95), training time.
- **Latency:** measured end-to-end on CPU (preprocessing and feature extraction included), batch 1, 10 warm-up + 100 timed runs.
- Training uses MPS or CUDA when available; benchmarks are always on CPU.
- Course conventions: `torch.manual_seed`, a `FAST_RUN` env flag for quick re-runs, `assert` checks after key steps.

## 4. Decision criteria for the director (fixed before any results)

1. **Accuracy:** test top-1 ≥ 85 % (point estimate, CI reported), **and** significantly better than the best classical model (McNemar p < 0.05).
2. **Latency:** ≤ 200 ms per image on a kiosk.
   - Both our laptops are Apple Silicon, far faster than a kiosk CPU. So we measure on 1 CPU thread and assume a **×5 safety factor**: measured ≤ 40 ms passes.
   - The ×5 factor is an assumption, and the slides say so.
3. **Footprint:** model ≤ 100 MB on disk; trainable on a laptop in < 1 hour.

**Recommendation** = the cheapest model meeting all three. If none does, we say which criterion fails and by how much.

## 5. Repo layout

```
README.md                 setup, data, how to reproduce, results table
PLAN.md                   this file
requirements.txt          torch, torchvision, scikit-learn, scikit-image, scipy, numpy, pandas, matplotlib, pillow, jupyter
requirements-data.txt     only needed to rebuild the dataset (datasets, pillow)
.gitignore                images, zips, .venv, *.pt, cached features, caches, __pycache__, .ipynb_checkpoints
scripts/                  make_wikiart_subset.py, validate_wikiart_subset.py, make_split.py
data/wikiart_style5/      metadata.csv, subset_info.json (committed); images (ignored)
data/split.csv            train/val/test assignment (committed)
src/museum/
  data.py                 Anthony — split, Dataset/DataLoaders, normalisation, augmentation
  train.py                Anthony — train_one_epoch / evaluate / fit (+ early stopping, grad norms)
  models.py               StyleMLP (Anthony) · build_resnet18 (Antoine)
  diagnostics.py          Anthony — grad norms, dead ReLUs, activation stds, init, LR sweep, plots
  evaluation.py           Antoine — metrics, Wilson CI, McNemar, latency benchmark, size, save_result
  guide.py, stories.py    Anthony — demo: image → style + confidence + story, abstains when unsure
notebooks/
  01_data.ipynb             Anthony
  02_ffn.ipynb              Anthony
  03_diagnosis.ipynb        Anthony
  04_classical.ipynb        Antoine
  05_resnet_frozen.ipynb    Antoine
  06_resnet_finetune.ipynb  Antoine
  07_comparison.ipynb       Antoine (recommendation written together)
results/                  <model>.json + figures/*.png (committed; slides and README use only these)
slides/                   final .pptx
```

Code we port from the course (Session 4 TP05 notebook): `evaluate`, `train_one_epoch`, `fit`, `make_mlp`, `layer_grad_norms`, the dead-ReLU fraction, `init_weights`, `activation_stds`, train-only standardisation, and the confusion matrix. The LR-curve plot style comes from Session 2 TP02.

## 6. Shared interfaces (agreed on Day 1 so we can work in parallel)

```python
# data.py (Anthony)
load_split() -> pd.DataFrame                       # path, label (name), label_idx, split
get_loaders(img_size=128, batch_size=64, augment=False,
            normalize="train" | "imagenet", seed=0) -> (train_dl, val_dl, test_dl, classes)
get_arrays(img_size, split) -> (X: np.ndarray uint8 [N,H,W,3], y)   # for sklearn

# train.py (Anthony)
fit(model, train_dl, val_dl, loss_fn, optimizer, n_epochs, device,
    patience=None, track_grads=False) -> history
    # history keys: train_loss, train_acc, val_loss, val_acc, grad_norms, epoch_time
    # restores the best-validation weights when patience is set
evaluate(model, dl, loss_fn, device) -> (loss, acc)
predict(model, dl, device) -> (y_true, y_pred, probs)

# evaluation.py (Antoine)
wilson_ci(k, n, z=1.96) -> (low, high)
mcnemar(y_true, pred_a, pred_b) -> p_value         # exact binomial
benchmark_latency(predict_one, sample, n_threads, warmup=10, runs=100) -> {"median_ms", "p95_ms"}
model_size_mb(obj_or_path) -> float
plot_confusion(y_true, y_pred, classes, path)
save_result(name, dict) -> writes results/<name>.json (checks required fields)
```

## 7. Task split

Anthony and Antoine each own one or more of the three required deliverables, build their own slides and commit their own work. Lenny runs the reproducibility and consistency checks.

### Anthony — FFN, monitoring, failure diagnosis (deliverables 1 & 2) + data/training core + demo

| ID | Task | Output |
|---|---|---|
| A1 | Repo setup: `.gitignore`, README, `PLAN.md`, `requirements.txt`, data scripts + metadata, folder skeleton; add Antoine and Lenny as collaborators | first commit on `main`, clonable |
| A2 | `make_split.py` → `data/split.csv`; `data.py` (loaders, train-only normalisation, augmentation: h-flip + light crop, **no hue jitter** because colour is a style cue) | `feat/data` PR |
| A3 | `train.py`: port and extend `fit` (train acc, early stopping, best-weights restore, grad norms, wall time) | `feat/train-loop` PR |
| A4 | `01_data.ipynb`: class grid, stats, split asserts, limitations | notebook |
| A5 | `StyleMLP` + `02_ffn.ipynb`: He init, last layer ×0.01; parameter count by hand; input-size sweep 32/64/128 px | FFN baseline curves |
| A6 | `diagnostics.py` + `03_diagnosis.ipynb` **must-haves**: initial loss ≈ ln 5 (assert), overfit one batch, LR sweep, overfitting diagnosis (large FFN at 128 px) | figures + "what we see / why / fix" cells |
| A7 | Fix ablation (3 seeds, cumulative, order-dependence stated): baseline → smaller input → dropout → weight decay → augmentation → early stopping. Freeze the FFN candidate. | ablation table, `results/ffn.json` (val part) |
| A8 | FFN error analysis: confusion matrix, per-class recall, most-confident mistakes | figure |
| A9 | **Should-haves:** shuffled-label control; gradient health (deep sigmoid vs ReLU+He, dead ReLUs at high LR, activation std by init) | figures |
| A10 | `guide.py` + `stories.py` (5 movement texts), abstention threshold chosen on validation; demo screenshot | demo |
| A11 | Final README (setup, reproduce, results table from `results/`) | README |
| A12 | Slides: problem & criteria, data & limitations, FFN, monitoring, diagnosis, ablation (~5 min) | 6 slides |

### Antoine — Pretrained ResNet (deliverable 3) + classical baseline + evaluation + comparison

| ID | Task | Output |
|---|---|---|
| N1 | `evaluation.py`: Wilson CI, McNemar, latency benchmark, model size, confusion plot, `save_result` | `feat/evaluation` PR |
| N2 | `04_classical.ipynb`: HSV colour histogram + HOG (64 px grayscale); raw 32 px pixels as a control. LogisticRegression / linear SVM / RBF SVC, C tuned on val, latency including feature extraction. Freeze the classical candidate. | `results/classical.json` (val part) |
| N3 | `build_resnet18` + `05_resnet_frozen.ipynb`: ImageNet weights, extract 512-d features once; `nn.Linear` head (shared `fit`) vs sklearn LogisticRegression; **near-duplicate check** across splits (cosine > 0.95, inspect, report). Freeze the frozen-ResNet candidate. | `results/resnet_frozen.json` (val part) |
| N4 | `06_resnet_finetune.ipynb`: unfreeze layer4 + head, LR 1e-4, augmentation, early stopping, monitoring curves. 128 px (native) vs 224 px (upsampled to ImageNet scale; adds no detail), accuracy vs latency. Freeze the fine-tuned candidate. | `results/resnet_finetune.json` (val part) |
| N5 | ResNet error analysis: confusion matrix, confident mistakes | figure |
| N6 | `07_comparison.ipynb`: one-time test evaluation of the 4 candidates, results table, McNemar matrix, accuracy-vs-latency Pareto plot, check against the criteria | final table + figures |
| N7 | **Should-haves:** camera-robustness check (perspective / blur / brightness / crop on test images, accuracy drop per model); ResNet18 from random init; Grad-CAM for one slide | figures |
| N8 | Slides: classical baseline, ResNet frozen + fine-tuned, comparison/Pareto, recommendation & future work (~5 min) | 5–6 slides |

### Lenny — reproducibility and consistency checks

Checks that need no data or ML engineering. Findings go into GitHub issues, and the owner of the file fixes them.

| ID | Task | Output |
|---|---|---|
| L1 | **Fresh-clone check.** Follow the README on his own laptop, from `git clone` to `pytest`, and open an issue for every step that fails or is unclear. Second pass on Day 5: run every notebook with `FAST_RUN=1` (definition of done, §10). | GitHub issues |
| L2 | **Numbers check and timekeeping.** Check that every number on the slides and in the README matches `results/*.json` or a notebook output, and time each part at the rehearsals. | list of mismatches, timings |

### Together
- **Day 1 kickoff (30 min):** agree on the interfaces in §6 and on the decision criteria in §4.
- Recommendation text, written together from `07_comparison` results.
- Deck assembly, two full rehearsals, and a Q&A drill where each of us answers questions on the other's part.
- Q&A prep sheet:
  - why FFNs struggle on images, and why transfer learning works
  - why the test set was used only once; CI width and McNemar
  - the leakage check
  - where and how latency was measured, and the ×5 assumption
- Peer review: Anthony and Antoine review each other's PRs before merging.

## 8. Schedule

| Day | Anthony | Antoine | Lenny |
|---|---|---|---|
| 1 | A1, A2, A3 (push `data.py` + `split.csv` by midday) | Env + data; N1; start N2 features | GitHub access |
| 2 | A4, A5 | N2 done (candidate frozen); start N3 | L1 first pass (setup, `pytest`) |
| 3 | A6, A7 | N3 done (near-dup check reported); N4 | – |
| 4 | A8, A9 (if time), A10 | N4, N5; N7 (if time) | – |
| 5 | A11; slides A12 | N6 (test run once, together); slides N8 | L1 second pass (notebooks); L2 numbers check |
| 6 | Rehearse ×2, Q&A drill, tag `v1.0-submission` | Same | L2 timekeeping at rehearsals |

Dependencies: Antoine needs `data.py`, `split.csv` and `fit` (A2, A3) on Day 1. Anthony needs `evaluation.py` (N1) by Day 2. Lenny needs every notebook merged by Day 5 morning (L1) and the final slides by Day 5 evening (L2).

## 9. Git workflow

- One feature branch per task (`feat/data`, `feat/ffn`, `feat/classical`, `feat/resnet-frozen`, …), merged into `main` through a PR that the other person reviews.
- Small, descriptive commits written by whoever did the work.
- One owner per notebook, so no merge conflicts. Notebook outputs are kept (they are our evidence).
- No author names other than Anthony, Antoine and Lenny; no co-author trailers; no tool or generator credits in commits, PRs, code or docs.
- Images and checkpoints never committed. Tag `v1.0-submission` at the end.
- CI (GitHub Actions, `.github/workflows/ci.yml`) runs `ruff check .` and `pytest` on every PR and on `main`. A PR is merged only when CI is green and the other person has approved it.
- New shared code in `src/museum/` comes with a test in `tests/` (on synthetic data: CI has no images).

## 10. Definition of done

- `python scripts/validate_wikiart_subset.py data/wikiart_style5` passes.
- Notebook asserts pass:
  - split 1050/225/225 with 210/45/45 per class and no overlap
  - initial loss within 0.05 of 1.609
  - one-batch overfit reaches loss < 0.05
  - every `results/*.json` has all required fields
- Fresh-clone check: clone, create the venv, add images, run every notebook with `FAST_RUN=1` (`jupyter nbconvert --execute`). All succeed.
- Every number on the slides and in the README comes from `results/*.json` or a notebook output.
- `git log --graph` shows a readable per-feature history with both of us as authors.
- CI is green on `main` at the `v1.0-submission` tag.
