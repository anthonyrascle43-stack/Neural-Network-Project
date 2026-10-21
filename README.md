# Museum Guide — Artwork Style Recognition

[![CI](https://github.com/anthonyrascle43-stack/Neural-Network-Project/actions/workflows/ci.yml/badge.svg)](https://github.com/anthonyrascle43-stack/Neural-Network-Project/actions/workflows/ci.yml)

A (fictional) mid-sized French fine-arts museum wants a guide on visitors' phones and gallery kiosks: point the camera at a painting, recognise its **style**, and tell the story of that movement. We build, diagnose and defend the neural-network components of that guide, and answer the director's question:

> Kiosks run on modest CPUs, the team is small and the budget is limited. **Is a neural network worth it in production, or would a simpler classical model do the job?**

Final project of the Neural Networks & Deep Learning course. Networks are built from PyTorch primitives (`nn.Module`, `nn.Linear`, hand-written training loops), and every number we report is measured.

## Team

| Member | Owns |
|---|---|
| **Anthony** | Feed-forward network, training monitoring & failure diagnosis, data & training code, guide demo |
| **Antoine** | Pretrained ResNet, classical baseline, evaluation tools, final comparison |
| **Lenny** | Reproducibility checks (fresh clone, notebooks) and consistency of the reported numbers |

The full task split, schedule, shared interfaces and rules are in **[PLAN.md](PLAN.md)**.

## Deliverables

- [x] Feed-forward network in PyTorch for artwork style classification (`02_ffn`, `03_diagnosis`)
- [x] Training monitoring and failure diagnosis (`03_diagnosis`)
- [x] Pretrained CNN (ResNet18) for style classification (`05_resnet_frozen`, `06_resnet_finetune`)
- [x] Classical baseline (colour histograms + HOG, scikit-learn) for comparison (`04_classical`)
- [x] Recommendation for the museum director (accuracy, CPU latency, footprint) (`07_comparison`, [Results](#results))
- [x] A museum guide demo: photo → style, confidence and the movement's story, or "not sure" (`08_guide`)
- [x] Slides for the 10-minute presentation (`slides/museum_guide.pptx`)
- [x] Optional: natural-language search ("show me a stormy sea") with a short chat reply (`09_search`). It uses a pretrained CLIP and stays outside the guide, because it is far over the kiosk's 100 MB budget.

## Quick start

```bash
git clone https://github.com/anthonyrascle43-stack/Neural-Network-Project.git
cd Neural-Network-Project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.11–3.13. Then get the images (next section) and check them.

## Getting the data

The images are **not** in the repository: WikiArt images belong to their rights holders (licence unknown), so they are used for this coursework only and never committed. What *is* versioned is `data/wikiart_style5/metadata.csv` and `data/wikiart_style5/subset_info.json`, so we all work on exactly the same 1,500 images.

**Option A — zip from Anthony (fastest).** Put `wikiart_style5_images.zip` at the repo root, then:

```bash
unzip wikiart_style5_images.zip -d data/
rm wikiart_style5_images.zip
```

**Option B — rebuild from the source dataset.** Streams about 4.2 GB from the Hugging Face Hub (10–40 min on a home connection):

```bash
pip install -r requirements-data.txt
export HF_HOME="$PWD/.cache/huggingface"

python scripts/make_wikiart_subset.py --label style \
  --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
  --per-class 300 --size 128 --seed 0 \
  --strategy fewest-shards --max-per-artist 50 \
  --out data/wikiart_style5 \
  --index-cache .cache/label_index.json
```

Add `--dry-run` first to see the plan without downloading. The script rewrites `subset_info.json` with a new timestamp; restore the committed version with `git checkout data/wikiart_style5/subset_info.json`.

**Either way, validate:**

```bash
python scripts/validate_wikiart_subset.py data/wikiart_style5
```

Expected: `"total": 1500`, 300 images per class, `"revision": "d559852d…"` and `"all_images_decoded": true`. `git status` must not list any `.jpg`.

## Using the code

The shared code lives in `src/museum/`. From a notebook in `notebooks/`:

```python
import sys
sys.path.insert(0, "../src")
from museum.data import get_loaders, get_arrays

train_dl, val_dl, test_dl, classes = get_loaders(img_size=64, augment=True)   # PyTorch
X_train, y_train = get_arrays(img_size=64, split="train")                      # scikit-learn, uint8 [N, H, W, 3]
```

`get_loaders(normalize="train")` standardises with per-channel statistics of the training images only; use `normalize="imagenet"` for pretrained ResNets.

**Trying the guide** on any photo (it needs the checkpoints, see [Reproducing the results](#reproducing-the-results)):

```bash
PYTHONPATH=src python -m museum.guide path/to/photo.jpg               # fine-tuned ResNet18
PYTHONPATH=src python -m museum.guide path/to/photo.jpg --model ffn   # our feed-forward network
```

The photo is turned upright, cropped to a square in the middle and resized like the dataset images. The guide prints the style, its confidence and the story of the movement (`src/museum/stories.md`), or a "not sure" message when the confidence is below the threshold chosen on validation. With the fine-tuned ResNet18 it answers 95 % of the validation images and is right 90 % of the time (`08_guide.ipynb`).

**Searching the collection with words** (the optional part, `09_search.ipynb`). It needs the optional requirements, and downloads CLIP (about 600 MB) and a small language model (about 1 GB) the first time:

```bash
pip install -r requirements-search.txt
PYTHONPATH=src python -m museum.chat "a stormy sea"          # the 6 closest works and a reply from a template
PYTHONPATH=src python -m museum.chat "a stormy sea" --llm    # the reply written by the language model, then checked
PYTHONPATH=src python -m museum.chat                         # several questions in a row
```

- **What is searched:** the train and validation images only (the test set stays out).
- **The search model:** CLIP recognises our five styles with no training (84–88 % on validation, depending on the wording), but its text part alone is 254 MB.
- **The reply:** by default, a template filled with what we know about the works found. The language model's replies read better, but 11 of 16 had a mistake. So with `--llm`, a reply is only shown if our check finds nothing that the facts don't say; otherwise the template is shown.

## Reproducing the results

Run the notebooks in this order, top to bottom. Times are from a fresh clone on an Apple Silicon laptop (training on the Apple GPU through MPS):

| Notebook | What it does | Writes | Time |
|---|---|---|---|
| `01_data` | data exploration, split checks | figures | ~10 s |
| `02_ffn` | feed-forward network, input-size sweep | figures | ~1 min |
| `03_diagnosis` | monitoring, diagnosis, ablation, the FFN candidate | `checkpoints/ffn.pt`, `results/ffn.json` | ~5–6 min |
| `04_classical` | classical baseline | `checkpoints/classical.pkl`, `results/classical.json` | ~2 min |
| `05_resnet_frozen` | frozen ResNet18, near-duplicate check | `checkpoints/resnet_frozen.pt`, `results/resnet_frozen.json` | ~15 s |
| `06_resnet_finetune` | fine-tuned ResNet18, from-scratch control, Grad-CAM | `checkpoints/resnet_finetune.pt`, `results/resnet_finetune.json` | ~5 min |
| `07_comparison` | the four candidates against the criteria | test fields of `results/*.json` (final run only) | ~20 s |
| `08_guide` | the guide's "not sure" threshold and demo cards | figures | ~10 s |
| `09_search` | optional: text search with CLIP and the chat reply (needs `requirements-search.txt`) | figures | ~1.5 min + downloads |

- **Checkpoints are not committed** (`checkpoints/` is ignored), so notebooks 03–06 must run once before 07 and 08. Notebook 05 downloads the ImageNet weights of ResNet18 the first time.
- **`07_comparison` runs on the validation set by default.** The test set was scored once, with `FINAL_TEST=1`, and its results are in `results/*.json`. Re-running it on another machine gives slightly different numbers, so those must not be committed.
- **Small differences between machines are expected:** training on MPS isn't reproducible bit for bit across computers. An independent fresh-clone re-run of the whole pipeline reproduced every final test accuracy within 2 images out of 225, and the same conclusion.
- **Quick check:** `FAST_RUN=1 jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb` runs a notebook in seconds (1 seed, few epochs), but it overwrites the notebook's outputs and figures with low-quality versions. Run `git restore notebooks results` afterwards and never commit them.

## Dataset

Subset of [`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart) at revision `d559852d2b232e0fcf195e775866964f0564f2b5`: 128×128 RGB JPEGs (centre crop), seed 0, at most 50 images per named artist.

| Index | Style | Images |
|---|---|---|
| 0 | Baroque | 300 |
| 1 | Cubism | 300 |
| 2 | Impressionism | 300 |
| 3 | Pop_Art | 300 |
| 4 | Ukiyo_e | 300 |

Class indices follow alphabetical folder order, as in `torchvision.datasets.ImageFolder`. A fixed stratified split (70/15/15 → 1,050 / 225 / 225, i.e. 210 / 45 / 45 per class, seed 0) lives in `data/split.csv`; every model uses it. It depends only on `metadata.csv`, and `python scripts/make_split.py` rebuilds the same file.

**Known limitations**
- 1,402 of the 1,500 images are labelled "Unknown Artist", and most have "Unknown Genre". So we can't split by artist, and genre classification isn't feasible on this subset.
- Impressionism and Pop_Art each come from only 1–2 source files, so there may be near-duplicates across splits. The check in `05_resnet_frozen` found 3 near-duplicate pairs (ResNet feature cosine > 0.95), none of them across splits.
- Some images are photos of sculptures or objects rather than paintings: about 7 % of Impressionism and 18 % of Pop Art in a sample of 60 per style (`01_data`). The error analysis in `03_diagnosis` found no shortcut based on them.
- 225 test images give accuracy intervals of roughly ±5 points; we report them.

## Repository structure

```
README.md, PLAN.md
L2_NUMBERS_AND_TIMEKEEPING.md  numbers check and talk timing                Lenny
requirements.txt            project dependencies
requirements-data.txt       only for rebuilding the dataset (Option B)
requirements-dev.txt        pytest and ruff (checks)
requirements-search.txt     only for the optional search (transformers)
pyproject.toml              ruff and pytest settings
.github/workflows/ci.yml    CI: lint + tests on every pull request
tests/                      checks run by CI (synthetic images, no WikiArt data needed)
scripts/
  make_wikiart_subset.py    builds the image subset
  validate_wikiart_subset.py checks images against the metadata
  make_split.py             writes data/split.csv                      Anthony
data/
  wikiart_style5/           images (ignored) + metadata.csv, subset_info.json
  split.csv                 train / val / test assignment
src/museum/
  data.py                   datasets, loaders, normalisation, augmentation  Anthony
  train.py                  training loop, early stopping, grad norms        Anthony
  models.py                 StyleMLP + frozen FFN candidate (Anthony) · ResNet18, TorchPredictor (Antoine)
  classical.py              colour histogram + HOG features                   Antoine
  diagnostics.py            gradient / activation / LR diagnostics            Anthony
  evaluation.py             metrics, CIs, McNemar, latency, result files      Antoine
  guide.py                  the guide: photo → style, confidence, story        Anthony
  stories.py, stories.md    the movement texts the guide shows                 Anthony
  search.py                 optional: text search over the collection (CLIP)  Anthony
  chat.py                   optional: the chat reply and its fact check       Anthony
notebooks/
  01_data.ipynb             data exploration and split checks                 Anthony
  02_ffn.ipynb              feed-forward network                              Anthony
  03_diagnosis.ipynb        training monitoring and failure diagnosis         Anthony
  04_classical.ipynb        classical baseline                                Antoine
  05_resnet_frozen.ipynb    ResNet18 frozen features + linear head            Antoine
  06_resnet_finetune.ipynb  ResNet18 fine-tuning                              Antoine
  07_comparison.ipynb       test evaluation, comparison, recommendation       Antoine
  08_guide.ipynb            the guide: "not sure" threshold, demo cards        Anthony
  09_search.ipynb           optional: text search and chat reply              Anthony
results/                    <model>.json + figures/ (source of every reported number)
checkpoints/                trained models, written by notebooks 03–06 (not committed)
slides/museum_guide.pptx    the presentation
```

## How we work

- **One branch per task** (`feat/data`, `feat/ffn`, `feat/classical`, …), merged into `main` through a pull request reviewed by Anthony or Antoine, whoever didn't write it.
- **One owner per notebook** (see above) to avoid merge conflicts. Notebook outputs are kept: they are our evidence.
- **Never commit** images, zips or checkpoints (`*.pt`). `.gitignore` covers them; check `git status` before each commit.
- **The test set is used once**, in `07_comparison.ipynb`. All tuning and model selection happen on the validation set.
- **Every reported number** comes from `results/*.json` or a notebook output.
- **Problems go into GitHub issues.** If a step of this README fails or is unclear, open an issue; the owner of the file fixes it. Lenny runs the full fresh-clone check on Day 2 and Day 5.
- Run notebooks top to bottom (Kernel → Restart & Run All) before committing them; see [Reproducing the results](#reproducing-the-results) for the order and the `FAST_RUN` quick check.
- **CI must be green before merging.** Run the same checks locally first:

  ```bash
  pip install -r requirements-dev.txt
  ruff check .      # lint (add --fix for the automatic fixes)
  pytest            # tests, a few seconds on CPU
  ```

  CI checks:
  - the code: lint, plus `data.py`, `train.py` and the split, tested on synthetic images
  - the project rules:
    - metadata consistent with the manifest
    - no images or checkpoints committed
    - every committed notebook run top to bottom with no error output
    - no test-set scoring outside `07_comparison.ipynb`

## Results

Final test set: 225 images, scored once in `notebooks/07_comparison.ipynb` (`FINAL_TEST=1`). Every number below comes from `results/*.json`.

| Model | Test accuracy (95 % CI) | Macro-F1 | CPU latency, 1 thread | Size |
|---|---|---|---|---|
| Classical (colour histogram + HOG) | 47.6 % [41.1, 54.1] | 0.475 | 2.98 ms | 16.5 MB |
| Feed-forward network | 40.4 % [34.2, 47.0] | 0.395 | 0.09 ms | 3.3 MB |
| ResNet18, frozen + linear head | 79.6 % [73.8, 84.3] | 0.795 | 19.1 ms | 44.8 MB |
| ResNet18, fine-tuned | **84.0 %** [78.6, 88.2] | 0.838 | 16.6 ms | 44.8 MB |

**Against the criteria fixed in advance** ([PLAN.md §4](PLAN.md#4-decision-criteria-for-the-director-fixed-before-any-results)), no model passes all three. The fine-tuned ResNet18 comes closest. It is significantly better than the classical model (McNemar p < 0.001) and passes latency (≤ 40 ms) and footprint (≤ 100 MB, trained in 78 s), but it misses the 85 % accuracy bar by 1.0 point (189 of 225 right, where 85 % needs 192). Its confidence interval contains 85 %. We recommend it for a supervised pilot with abstention turned on, rather than a full deployment. More data for Impressionism and Pop Art, blur and perspective augmentation, and a test with real kiosk photos are the next steps. The full reasoning is in section 9 of `07_comparison.ipynb`.

## Tested environment

macOS (Apple Silicon), Python 3.13.7, torch 2.14.1, torchvision 0.29.1, scikit-learn 1.9.1, scikit-image 0.26.0, numpy 2.5.4, pandas 3.0.6, Pillow 13.0.0, and transformers 5.20.2 for the optional search. Training uses the Apple GPU (MPS) when available; every latency is measured on the CPU.
