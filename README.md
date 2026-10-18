# Museum Guide — Artwork Style Recognition

A (fictional) mid-sized French fine-arts museum wants a guide on visitors' phones and gallery kiosks: point the camera at a painting, recognise its **style**, and tell the story of that movement. We build, diagnose and defend the neural-network components of that guide, and answer the director's question:

> Kiosks run on modest CPUs, the team is small and the budget is limited. **Is a neural network worth it in production, or would a simpler classical model do the job?**

Final project of the Neural Networks & Deep Learning course. Networks are built from PyTorch primitives (`nn.Module`, `nn.Linear`, hand-written training loops), and every number we report is measured.

## Team

| Member | Owns |
|---|---|
| **Anthony** | Feed-forward network, training monitoring & failure diagnosis, data & training code, guide demo |
| **Antoine** | Pretrained ResNet, classical baseline, evaluation tools, final comparison |

The full task split, schedule, shared interfaces and rules are in **[PLAN.md](PLAN.md)**.

## Deliverables

- [ ] Feed-forward network in PyTorch for artwork style classification
- [ ] Training monitoring and failure diagnosis
- [ ] Pretrained CNN (ResNet18) for style classification
- [ ] Classical baseline (colour histograms + HOG, scikit-learn) for comparison
- [ ] Recommendation for the museum director (accuracy, CPU latency, footprint)
- [ ] 10-minute presentation + Q&A

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

## Dataset

Subset of [`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart) at revision `d559852d2b232e0fcf195e775866964f0564f2b5`: 128×128 RGB JPEGs (centre crop), seed 0, at most 50 images per named artist.

| Index | Style | Images |
|---|---|---|
| 0 | Baroque | 300 |
| 1 | Cubism | 300 |
| 2 | Impressionism | 300 |
| 3 | Pop_Art | 300 |
| 4 | Ukiyo_e | 300 |

Class indices follow alphabetical folder order, as in `torchvision.datasets.ImageFolder`. A fixed stratified split (70/15/15 → 1,050 / 225 / 225, seed 0) will live in `data/split.csv`; every model uses it.

**Known limitations**
- 1,402 of the 1,500 images are labelled "Unknown Artist", and most have "Unknown Genre". So we can't split by artist, and genre classification isn't feasible on this subset.
- Impressionism and Pop_Art each come from only 1–2 source files, so there may be near-duplicates across splits. We check this before reporting results.
- 225 test images give accuracy intervals of roughly ±5 points; we report them.

## Repository structure

Target layout; files appear as tasks are merged.

```
README.md, PLAN.md
requirements.txt            project dependencies
requirements-data.txt       only for rebuilding the dataset (Option B)
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
  models.py                 StyleMLP (Anthony) · build_resnet18 (Antoine)
  diagnostics.py            gradient / activation / LR diagnostics            Anthony
  evaluation.py             metrics, CIs, McNemar, latency, result files      Antoine
  guide.py, stories.py      demo: image → style + story                       Anthony
notebooks/
  01_data.ipynb             data exploration and split checks                 Anthony
  02_ffn.ipynb              feed-forward network                              Anthony
  03_diagnosis.ipynb        training monitoring and failure diagnosis         Anthony
  04_classical.ipynb        classical baseline                                Antoine
  05_resnet_frozen.ipynb    ResNet18 frozen features + linear head            Antoine
  06_resnet_finetune.ipynb  ResNet18 fine-tuning                              Antoine
  07_comparison.ipynb       test evaluation, comparison, recommendation       Antoine
results/                    <model>.json + figures/ (source of every reported number)
slides/                     presentation
```

## How we work

- **One branch per task** (`feat/data`, `feat/ffn`, `feat/classical`, …), merged into `main` through a pull request reviewed by the other person.
- **One owner per notebook** (see above) to avoid merge conflicts. Notebook outputs are kept: they are our evidence.
- **Never commit** images, zips or checkpoints (`*.pt`). `.gitignore` covers them; check `git status` before each commit.
- **The test set is used once**, in `07_comparison.ipynb`. All tuning and model selection happen on the validation set.
- **Every reported number** comes from `results/*.json` or a notebook output.
- Run notebooks top to bottom. For a quick end-to-end check:
  `FAST_RUN=1 jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb`

## Results

| Model | Test accuracy (95 % CI) | Macro-F1 | CPU latency, 1 thread | Size |
|---|---|---|---|---|
| Classical (colour histogram + HOG) | – | – | – | – |
| Feed-forward network | – | – | – | – |
| ResNet18, frozen + linear head | – | – | – | – |
| ResNet18, fine-tuned | – | – | – | – |

Filled in from `results/*.json` after the final comparison. Decision criteria for the recommendation are fixed in advance in [PLAN.md §4](PLAN.md#4-decision-criteria-for-the-director-fixed-before-any-results).

## Tested environment

macOS (Apple Silicon), Python 3.13, torch 2.8.0, torchvision 0.23.0, scikit-learn 1.6.1, scikit-image 0.25.0.
