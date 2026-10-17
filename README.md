# Making Document Extraction Worth Automating

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AviDip/doc-extraction/blob/main/colab.ipynb)

Code for the PAI-ML 2 project: structural consistency training and layout-novelty routing for key
information extraction on CORD, WildReceipt and (optionally) DocILE.

Two questions, one pipeline:

1. **Fewer errors.** Does page jitter, key-phrase swapping or a structural consistency loss reduce the
   accuracy lost on unseen layouts, Δ = F1(many-shot) − F1(zero-shot)?
2. **Findable errors.** Which label-free signal best ranks documents for human review, measured as the
   share of documents that can skip review at a fixed F1?

## Code layout

| Module | Role |
|---|---|
| `schema.py` | `Document` (words, boxes, BIO tags in reading order), `LabelSpace`, BIO ↔ entity conversion |
| `layout.py` | reading order and line grouping, page jitter |
| `numbers.py` | amount parsing for both separator conventions; split-thousands merging |
| `data/cord.py`, `data/wildreceipt.py`, `data/docile.py` | dataset loaders; predictions → DocILE `Field`s for the official evaluator |
| `prepare.py` | CLI: raw datasets → cached `.pkl` splits; CORD template-disjoint split |
| `split.py` | template clustering on shared field keys (Laatiri et al.), disjoint assignment |
| `analyse.py` | CLI: noise floor — how often ground truth satisfies each constraint |
| `augment.py` | page jitter + key-phrase swapping with a key vocabulary mined from training data |
| `constraints.py` | equations, singleton fields, `ConsistencyLoss`, label-free violation count |
| `encoding.py` | tokenisation, 512-token sliding windows, word pooling |
| `models.py` | RoBERTa / LayoutLMv3 / BROS behind one interface (boxes in, logits out) |
| `train.py` | CLI: fine-tune, then predict and score every split |
| `predict.py` | batched inference: word labels, confidences, document embeddings |
| `metrics.py` | entity-level micro F1, zero/few/many-shot subsets |
| `evaluate.py` | CLI: re-score a run; DocILE official KILE evaluator |
| `routing.py` | CLI: novelty, MSP, Mahalanobis and consistency signals; risk–coverage |
| `summarise.py` | CLI: mean ± std over seeds |
| `hub.py` | CLI: push / pull results to a Hugging Face dataset repo |
| `runner.py` | CLI: runs `jobs.json` on a GPU machine and publishes results to the `results` branch |

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q        # offline: synthetic receipts, tiny random models
```

WildReceipt is public: `wget https://download.openmmlab.com/mmocr/data/wildreceipt.tar`.
Its test templates are unseen in training by construction, so it needs no token and no custom split.

DocILE (optional) needs a free token from <https://docile.rossum.ai>, then:

```bash
git clone --depth 1 https://github.com/rossumai/docile.git
bash docile/download_dataset.sh $DOCILE_TOKEN annotated-trainval data/docile_raw --unzip
```

## Automatic mode

`jobs.json` lists every experiment with its dependencies. `python -m docext.runner --repo AviDip/doc-extraction`
(started from the notebook's first two cells) pulls new code and jobs every minute, runs whatever is ready, and
pushes logs, run folders and `status.json` to the `results` branch after each job. Only `python -m docext.*` and
`bash scripts/*` commands are allowed. Raising a job's `rev` reruns it.

## Workflow

`colab.ipynb` runs every step below on Colab. It clones this repository, fetches each dataset from its official source, and pulls finished runs from the results repository on the Hugging Face Hub, so published numbers can be re-scored without a GPU.

```bash
# 1. Data
python -m docext.prepare cord --out data
python -m docext.prepare cord-disjoint --out data          # writes data/cord_disjoint + split_report.json
python -m docext.prepare wildreceipt --wildreceipt-path data/wildreceipt_raw --out data
python -m docext.prepare docile --docile-path data/docile_raw --out data   # optional

# 2. Noise floor (go / no-go for the consistency loss)
python -m docext.analyse data/cord/train.pkl --model layoutlmv3
python -m docext.analyse data/docile/train.pkl --model layoutlmv3

# 3. Train (one configuration, one seed)
python -m docext.train --data data/cord --eval-splits validation test
python -m docext.train --data data/docile --eval-splits val --epochs 8 \
    --jitter 0.05 --key-swap 0.5 --lambda-arith 0.3 --lambda-card 0.3 --seed 0

# 4. Score, route, aggregate, share
python -m docext.evaluate runs/<run> --docile-path data/docile_raw
python -m docext.routing runs/<run> --split val --target 0.95
python -m docext.summarise runs --split val --csv results.csv
python -m docext.hub push AviDip/doc-extraction-results
```

Each run directory holds `meta.json`, `history.json` (token, arithmetic and cardinality losses per epoch),
`pred_<split>.npz` (word labels, confidences, document embeddings; plain arrays, safe to share) and `metrics.json`. Evaluation and
routing need no GPU.

## Design decisions

These refine the proposal; each follows from a property of the data or the models.

| Decision | Reason |
|---|---|
| LayoutLMv3 runs without page images | The image reveals the exact layout that jitter hides, and RoBERTa and BROS receive none, so all three models see the same input. |
| Jitter is a page-level shift and scale, not per-word noise | Line spacing is about 1.5% of page height; per-word noise of up to 10% would scramble rows. A shared transform keeps relative layout and removes absolute position. |
| Keys are the unlabelled words left of a field on its line | Neither dataset annotates keys. Mining them gives a per-field vocabulary of real phrases for swapping. CORD's `is_key` words are tagged O. |
| Cardinality penalises only counts above one | A missing field is legitimate (not every invoice has a due date). Singleton fields are those that occur at most once in ≥ 95% of training documents. |
| Arithmetic loss applies only where ground truth satisfies the equation | Measured by `analyse`; documents that do not add up (discounts, rounding, OCR) would teach the wrong thing. |
| Word values are divided by the document total and capped at ±10 | Totals range from 10 to 10⁶, and invoice numbers or dates that parse as numbers would otherwise dominate the gradient. Huber loss bounds it further. |
| Both structural terms ramp up from zero over the first 20% of steps | Both are zero when nothing is labelled, so they only redistribute probability mass the token loss has committed. |
| Template clusters use header words and key wording, with the threshold calibrated on WildReceipt | Clustering on which fields appear made 94% of WildReceipt's test look "seen", although its authors guarantee unseen shops: nearly every receipt has a total and tax. On header words, a 0.5 Jaccard threshold leaves 2.5% replication on WildReceipt. Replication is measured at 0.5 (few false matches); the CORD disjoint split is built at 0.7 (few missed matches, so a shop rarely lands on both sides). |
| One label per word | Softmax gives P(word ∈ field) directly. Words matching two DocILE fields keep the larger overlap; `prepare` reports how many. |
| Fixed epochs, no checkpoint selection | DocILE results are reported on `val`, so it cannot also select checkpoints. Hyperparameters are tuned on CORD validation. |
| Official DocILE evaluator scores KILE fields | LIR also needs line-item grouping, which this project does not model. Entity F1 covers all 55 field types. |
| Row-major reading order | Multi-column text can break a field apart; `analyse` reports the share of fragmented fields. |

## Compute budget

A base-size run costs roughly (documents × epochs). Time one DocILE epoch first, then size the grid. A
practical grid: LayoutLMv3 × {baseline, jitter, key swap, consistency, all} × 3 seeds, plus RoBERTa and
BROS at baseline and best arm × 3 seeds (27 runs). Keep `--no-save-model` (default) to stay within Drive
storage. The notebook runs several jobs on one GPU at once (`PARALLEL`); small CORD runs leave an A100
mostly idle, so this roughly doubles throughput.

## Reproducing

| Want to | Do |
|---|---|
| Check published numbers | Open the notebook, run the first cell (pulls results), run each dataset's prepare cell, then the routing and summary cells. No GPU needed. |
| Rerun everything | Set `RESULTS_REPO` to your own Hugging Face repo and run all cells. |

Converted datasets and fine-tuned weights are not shared: anyone can rebuild them with `docext.prepare` and
`docext.train`, and DocILE's terms do not allow redistributing its data.

## Licence

Code: MIT (see `LICENSE`). Datasets keep their own licences: CORD (CC BY 4.0), WildReceipt (via MMOCR),
DocILE (terms accepted when requesting the token).
