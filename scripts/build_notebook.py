"""Build project.ipynb: the whole project in one explained notebook.

The package in docext/ stays the single source of truth; this script inlines
each module in dependency order, removes package plumbing (relative imports,
command-line entry points), and interleaves the explanations below.

    python scripts/build_notebook.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "docext"

# functions that clash once every module shares one namespace
RENAMES = {
    "data/cord.py": {"parse_record": "parse_cord_record"},
    "data/wildreceipt.py": {"parse_record": "parse_wildreceipt_record", "field_types": "wildreceipt_field_types"},
    "data/docile.py": {"field_types": "docile_field_types"},
    "prepare.py": {"field_types()": "docile_field_types()"},
}
# a module-level alias needed where a relative import used "as"
PRELUDES = {"evaluate.py": "read_predictions = load_predictions  # the .npz reader from the prediction section\n\n"}
DROP_FUNCTIONS = {"main", "parse_args"}


def inline(module: str) -> str:
    source = (PKG / module).read_text()
    for old, new in RENAMES.get(module, {}).items():
        pattern = re.escape(old) if old.endswith(")") else rf"\b{re.escape(old)}\b"
        source = re.sub(pattern, new, source)
    tree = ast.parse(source)
    drop: list[tuple[int, int]] = []
    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant) \
            and isinstance(node.value.value, str) and node is tree.body[0]
        is_entry = isinstance(node, ast.FunctionDef) and node.name in DROP_FUNCTIONS
        is_main_guard = isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
        is_future = isinstance(node, ast.ImportFrom) and node.module == "__future__"
        if is_docstring or is_entry or is_main_guard or is_future:
            drop.append((node.lineno, node.end_lineno))
    lines = source.splitlines()
    keep = [line for i, line in enumerate(lines, 1) if not any(a <= i <= b for a, b in drop)]
    keep = [line for line in keep if not re.match(r"\s*from \.+\w*(\.\w+)* import ", line)]
    text = "\n".join(keep)
    if "argparse." not in text:
        text = re.sub(r"^import argparse\n", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n\n", text).strip() + "\n"
    return PRELUDES.get(module, "") + text


def top_level_names(code: str) -> set[str]:
    names = set()
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


# --------------------------------------------------------------------------- text

INTRO = """
# Making Document Extraction Worth Automating

**Avishek Paul Dipto (a1988440) · Programming for Artificial Intelligence and Machine Learning 2 · University of Adelaide**

Document extraction models read invoices and receipts with over 90% field accuracy, yet most documents are still checked by hand, because nobody knows *which* outputs are wrong. This notebook implements and evaluates two responses:

| Goal | Method | Where |
|---|---|---|
| **Fewer errors** on unfamiliar layouts | page jitter, key-phrase swapping, and a **structural consistency loss** (amounts must add up; key fields appear once) | Sections 4, 6, 10 |
| **Findable errors** without labels | four risk signals ranked by **risk–coverage curves**: layout novelty, model confidence, feature-space distance, constraint violations | Section 13 |

**How to run.** Everything is in this notebook; only the datasets are downloaded, from their public sources (Hugging Face, OpenMMLab, the DocILE GitHub toolkit). Run top to bottom. Part A defines the code with explanations; Part B runs the experiments. With `USE_PUBLISHED_RESULTS = True` (Part B), finished runs are fetched from the project's GitHub repository, so results and figures can be reproduced on a CPU in minutes; set it to `False` to retrain on a GPU.

| Section | Contents |
|---|---|
| 0 | Setup |
| **Part A — the code** | |
| 1 | Data structures: documents, BIO labels, entities |
| 2 | Page geometry: reading order, jitter |
| 3 | Reading amounts from OCR text |
| 4 | Training-time augmentation |
| 5 | Template clustering and layout-disjoint splits |
| 6 | Dataset loaders: CORD, WildReceipt, DocILE |
| 7 | Structural constraints and the consistency loss |
| 8 | Noise floor: does the ground truth satisfy the constraints? |
| 9 | Models: RoBERTa, LayoutLMv3, BROS |
| 10 | Encoding: tokens, sliding windows, word pooling |
| 11 | Inference and metrics |
| 12 | Training loop |
| 13 | Evaluation, routing signals and risk–coverage |
| 14 | Result tables and the document viewer |
| **Part B — the experiments** | |
| 15 | Data, calibration and noise floor |
| 16 | Training runs |
| 17 | Results, routing and worked examples |
"""

SETUP = """
## 0. Setup

The libraries used, and why:

| Library | Role |
|---|---|
| `torch` | tensors, automatic differentiation, GPU training |
| `transformers` | pretrained encoders (RoBERTa, LayoutLMv3, BROS) and their tokenizers |
| `datasets` | downloads CORD from the Hugging Face Hub |
| `docile-benchmark` | reads DocILE and runs its official evaluator |
| `scikit-learn` | clustering, nearest neighbours, covariance estimation |
| `numpy`, `matplotlib`, `pandas` | arrays, figures, tables |

Colab's own `numpy`, `pandas` and `torch` versions are pinned during installation so the preinstalled packages keep working.
"""

SETUP_CODE = """
!pip freeze | grep -iE '^(numpy|pandas|torch|scipy)==' > /tmp/pins.txt
!pip install -q "transformers>=4.45" "datasets>=2.19" "docile-benchmark>=0.3" scikit-learn matplotlib -c /tmp/pins.txt

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (Part B can still reuse published results)')
"""

SECTIONS: list[tuple[str, str]] = [
    ("schema.py", """
# Part A — the code

## 1. Data structures

Every dataset is converted into one common form, so the rest of the code never needs to know where a document came from.

| Term | Meaning |
|---|---|
| **Word** | One OCR token with its text and bounding box. |
| **Bounding box** | `(x0, y0, x1, y1)`: left, top, right, bottom, as fractions of the page (0 to 1), so pages of any size are comparable. |
| **Field / entity** | A run of words with one meaning, e.g. the total amount. |
| **BIO tags** | One label per word: `B-f` begins field *f*, `I-f` continues it, `O` is outside any field. This turns field extraction into word classification. |
| **Label space** | The list of all BIO labels; with *F* field types there are 2*F*+1 labels. |

**Worked example.** The receipt line `TOTAL 50,000` is two words. `TOTAL` is printed text naming the field, not the value, so it gets `O`; `50,000` gets `B-total.total_price`. If the total were split as `50` `000`, the second word would get `I-total.total_price`.

`tags_to_entities` reverses this: it groups tags back into `Entity(label, start, end)` spans. An `I-` tag that does not continue the current field starts a new one (lenient decoding, as in the standard `seqeval` scorer). Splits are stored with `pickle` because they are rebuilt locally and never shared; predictions, which are shared, use `.npz` (Section 11).
"""),
    ("layout.py", """
## 2. Page geometry

**Reading order.** OCR engines often return words grouped by text block, which mixes rows. `reading_order` sorts words top to bottom and groups them into lines: a word joins the current line when its vertical centre lies within half a typical word height of the line's centre. Within a line, words are sorted left to right.

**Worked example.** Words at heights 0.100, 0.105 and 0.300 (word height 0.02): the first two centres differ by 0.005 < 0.01, so they share line 0; the third starts line 1.

**BIO after sorting.** `build_document` assigns `B-`/`I-` *after* sorting, so `B-` always marks the first word of a field in reading order. It also counts **fragmented fields**: fields whose words are separated by other words in reading order (common in multi-column text). These are reported as a data-quality figure.

**Page jitter** (augmentation, Section 4). Each page is randomly rescaled and shifted by up to a magnitude *m* (a share of the page size). The same transform applies to every word on the page, so rows, columns and key-value alignment are preserved, but absolute positions never repeat. This removes the shortcut of memorising *where* a field sits on a familiar template. Per-word noise was rejected: text lines are about 1.5% of the page apart, so 10% per-word noise would scramble rows.
"""),
    ("numbers.py", """
## 3. Reading amounts from OCR text

The consistency loss (Section 7) needs the numeric value of every word. Receipts and invoices print amounts in several conventions:

| Printed | Convention | Value |
|---|---|---|
| `1,234.56` | comma thousands, point decimal | 1234.56 |
| `1.234,56` | point thousands, comma decimal (Europe) | 1234.56 |
| `45,000` | thousands only (CORD's Rupiah receipts) | 45000 |
| `12,50` | comma decimal | 12.5 |
| `(5.00)` or `5.00-` | negative | -5.0 |
| `12.03.2021`, `21%`, `12:30` | dates, rates, times | not an amount |

The rule: when both separators appear, the last one is the decimal point; when only one appears, it is a thousands separator if exactly three digits follow it, otherwise a decimal point. Thousands groups must have exactly three digits, which rejects dates.

**Split amounts.** OCR often splits `1 234,56` into two words. `word_values` merges a 1–3 digit word with a following word that starts with a three-digit group on the same line, and puts the whole value on the second word. Summing word values over a labelled field then still gives the right amount.
"""),
    ("augment.py", """
## 4. Training-time augmentation

Neither dataset annotates **keys**, the printed text that names a field (`Total:`, `Amount due`). A key is taken to be the unlabelled, non-numeric words directly left of a field on the same line (up to four words).

| Function | What it does |
|---|---|
| `find_key_spans` | finds the key phrase before each field |
| `key_vocabulary` | collects, per field type, every key phrase seen at least twice in the training set |
| `swap_keys` | with probability *p*, replaces a key with another phrase of the same field; new words share the old key's box, split by character length |
| `Augmenter` | applies key swapping and page jitter to one training document |

**Worked example.** In training, `total.total_price` appears after `TOTAL`, `GRAND TOTAL` and `Total Due`. With *p* = 0.5, a receipt's `TOTAL 60,500` may become `GRAND TOTAL 60,500`. The value and its label are unchanged, so the model learns to recognise the total from several wordings rather than one exact string.
"""),
    ("split.py", """
## 5. Template clustering and layout-disjoint splits

A **template** is a layout shared by documents from one source, such as all receipts from one shop. If a test document's template also appears in training, the model can succeed by memorising positions; Laatiri et al. (2023) found this in 75% of SROIE's official test set.

**Signature.** Each document is described by the words in its first five lines (shop name, address) and its key phrases. Field types are deliberately excluded: almost every receipt has a total and tax, so they make unrelated shops look alike. (The first version used them and labelled 94% of WildReceipt's unseen test shops as "seen".)

**Distance and clustering.** The distance between two signatures *A* and *B* is the Jaccard distance:

$$d(A, B) = 1 - \\frac{|A \\cap B|}{|A \\cup B|}$$

Agglomerative clustering with average linkage merges documents while their average distance is below a threshold.

**Worked example.** `{safeway, grocery, key:total:total}` and `{safeway, key:total:total}` share 2 of 3 items, so *d* = 1 − 2/3 = 0.33, below 0.5: same template.

**Calibration.** WildReceipt's authors guarantee that test shops never appear in training, so the correct clustering finds almost none of them "seen". `calibrate` measures, at each threshold, the **replication rate** (share of test documents whose template occurs in training). The chosen threshold (0.5) is the loosest that keeps WildReceipt near zero.

| Function | Purpose |
|---|---|
| `disjoint_split` | assigns whole clusters to test, validation and train, so no template crosses splits |
| `seen_template_holdout` | picks validation documents whose template stays in training: a *seen-layout* set |
"""),
    ("data/cord.py", """
## 6. Dataset loaders

All three loaders produce the `Document` structure of Section 1.

| Dataset | Documents | Fields | Layout split | Source |
|---|---|---|---|---|
| **CORD** | 1,000 receipts | 30 | built here (Section 5) | Hugging Face `naver-clova-ix/cord-v2` |
| **WildReceipt** | 1,740 receipts | 12 values (+ keys) | official: test shops unseen | OpenMMLab download |
| **DocILE** | 5,680 invoices (train + val) | 55 | official zero/few/many-shot clusters | DocILE toolkit, free token |

### 6.1 CORD
Each annotated line carries one category. Words flagged `is_key` (printed keys) become `O`, so only values are targets, consistent with DocILE. Quadrilateral word boxes are reduced to their bounding rectangle and divided by the image size.
"""),
    ("data/wildreceipt.py", """
### 6.2 WildReceipt
Each OCR box has one of 25 classes: a value or a key for 12 fields, plus *Others*. Value boxes become fields; keys and *Others* become `O`. The released text has spaces removed (`THANKYOU`), so each box is one word.
"""),
    ("data/docile.py", """
### 6.3 DocILE
DocILE gives field boxes and OCR word boxes separately. A word is assigned to a field when at least half of its area lies inside the field box (the rule used by the DocILE baselines). Where a word qualifies for two fields, it takes the larger overlap and the conflict is counted. Field types are the 36 KILE fields (document-level) plus the 19 line-item fields.

`to_kile_fields` converts predictions back into DocILE's format for the official evaluator, which matches fields by location: each predicted entity becomes the union box of its words.
"""),
    ("prepare.py", """
### 6.4 Preparing splits
These functions download and convert each dataset and save the splits. For CORD they also build the layout-disjoint split; for WildReceipt they hold out a seen-template validation set, so that

$$\\Delta = F_1(\\text{seen layouts}) - F_1(\\text{unseen layouts})$$

can be measured within one dataset.
"""),
    ("constraints.py", """
## 7. Structural constraints and the consistency loss

This is the project's main contribution. A standard model is trained word by word with cross-entropy; nothing tells it that a document's items, tax and total must agree.

**Expected field value.** Numbers on the page are fixed by OCR; only labels are learned. If $p_i(f)$ is the model's probability that word *i* belongs to field *f* and $v_i$ is the word's numeric value, the field's expected value is

$$S(f) = \\sum_i p_i(f)\\, v_i$$

which is differentiable in the model's outputs.

**Arithmetic term.** For an equation such as *items + tax = total*:

$$r = \\frac{S(\\text{items}) + S(\\text{tax}) - S(\\text{total})}{|\\text{total}|}, \\qquad L_{\\text{arith}} = \\text{Huber}(r)$$

**Cardinality term.** Key fields such as the total occur at most once. The expected number of instances is $N(f) = \\sum_i p_i(B\\text{-}f)$, and only excess is penalised: $L_{\\text{card}} = \\sum_f \\max(0, N(f) - 1)$. A missing field is legitimate (not every invoice has a due date).

**Worked example.** Items 25,000 + 5,000, tax 3,000, total 33,000. If the model gives the true total word $p = 0.5$ and wrongly gives an item price (25,000) $p = 0.5$ of being the total, then $S(\\text{total}) = 0.5 \\cdot 33{,}000 + 0.5 \\cdot 25{,}000 = 29{,}000$, the residual is $(33{,}000 - 29{,}000)/33{,}000 = 0.12$, and $N(\\text{total}) = 1.0$. The gradient pushes probability away from the item price.

**Design choices.**

| Choice | Reason |
|---|---|
| Arithmetic applies only where the ground truth satisfies the equation | Documents with discounts or OCR errors would teach the wrong relation (Section 8 measures how many) |
| Values divided by the total and capped at ±10 | totals range over six orders of magnitude; IDs and dates parsed as numbers would otherwise dominate |
| Huber loss | quadratic for small residuals, linear for large ones: bounded gradients |
| Singleton fields chosen from data | a field counts as "at most once" if 95% of training documents agree |
| Both terms are zero when nothing is labelled | they cannot replace cross-entropy; they redistribute mass it has committed, so they are weighted and warmed up |

`count_violations` applies the same checks to hard predictions; it is the label-free **consistency signal** for routing.
"""),
    ("analyse.py", """
## 8. Noise floor

Before training with the consistency loss, it must be checked that the ground truth itself satisfies the constraints often enough. `analyse` reports, per equation, how many documents it applies to and the share where the true labels add up. A second parse (each field's words joined and read as one number) separates word-level parsing failures from documents that genuinely do not add up. It also reports singleton fields, fragmented fields, key phrases, and how many documents exceed one 512-token window.
"""),
    ("models.py", """
## 9. Models

Three encoders form a ladder that isolates the contribution of layout:

| Model | Layout information | Role |
|---|---|---|
| **RoBERTa** | none: text only | control |
| **LayoutLMv3** | absolute 2-D position embeddings + relative spatial attention bias | working model |
| **BROS** | relative position between every pair of boxes | reference for relative layout |

Each model reads words and boxes and outputs, for every token, a score per BIO label (a **token classifier**). The `Extractor` class hides the differences: boxes are always passed as floats in [0, 1] and converted to what each architecture expects (none, integers on a 0–1000 grid, or floats).

LayoutLMv3 is used **without page images**: the image would reveal the exact layout that jitter is meant to hide, and RoBERTa and BROS receive none, so all three see the same input.
"""),
    ("encoding.py", """
## 10. Encoding: tokens, windows and word pooling

**Sub-word tokens.** Transformers split words into sub-word tokens (`45,000` → `45`, `,`, `000`). Each word is labelled and predicted through its **first** sub-word token; the others are ignored in the loss (label −100).

**Sliding windows.** The encoders accept 512 tokens. Longer documents (about 40% of DocILE) are cut into overlapping windows of 510 tokens plus the two special tokens, with 128 tokens of overlap.

**Worked example.** A 1,200-token invoice gives windows starting at tokens 0, 382 and 690. A word in two windows receives the average of both predictions.

**Word pooling.** `pool_words` averages token-level vectors back into one vector per word across windows. The consistency loss needs this: an invoice's line items and total often sit in different windows.

The `Collator` turns a list of documents into one batch: padded windows, boxes, labels, the document index of every word, word values, and per-document flags for which equations hold in the ground truth. During training a document is capped at four windows per step; such partial documents are excluded from the document-level loss terms.
"""),
    ("predict.py", """
## 11. Inference and metrics

### 11.1 Inference
`predict` runs the model without gradients, in chunks of 32 windows, and stores per word the predicted label and its probability (**maximum softmax probability**, the model's confidence), plus one **document embedding**: the mean final-layer vector over the document's words, used by the Mahalanobis signal. Mixed precision (bfloat16 on A100) halves memory and speeds up inference. Predictions are saved as `.npz`, plain arrays that are safe to share: unlike pickle, loading them cannot execute code.
"""),
    ("metrics.py", """
### 11.2 Metrics
An extracted field counts as correct only if its label and its exact word span match the ground truth.

$$\\text{precision} = \\frac{TP}{TP + FP}, \\quad \\text{recall} = \\frac{TP}{TP + FN}, \\quad F_1 = \\frac{2 \\cdot \\text{precision} \\cdot \\text{recall}}{\\text{precision} + \\text{recall}}$$

Counts are summed over all documents before dividing (**micro-averaging**).

**Familiar vs unfamiliar layouts.** `shot_groups` uses DocILE's definition: a test document is *k*-shot if *k* training documents share its template. Groups are zero-shot (0), few-shot (1–3) and many-shot (4+), and the headline diagnostic is Δ = F1(many-shot) − F1(zero-shot).
"""),
    ("train.py", """
## 12. Training loop

| Setting | Value | Reason |
|---|---|---|
| Optimiser | AdamW, learning rate 5·10⁻⁵, weight decay 0.01 | standard for fine-tuning transformers |
| Schedule | linear warm-up over 10% of steps, then linear decay | stable start from pretrained weights |
| Batch | 4 documents (up to 4 windows each) | fits the document-level loss in memory |
| Gradient clipping | norm 1.0 | guards against loss spikes |
| Structural weights | λ grows linearly from 0 over the first 20% of steps | the terms are meaningful only once the classifier commits to labels |
| Checkpoint | the final epoch | DocILE results are reported on its validation split, which therefore cannot also choose checkpoints |

Each step computes

$$L = L_{\\text{CE}} + \\text{ramp} \\cdot (\\lambda_1 L_{\\text{arith}} + \\lambda_2 L_{\\text{card}})$$

The arithmetic and cardinality terms are logged even when their weights are zero, which shows how consistent an unconstrained model is. After training, predictions for every split are saved and scored. Each run gets its own folder named after its settings, so runs never overwrite each other.
"""),
    ("evaluate.py", """
## 13. Evaluation, routing signals and risk–coverage

### 13.1 Evaluation
`evaluate_run` re-scores a finished run from its saved predictions (no GPU). For DocILE it also runs the **official evaluator**, which matches key-information fields by location and reports average precision and F1 per layout group.
"""),
    ("routing.py", """
### 13.2 Routing

**The question:** which documents can skip human review? A **risk signal** gives each document a score; documents are processed automatically from the lowest risk upward.

| Signal | Computed from | Needs the model? |
|---|---|---|
| **Layout novelty** | mean cosine distance of the document's 16×16 ink map to its 5 nearest training documents | no: boxes only, before the model runs |
| **MSP** | 1 − the lowest word confidence | yes |
| **Mahalanobis** | distance of the document embedding from the training embeddings' Gaussian (Ledoit–Wolf covariance) | yes |
| **Consistency** | number of broken constraints in the prediction | yes |
| **Oracle** | the true number of errors | reference only |

**Risk–coverage curve.** Sort documents by risk; for each **coverage** *c* (share auto-processed), compute F1 on the safest *c*. Documents with equal risk enter together, so ties never favour a signal.

**Worked example.** 4 documents with risks 0.1, 0.2, 0.2, 0.9 and per-document errors 0, 0, 1, 3. At coverage 0.25 only the first is processed (F1 = 1.0). At 0.75 the two tied documents enter together and include one error. The headline figure is the largest coverage whose F1 still meets a target (0.95): the **straight-through rate** that industry reports. **AURC** (area under the error–coverage curve) summarises the whole curve; lower is better.
"""),
    ("summarise.py", """
## 14. Result tables and the document viewer

`collect` groups runs that differ only in seed. Results are reported as mean ± standard deviation over seeds, because differences of one or two F1 points lie within seed noise.
"""),
    ("demo.py", """
`show` draws one document with each word coloured (green: correct field, red: wrong, amber: missed), lists the extracted fields with their confidence, checks the arithmetic, and prints the routing decision for that document.
"""),
]

TABLE_HELPER = '''
import pandas as pd


def results_table(runs_dir: Path, split: str) -> pd.DataFrame:
    """Mean ± std entity F1 (in %) per setting, over seeds."""
    rows = {}
    for setting, values in sorted(collect(Path(runs_dir), split).items()):
        rows[setting] = {c: f"{100 * np.mean(v):.2f} ± {100 * np.std(v, ddof=1 if len(v) > 1 else 0):.2f}"
                         for c, v in values.items()} | {"seeds": max(len(v) for v in values.values())}
    return pd.DataFrame(rows).T
'''

PART_B = [
    ("md", """
# Part B — the experiments

## 15. Data, calibration and noise floor

| Setting | Meaning |
|---|---|
| `USE_PUBLISHED_RESULTS` | `True`: fetch finished runs from the project's GitHub `results` branch, so Part B runs on a CPU. `False`: retrain everything (A100: about 7 minutes per CORD or WildReceipt run, 20 per DocILE run). |
| `RUN_DOCILE` | DocILE needs a free personal token from docile.rossum.ai; skip it to run CORD and WildReceipt only. |
"""),
    ("code", """
from pathlib import Path

DATA, RUNS = Path('data'), Path('runs')
USE_PUBLISHED_RESULTS = True
RUN_DOCILE = False
DOCILE_RAW = Path('/content/docile_raw')

if USE_PUBLISHED_RESULTS:
    !rm -rf /tmp/published && git clone -q --depth 1 --branch results https://github.com/AviDip/doc-extraction /tmp/published
    !mkdir -p runs data && cp -rn /tmp/published/runs/. runs/ && cp -rn /tmp/published/data/. data/
    print(len(list(RUNS.glob('*/metrics.json'))), 'finished runs fetched')
"""),
    ("md", """
### 15.1 Download and convert the datasets
CORD comes from the Hugging Face Hub, WildReceipt from OpenMMLab's public server; DocILE uses its official download script with your token (typed into a hidden prompt, never stored).
"""),
    ("code", """
prepare_cord(DATA)
prepare_cord_disjoint(DATA, seed=0, threshold=0.7)   # loose threshold: a shop rarely lands on both sides

!mkdir -p data/wildreceipt_raw && wget -q -nc https://download.openmmlab.com/mmocr/data/wildreceipt.tar -P data
!tar -xf data/wildreceipt.tar -C data/wildreceipt_raw --skip-old-files
WR_RAW = next(Path('data/wildreceipt_raw').rglob('class_list.txt')).parent
prepare_wildreceipt(WR_RAW, DATA, seed=0, threshold=DISTANCE_THRESHOLD)
"""),
    ("code", """
if RUN_DOCILE:
    from getpass import getpass
    token = getpass('DocILE token: ')
    !git clone -q --depth 1 https://github.com/rossumai/docile.git /tmp/docile_repo
    !bash /tmp/docile_repo/download_dataset.sh {token} annotated-trainval data 2>&1 | sed "s/{token}/***/g"
    !mkdir -p {DOCILE_RAW} && unzip -qn data/annotated-trainval.zip -d {DOCILE_RAW}
    prepare_docile(str(DOCILE_RAW), DATA)
"""),
    ("md", """
### 15.2 Calibrating the template threshold
On WildReceipt the replication rate should be close to zero at the right threshold (its test shops are unseen by construction). The same threshold then measures CORD's official split.
"""),
    ("code", """
def load_docs(path, *names):
    return [d for n in names for d in load_split(path / f'{n}.pkl')[2]]

calibration = {
    'WildReceipt test (should be ~0)': calibrate(load_docs(DATA / 'wildreceipt', 'train', 'validation'), load_docs(DATA / 'wildreceipt', 'test')),
    'CORD official test': calibrate(load_docs(DATA / 'cord', 'train'), load_docs(DATA / 'cord', 'test')),
}
pd.DataFrame({k: {r['threshold']: r['test_replication'] for r in v} for k, v in calibration.items()}).style.format('{:.1%}')
"""),
    ("md", "### 15.3 Noise floor\nThe share of documents whose ground-truth labels satisfy each equation; the consistency loss trains only on those."),
    ("code", """
datasets = ['cord', 'wildreceipt'] + (['docile'] if RUN_DOCILE else [])
floor = {name: analyse(DATA / name / 'train.pkl', model='layoutlmv3') for name in datasets}
rows = {(n, eq): v for n, r in floor.items() for eq, v in r['equations'].items()}
display(pd.DataFrame(rows).T.style.format({'satisfied_share': '{:.1%}', 'satisfied_share_text_parse': '{:.1%}'}))
pd.DataFrame({n: {'documents': r['documents'], 'fragmented fields': f"{r['fragmented_fields_share']:.1%}",
                  'longer than 512 tokens': f"{r['tokens']['longer_than_one_window']:.1%}",
                  'singleton fields': len(r['singleton_fields'])} for n, r in floor.items()})
"""),
    ("md", """
## 16. Training runs

Each entry is one `TrainConfig`; a run is skipped when its results already exist (published or trained earlier).

| Group | Runs | Question |
|---|---|---|
| CORD official, 3 seeds | baseline | reference accuracy and seed noise |
| Model ladder | RoBERTa, BROS | does the finding depend on architecture? |
| CORD disjoint | baseline (3 seeds), jitter 1–10%, key swap, λ ∈ {0.1, 0.3, 1.0} | tuning on unfamiliar layouts |
| WildReceipt | baseline and each method, 3 seeds | main results with an official unseen-layout test |
| DocILE | baseline and each method | large scale, official evaluator |
"""),
    ("code", """
CORD, WR = ['validation', 'test'], ['validation', 'test']
BEST = dict(jitter=0.05, key_swap=0.5, lambda_arith=0.3, lambda_card=0.3)   # chosen from CORD-disjoint validation F1
ARMS = {'baseline': {}, 'jitter': {'jitter': BEST['jitter']}, 'key_swap': {'key_swap': BEST['key_swap']},
        'consistency': {'lambda_arith': BEST['lambda_arith'], 'lambda_card': BEST['lambda_card']}, 'all': BEST}

experiments = [TrainConfig(data='data/cord', eval_splits=CORD, seed=s) for s in (0, 1, 2)]
experiments += [TrainConfig(data='data/cord', eval_splits=CORD, model=m) for m in ('roberta', 'bros')]
experiments += [TrainConfig(data='data/cord_disjoint', eval_splits=CORD, seed=s) for s in (0, 1, 2)]
experiments += [TrainConfig(data='data/cord_disjoint', eval_splits=CORD, jitter=j) for j in (0.01, 0.03, 0.05, 0.1)]
experiments += [TrainConfig(data='data/cord_disjoint', eval_splits=CORD, key_swap=0.5)]
experiments += [TrainConfig(data='data/cord_disjoint', eval_splits=CORD, lambda_arith=w, lambda_card=w) for w in (0.1, 0.3, 1.0)]
experiments += [TrainConfig(data='data/wildreceipt', eval_splits=WR, seed=s, **a) for a in ARMS.values() for s in (0, 1, 2)]
if RUN_DOCILE:
    experiments += [TrainConfig(data='data/docile', eval_splits=['val'], epochs=6, **a) for a in ARMS.values()]

todo = [c for c in experiments if not (RUNS / c.run_name() / 'metrics.json').exists()]
print(f'{len(experiments) - len(todo)} finished, {len(todo)} to train')
if todo and not torch.cuda.is_available():
    print('No GPU: training skipped; the results below use the finished runs.')
else:
    for cfg in todo:
        train(cfg)
"""),
    ("md", "## 17. Results\n\n### 17.1 Extraction F1 (%, mean ± std over seeds)\nOn CORD and WildReceipt the test split holds the unfamiliar layouts (disjoint split, or WildReceipt's unseen shops); the validation split of WildReceipt holds familiar ones."),
    ("code", """
display(results_table(RUNS, 'test'))
results_table(RUNS, 'validation')
"""),
    ("md", "### 17.2 Routing: which documents can skip review?"),
    ("code", """
RUN = RUNS / 'cord_layoutlmv3_j0.0_k0.0_a0.0_c0.0_s0'
if not (RUN / 'routing_test' / 'summary.json').exists():
    route(RUN, 'test', target=0.95)
display(Image(str(RUN / 'routing_test' / 'risk_coverage.png')))
pd.DataFrame(json.loads((RUN / 'routing_test' / 'summary.json').read_text())['signals']).T.round(3)
"""),
    ("md", "### 17.3 Worked examples\nThe riskiest document by the Mahalanobis signal, then the safest."),
    ("code", """
show(RUN, 'test', pick(RUN, 'test', 'riskiest', 1)[0])
"""),
    ("code", """
show(RUN, 'test', pick(RUN, 'test', 'safest', 1)[0])
"""),
    ("md", """
### 17.4 DocILE official evaluation
With DocILE data available, the official evaluator scores key-information fields by location, per layout group.
"""),
    ("code", """
if RUN_DOCILE:
    for run in sorted(RUNS.glob('docile_*')):
        report = evaluate_run(run, str(DOCILE_RAW))['val']['docile_kile']
        print(run.name, {k: round(v['f1'], 4) for k, v in report.items() if isinstance(v, dict)})
"""),
    ("md", """
## Reproducibility

| Item | Where |
|---|---|
| Code (this notebook is generated from it) | github.com/AviDip/doc-extraction, package `docext/` |
| Published runs, logs, reports | the same repository, branch `results` |
| Unit tests (22, offline) | `tests/` in the repository |
| Datasets | CORD (CC BY 4.0, Hugging Face), WildReceipt (OpenMMLab), DocILE (token, docile.rossum.ai) |
"""),
]


def cell(kind: str, source: str, tags: list[str] | None = None) -> dict:
    lines = source.strip("\n").split("\n")
    c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if tags:
        c["metadata"]["tags"] = tags
    if c["cell_type"] == "code":
        c |= {"execution_count": None, "outputs": []}
    return c


def build() -> dict:
    cells = [cell("md", INTRO), cell("md", SETUP), cell("code", SETUP_CODE)]
    seen: dict[str, str] = {}
    for module, text in SECTIONS:
        code = inline(module)
        if module == "summarise.py":
            code += "\n\n" + TABLE_HELPER.strip() + "\n"
        # evaluate's run-level load_predictions intentionally replaces the file-level reader (kept as read_predictions)
        allowed = {"load_predictions"}
        clashes = {n: seen[n] for n in top_level_names(code) if n in seen and n not in allowed}
        if clashes:
            raise SystemExit(f"{module} redefines {clashes}")
        seen |= {n: module for n in top_level_names(code)}
        cells += [cell("md", text), cell("code", code, ["library"])]
    cells += [cell("code", "import json\nfrom IPython.display import Image, display")]
    cells += [cell(kind, src) for kind, src in PART_B]
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    nb = build()
    (ROOT / "project.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"project.ipynb: {len(nb['cells'])} cells")
