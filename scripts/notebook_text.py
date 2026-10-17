"""Text of project.ipynb: the explanations around each piece of code.

Each section names a module of docext/; its code is split into small cells at
the listed definitions, each introduced by the text given for it.
"""

INTRO = r"""
# Making Document Extraction Worth Automating

**Avishek Paul Dipto (a1988440) · Programming for AI and ML 2 · University of Adelaide**

A company receives a thousand invoices a month. A model reads them with over 90% of fields correct, yet staff still check almost every one, because nobody knows *which* forty are wrong. High accuracy saves little when the errors cannot be found.

This notebook builds and tests two ideas:

| Goal | Idea |
|---|---|
| **Make fewer errors on unfamiliar layouts** | Train with small random changes to the page and wording, and add a check that the extracted amounts add up. |
| **Find the errors that remain, without labels** | Score each document's risk; send only risky ones to a person. |

**How to run.** Run the cells from top to bottom. Part A builds the code, one small piece at a time. Part B uses it. Only the datasets are downloaded, from their public sources. By default, Part B reuses finished training runs from the project's GitHub repository, so everything can be reproduced on an ordinary computer in minutes.

| | Section |
|---|---|
| 0 | Setup |
| **A** | **The code** |
| 1 | How a document is represented |
| 2 | Putting words in reading order |
| 3 | Reading amounts |
| 4 | Varying the training data |
| 5 | Finding documents that share a layout |
| 6 | Loading the datasets |
| 7 | Checking that amounts add up |
| 8 | How often the real labels pass the check |
| 9 | The three models |
| 10 | Feeding documents to a model |
| 11 | Predictions and scores |
| 12 | Training |
| 13 | Deciding which documents need a person |
| 14 | Tables and a document viewer |
| **B** | **The experiments** |
| 15 | Data, layout check and label check |
| 16 | Training runs |
| 17 | Results |
"""

SETUP = r"""
## 0. Setup

| Library | Used for |
|---|---|
| `torch` | numbers on the GPU and training |
| `transformers` | the pretrained models and their tokenizers |
| `datasets` | downloading CORD |
| `docile-benchmark` | reading DocILE and its official scorer |
| `scikit-learn` | grouping documents, nearest neighbours |
| `numpy`, `matplotlib`, `pandas` | arrays, figures, tables |

Colab already has `numpy`, `pandas` and `torch`; their versions are kept as they are, so nothing else on Colab breaks.
"""

SETUP_CODE = r"""
!pip freeze | grep -iE '^(numpy|pandas|torch|scipy)==' > /tmp/pins.txt
!pip install -q "transformers>=4.45" "datasets>=2.19" "docile-benchmark>=0.3" scikit-learn matplotlib -c /tmp/pins.txt

import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (Part B reuses finished runs)')
"""

SECTIONS = [
    {
        "module": "schema.py",
        "text": r"""
# Part A — the code

## 1. How a document is represented

The three datasets store documents in different ways. The first step turns each into the same simple form, so the rest of the code works on all of them.

| Part | What it is |
|---|---|
| **Word** | One piece of text found on the page, e.g. `TOTAL` |
| **Box** | Where the word sits: left, top, right, bottom, as fractions of the page (0 to 1), so small and large pages compare directly |
| **Field** | One or more words with a meaning, e.g. the total amount |
| **Label** | One per word: `B-name` starts a field, `I-name` continues it, `O` is not part of any field |

Labelling every word turns "find the fields" into "give each word a label", which a model learns well.

**Example.** The line `TOTAL 50,000` has two words. `TOTAL` only names the field, so it is `O`. `50,000` is the value: `B-total.total_price`. Had the amount been read as two words, `50` and `000`, the second would be `I-total.total_price`.
""",
        "chunks": {
            "LabelSpace": "Models work with numbers, so `LabelSpace` numbers the labels: `O` is 0, then `B-` and `I-` for each field in turn. With 30 fields there are 61 labels.",
            "tags_to_entities": "Going back from word labels to fields: `O, B-total, I-total` becomes one field, *total*, covering words 1 and 2. An `I-` label that does not continue the field before it starts a new one.",
            "save_split": "Converted datasets are saved to disk, so each is prepared only once.",
        },
    },
    {
        "module": "layout.py",
        "text": r"""
## 2. Putting words in reading order

The OCR software, which reads text from page images, often returns words grouped in blocks. A label such as `Total` and its amount can then end up far apart in the list. Sorting the words the way a person reads, top to bottom and then left to right, keeps them together.

Two words share a line when their vertical centres are less than half a word's height apart.

**Example.** Words with centres at heights 0.100, 0.105 and 0.300 on a page where words are 0.02 tall: the first two differ by 0.005, less than 0.01, so they form line 0; the third starts line 1.
""",
        "chunks": {
            "build_document": "Labels are given *after* sorting, so `B-` always marks a field's first word in reading order. The function also counts fields split apart by other words, which happens with text in columns; this is reported later as a measure of data quality.",
            "jitter_boxes": "**Page jitter.** During training, each page is moved and resized a little at random, by up to a chosen share of the page. All words move together: rows and columns stay intact, but a field never sits at exactly the same spot twice. This stops the model from simply memorising where fields sit on layouts it has seen. Moving words one by one would scramble the lines, which are only about 1.5% of the page apart.",
        },
    },
    {
        "module": "numbers.py",
        "text": r"""
## 3. Reading amounts

The check in Section 7 needs the value of every number on the page, and documents write numbers in different ways:

| Written | Value |
|---|---|
| `1,234.56` | 1234.56 |
| `1.234,56` (European style) | 1234.56 |
| `45,000` (Indonesian receipts) | 45000 |
| `12,50` | 12.5 |
| `(5.00)` or `5.00-` | −5 |
| `12.03.2021`, `21%`, `12:30` | not an amount |

The rule: if both `,` and `.` appear, the last one is the decimal point. If only one appears, it separates thousands when exactly three digits follow it, and is the decimal point otherwise. Thousand groups must have exactly three digits, which rules out dates.
""",
        "chunks": {
            "word_values": "Sometimes `1 234,56` is read as two words. When a word of one to three digits is followed on the same line by one starting with three digits, the two are read together and the value is put on the second word. Adding up the values of a field's words then still gives the right amount.",
        },
    },
    {
        "module": "augment.py",
        "text": r"""
## 4. Varying the training data

Documents print a **key** before most values: `Total:`, `Amount due`, `Celkem`. A model can learn to rely on one exact key and fail when another company writes it differently.

Neither dataset marks keys, so a key is taken to be the words just left of a value on the same line, if they contain letters and no digits.
""",
        "chunks": {
            "key_vocabulary": "Collecting every key seen at least twice, per field, gives a list of real wordings: for the total, `TOTAL`, `GRAND TOTAL`, `Total Due`, and so on.",
            "swap_keys": "With a chosen probability, each key is replaced by another wording of the same field. The new words take the old key's place, sharing its width by length.\n\n**Example.** `TOTAL 60,500` can become `GRAND TOTAL 60,500`. The amount and its label stay the same, so the model learns to find the total whichever wording is used.",
            "Augmenter": "`Augmenter` applies both changes, key swapping and page jitter, to a training document. It is used only while training, never when testing.",
        },
    },
    {
        "module": "split.py",
        "text": r"""
## 5. Finding documents that share a layout

Receipts from the same shop look alike: same header, same wording, same arrangement. If the test set contains shops that also appear in training, a model can score well by remembering where things sit, and the score says little about new shops. Laatiri et al. (2023) found this in three of every four test documents of the SROIE dataset.

**How shops are recognised.** Each document is described by the words in its first five lines (usually the shop's name and address) and by its keys. Which fields appear is deliberately left out: almost every receipt has a total and a tax line, so that would make all shops look alike. An earlier version made exactly this mistake and wrongly found most of WildReceipt's new shops in training.

**How alike two documents are.** The share of words they do *not* have in common:

$$\text{distance} = 1 - \frac{\text{words in both}}{\text{words in either}}$$

**Example.** `{safeway, grocery, total}` and `{safeway, total}` share 2 of 3 words: distance 1 − 2/3 = 0.33.

Documents are then grouped step by step, always joining the closest groups first, until the remaining groups are further apart than a set threshold. Each group is one layout.
""",
        "chunks": {
            "replication_rate": "With layouts known, the code can measure how many test documents share a layout with training (the **reuse rate**), and can build splits where no layout appears on both sides. `seen_template_holdout` does the opposite: it picks documents whose layout *does* stay in training, to measure accuracy on familiar layouts.",
            "calibrate": "**Choosing the threshold.** WildReceipt's authors guarantee that its test shops never appear in training, so on WildReceipt the correct threshold should find almost no reuse. `calibrate` measures reuse at several thresholds; the chosen one (0.5) is the loosest that keeps WildReceipt's reuse near zero.",
        },
    },
    {
        "module": "data/cord.py",
        "text": r"""
## 6. Loading the datasets

| Dataset | Documents | Fields | Familiar vs new layouts | Source |
|---|---|---|---|---|
| **CORD** | 1,000 receipts | 30 | built here (Section 5) | Hugging Face |
| **WildReceipt** | 1,740 receipts | 12 | test shops are new by design | OpenMMLab |
| **DocILE** | 5,680 invoices | 55 | official layout groups | DocILE, free token |

Each loader produces the documents of Section 1.

**CORD.** Each line of a receipt has a category. Words marked as keys become `O`, so only values are targets, as in the other datasets.
""",
        "chunks": {},
    },
    {
        "module": "data/wildreceipt.py",
        "text": "**WildReceipt.** Each text box is one of 25 classes: a value or a key for 12 fields, or *other*. Value boxes become fields; keys and *other* become `O`. The text has its spaces removed (`THANKYOU`), so each box is treated as one word.",
        "chunks": {},
    },
    {
        "module": "data/docile.py",
        "text": "**DocILE.** Fields and words come as separate boxes. A word belongs to a field when at least half of the word lies inside the field's box. When a word falls inside two fields, it takes the one it overlaps most, and these cases are counted.",
        "chunks": {
            "convert_document": "One document at a time: for each page, find the field each word overlaps most, then put the words in reading order.",
            "to_kile_fields": "DocILE's official scorer compares fields by *position*, so each predicted field is turned back into one box around its words.",
        },
    },
    {
        "module": "prepare.py",
        "text": r"""
These functions download and convert each dataset once. For CORD they also build a split with no shared layouts. For WildReceipt they set aside 100 training receipts from shops that stay in training, giving one set of familiar layouts (validation) and one of new layouts (test). The difference between the two scores is the **layout gap**:

$$\Delta = F_1(\text{familiar layouts}) - F_1(\text{new layouts})$$
""",
        "chunks": {
            "prepare_cord_disjoint": "**CORD without shared layouts.** All 1,000 receipts are grouped by layout, and whole groups are dealt to test, validation and training, so no layout appears in two places. The report also gives the reuse rate of CORD's official split.",
            "prepare_wildreceipt": "**WildReceipt.** The official test set already has only new shops; 100 training receipts from shops that stay in training become the familiar-layout set.",
            "prepare_docile": "**DocILE.** Converts the official training and validation sets and records how many words fell inside two fields or none.",
        },
    },
    {
        "module": "constraints.py",
        "text": r"""
## 7. Checking that amounts add up

This is the project's main idea. A normal model learns each word's label on its own; nothing tells it that the items, tax and total of a receipt must agree. Here, that rule becomes part of training.

**Turning the rule into a number.** The numbers on the page are fixed; only the labels are uncertain. If the model is 70% sure a word is the total, that word counts 70% towards the total. Adding this up over all words gives the model's **expected total**:

$$\text{expected value of a field} = \sum_{\text{words}} P(\text{word is in the field}) \times \text{word's value}$$

Because this changes smoothly with the model's probabilities, training can reduce any mismatch.

**Example.** Items 25,000 and 5,000, tax 3,000, total 33,000. The model is 50% sure the real total is the total, and also 50% sure, wrongly, that the 25,000 item is. Its expected total is 0.5 × 33,000 + 0.5 × 25,000 = 29,000, while items plus tax give 33,000: a 12% mismatch. Training lowers it by moving belief away from the item price.
""",
        "chunks": {
            "equations_for": "Only rules whose fields exist in the dataset are used. `coefficient_matrix` writes each rule as +1 for the parts and −1 for the total, so all rules can be checked in one step.",
            "field_sums": "**Which receipts to use.** Some receipts do not add up even with correct labels (discounts, rounding, reading errors). The rule is applied only to receipts whose correct labels pass it, with a 1% allowance for rounding.",
            "singleton_fields": "**The second rule.** Key fields such as the total appear at most once. A field counts as \"at most once\" when 95% of training documents agree. Missing fields are fine: not every invoice has a due date.",
            "ConsistencyLoss": r"""**The penalty added to training.**

| Part | What it penalises |
|---|---|
| Sum rule | the mismatch between expected parts and expected total, as a share of the total |
| Once rule | any expected count above one for a field that appears at most once |

Choices that keep it stable:
- Each word's value is divided by the document's total, and very large ratios are capped at 10, so invoice numbers or dates read as numbers cannot dominate.
- The penalty grows gently for large mismatches, so one odd document cannot swing training.
- Both rules are satisfied when nothing is labelled at all, so they cannot replace normal training; they are added on top of it, with a small weight.""",
            "count_violations": "The same rules, applied to a finished prediction, count how many it breaks. This needs no correct labels, so it becomes one of the risk scores in Section 13.",
        },
    },
    {
        "module": "analyse.py",
        "text": r"""
## 8. How often the real labels pass the check

Before training with the rule, it is worth knowing how often the correct labels themselves satisfy it. If they rarely did, the rule would have little to learn from. `analyse` reports, for each rule, how many documents it applies to and the share that pass. It also counts fields split by reading order and documents too long for the model's 512-token limit.
""",
        "chunks": {},
    },
    {
        "module": "models.py",
        "text": r"""
## 9. The three models

All three are pretrained on millions of documents and then trained further here. They differ in how much they know about position on the page:

| Model | Knows about position | Role |
|---|---|---|
| **RoBERTa** | nothing: reads text only | baseline without layout |
| **LayoutLMv3** | where each word sits, and how far apart words are | main model |
| **BROS** | how each word sits relative to every other word | comparison |

Comparing them shows how much each kind of layout knowledge helps. LayoutLMv3 is used without the page image, so all three receive the same input, and the image cannot reveal the exact positions that jitter hides.

The first lines record, for each model, where to download it and what form of box it expects.
""",
        "chunks": {
            "Extractor": "`Extractor` gives the three models one interface: boxes always go in as fractions of the page and are converted to whatever each model expects.",
        },
    },
    {
        "module": "encoding.py",
        "text": r"""
## 10. Feeding documents to a model

**Word pieces.** Models split words into smaller pieces, called tokens: `45,000` may become `45`, `,`, `000`. Each word is labelled through its first piece; the other pieces are ignored when learning.

**Windows.** The models read at most 512 tokens at a time. Longer documents (about 40% of the invoices) are read in overlapping windows.

**Example.** A 1,200-token invoice is read as three windows starting at tokens 0, 382 and 690, with 128 tokens shared between neighbours. A word read twice gets the average of both predictions.
""",
        "chunks": {
            "Collator": "`Collator` turns a handful of documents into one batch for the model: the windows, their boxes and labels, which document each word belongs to, the value of every word, and which sum rules hold for each document. While training, at most four windows per document are used per step; documents cut short this way are left out of the two rules.",
            "pool_words": "`pool_words` gathers the predictions from all windows back into one prediction per word. The sum rule needs this, because an invoice's items and its total often sit in different windows.",
        },
    },
    {
        "module": "predict.py",
        "text": r"""
## 11. Predictions and scores

For every word, `predict` keeps the predicted label and how sure the model is of it (its **confidence**: the highest probability it gave). For every document it also keeps a **description**: the average of the model's internal numbers over the document's words. Section 13 uses both.

On the GPU, numbers are stored with half the usual precision, which is faster and uses less memory. Predictions are saved as plain number arrays, which are safe to share: opening them cannot run hidden code.
""",
        "chunks": {
            "predict": "`predict` runs the model over documents, 32 windows at a time and without learning, and keeps the label, confidence and description described above.",
            "save_predictions": "Saving and reading predictions.",
        },
    },
    {
        "module": "metrics.py",
        "text": r"""
**Scoring.** A field counts as correct only if both its name and its exact words match the answer.

| Measure | Question it answers |
|---|---|
| Precision | Of the fields found, how many are right? |
| Recall | Of the real fields, how many were found? |
| F1 | Both together: $2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$ |

Correct and wrong fields are counted over all documents first, then the measures are computed once.

""",
        "chunks": {
            "shot_groups": "**Familiar or new layout.** As in DocILE, a test document is *new* (zero-shot) if no training document shares its layout, *rare* (few-shot) if one to three do, and *familiar* (many-shot) if four or more do. The layout gap is the score on familiar layouts minus the score on new ones.",
        },
    },
    {
        "module": "train.py",
        "text": r"""
## 12. Training

| Setting | Value | Why |
|---|---|---|
| Optimiser | AdamW, step size 5·10⁻⁵ | the usual choice for training pretrained models further |
| Step size over time | rises over the first 10% of steps, then falls to zero | avoids large early changes to pretrained knowledge |
| Batch | 4 documents | the sum rule needs whole documents |
| Rule weights | start at zero, reach full weight after 20% of steps | the rules only make sense once the model starts choosing labels |
| Kept model | the last one | DocILE is scored on its validation set, so that set cannot also be used to pick a model |

Each step minimises

$$\text{loss} = \text{label loss} + \text{ramp} \times (\lambda_1 \cdot \text{sum rule} + \lambda_2 \cdot \text{once rule})$$
""",
        "chunks": {
            "AugmentedDocs": "Training documents are varied afresh every time they are read, so the model never sees exactly the same page twice. Fixing the random seed makes each run repeatable.",
            "train": "The training loop. Both rules are measured even when their weight is zero, which shows how well an ordinary model already satisfies them. At the end, predictions for every split are saved and scored. Each run is saved in a folder named after its settings, so runs never overwrite each other.",
        },
    },
    {
        "module": "evaluate.py",
        "text": r"""
## 13. Deciding which documents need a person

First, reading back a finished run: its settings and saved predictions.
""",
        "chunks": {
            "docile_official": "`evaluate_run` scores a finished run again from its saved predictions, without a GPU. For DocILE it also runs the official scorer.",
        },
    },
    {
        "module": "routing.py",
        "text": r"""
A **risk score** ranks documents from safest to riskiest. The safest are processed automatically; the rest go to a person. Four scores are compared, none of which looks at the correct answers:

| Score | Asks |
|---|---|
| **Layout novelty** | Does this page look unlike every training page? (uses only word positions, before the model runs) |
| **Confidence** | How unsure was the model about its least certain word? |
| **Mahalanobis** | How unusual is the document's description compared with training documents? |
| **Rules** | How many of the Section 7 rules does the prediction break? |
| **Perfect ranking** | The true number of mistakes, as a best-possible reference |
""",
        "chunks": {
            "msp_risk": "**Confidence.** One minus the lowest confidence on any word of the document.",
            "mahalanobis": "**Mahalanobis.** Measures how far a document's description lies from the typical training document. Unlike a plain distance, it takes into account which directions normally vary a lot and which hardly vary.",
            "consistency_risk": "**Rules.** The number of Section 7 rules the prediction breaks.",
            "risk_coverage": r"""**Comparing scores.** Sort documents by risk and process the safest share automatically: that share is the **coverage**. A good score keeps accuracy high while covering many documents.

**Example.** Four documents with risks 0.1, 0.2, 0.2, 0.9 and 0, 0, 1, 3 mistakes. Processing 25% means the first document only: no mistakes. At 75%, the two documents with equal risk come in together, bringing one mistake. Equal risks always enter together, so ties never favour a score.

The headline number is the largest coverage whose F1 still reaches 0.95: the share of documents that could skip a person. The area under the error curve (lower is better) sums up the whole curve.""",
            "plot": "Drawing the curves and running the full comparison for one run.",
        },
    },
    {
        "module": "summarise.py",
        "text": r"""
## 14. Tables and a document viewer

Runs that differ only in their random seed are grouped, and results are shown as average ± spread. One or two points of F1 can come from the seed alone, so single runs are not compared.
""",
        "chunks": {},
    },
    {
        "module": "demo.py",
        "text": "To look at single documents, each word is first marked as part of a field found correctly, found wrongly, or missed.",
        "chunks": {
            "auto_threshold": "The risk cut-off is the risk of the last document still processed automatically at the chosen coverage. `pick` lists the riskiest or safest documents, or those with mistakes.",
            "show": "`show` draws one document with each word coloured (green: found correctly, red: wrong, amber: missed), lists the fields found with the model's confidence, checks the sums, and says whether the document would go to a person.",
        },
    },
]

TABLE_HELPER = r'''
import pandas as pd


def results_table(runs_dir: Path, split: str) -> pd.DataFrame:
    """Average ± spread of entity F1 (in %) per setting, over seeds."""
    rows = {}
    for setting, values in sorted(collect(Path(runs_dir), split).items()):
        rows[setting] = {c: f"{100 * np.mean(v):.2f} ± {100 * np.std(v, ddof=1 if len(v) > 1 else 0):.2f}"
                         for c, v in values.items()} | {"seeds": max(len(v) for v in values.values())}
    return pd.DataFrame(rows).T
'''

PART_B = [
    ("md", r"""
# Part B — the experiments

## 15. Data, layout check and label check

Part B decides by itself what to do:

| Situation | What happens |
|---|---|
| Finished runs exist in the project's GitHub repository (`results` branch) | they are reused; no account is needed |
| A GPU is available | any run not yet finished is trained |
| No GPU | training is skipped and the finished runs are used |
| A DocILE token is found | DocILE is downloaded and included; otherwise it is skipped |

**The DocILE token** (free, from docile.rossum.ai) is read from, in order: the `DOCILE_TOKEN` setting below, Colab's **Secrets** panel (the key icon on the left; add a secret named `DOCILE_TOKEN`), or a file `docile_token.txt` uploaded next to the notebook. Keep the token out of any copy you share.
"""),
    ("code", r"""
import os
from pathlib import Path

DOCILE_TOKEN = ''   # optional: paste your DocILE token here (in your own copy only)

DATA, RUNS = Path('data'), Path('runs')
DOCILE_RAW = Path('/content/docile_raw')
HAS_GPU = torch.cuda.is_available()


def find_docile_token() -> str | None:
    if DOCILE_TOKEN.strip():
        return DOCILE_TOKEN.strip()
    try:
        from google.colab import userdata
        return userdata.get('DOCILE_TOKEN')
    except Exception:  # not on Colab, or no such secret
        pass
    if os.environ.get('DOCILE_TOKEN'):
        return os.environ['DOCILE_TOKEN']
    for file in (Path('docile_token.txt'), Path('/content/docile_token.txt')):
        if file.exists():
            return file.read_text().strip()
    return None


TOKEN = find_docile_token()
RUN_DOCILE = bool(TOKEN)
print(f"GPU: {'yes' if HAS_GPU else 'no'} · DocILE: {'included' if RUN_DOCILE else 'skipped (no token found)'}")

# finished runs from the project's public results branch
!rm -rf /tmp/published && git clone -q --depth 1 --branch results https://github.com/AviDip/doc-extraction /tmp/published
!mkdir -p runs data && cp -rn /tmp/published/runs/. runs/ && cp -rn /tmp/published/data/. data/
print(len(list(RUNS.glob('*/metrics.json'))), 'finished runs reused')
"""),
    ("md", "### 15.1 Downloading the data\nCORD comes from Hugging Face and WildReceipt from OpenMMLab. DocILE uses its official download script; the token is hidden from all printed output."),
    ("code", r"""
prepare_cord(DATA)
prepare_cord_disjoint(DATA, seed=0, threshold=0.7)   # a looser threshold, so one shop rarely ends up on both sides

!mkdir -p data/wildreceipt_raw && wget -q -nc https://download.openmmlab.com/mmocr/data/wildreceipt.tar -P data
!tar -xf data/wildreceipt.tar -C data/wildreceipt_raw --skip-old-files
WR_RAW = next(Path('data/wildreceipt_raw').rglob('class_list.txt')).parent
prepare_wildreceipt(WR_RAW, DATA, seed=0, threshold=DISTANCE_THRESHOLD)
"""),
    ("code", r"""
if RUN_DOCILE:
    !git clone -q --depth 1 https://github.com/rossumai/docile.git /tmp/docile_repo 2>/dev/null || true
    !bash /tmp/docile_repo/download_dataset.sh {TOKEN} annotated-trainval data 2>&1 | sed "s/{TOKEN}/***/g" | tail -2
    !mkdir -p {DOCILE_RAW} && unzip -qn data/annotated-trainval.zip -d {DOCILE_RAW}
    prepare_docile(str(DOCILE_RAW), DATA)
"""),
    ("md", "### 15.2 How often test layouts already appear in training\nOn WildReceipt, whose test shops are new by design, this should be close to zero at the chosen threshold (0.5). The same threshold then measures CORD's official split."),
    ("code", r"""
def load_docs(path, *names):
    return [d for n in names for d in load_split(path / f'{n}.pkl')[2]]

calibration = {
    'WildReceipt test (should be ~0)': calibrate(load_docs(DATA / 'wildreceipt', 'train', 'validation'), load_docs(DATA / 'wildreceipt', 'test')),
    'CORD official test': calibrate(load_docs(DATA / 'cord', 'train'), load_docs(DATA / 'cord', 'test')),
}
pd.DataFrame({k: {r['threshold']: r['test_replication'] for r in v} for k, v in calibration.items()}).style.format('{:.1%}')
"""),
    ("md", "### 15.3 How often the real labels pass the sum rule"),
    ("code", r"""
datasets = ['cord', 'wildreceipt'] + (['docile'] if RUN_DOCILE else [])
floor = {name: analyse(DATA / name / 'train.pkl', model='layoutlmv3') for name in datasets}
rows = {(n, eq): v for n, r in floor.items() for eq, v in r['equations'].items()}
display(pd.DataFrame(rows).T.style.format({'satisfied_share': '{:.1%}', 'satisfied_share_text_parse': '{:.1%}'}))
pd.DataFrame({n: {'documents': r['documents'], 'fields split by reading order': f"{r['fragmented_fields_share']:.1%}",
                  'longer than 512 tokens': f"{r['tokens']['longer_than_one_window']:.1%}",
                  'fields that appear once': len(r['singleton_fields'])} for n, r in floor.items()})
"""),
    ("md", r"""
## 16. Training runs

Each run is described by one `TrainConfig`. Runs whose results already exist are skipped.

| Group | Runs | Question |
|---|---|---|
| CORD, official split | 3 seeds | baseline accuracy, and how much results vary by seed |
| CORD, other models | RoBERTa, BROS | does the kind of layout knowledge matter? |
| CORD, no shared layouts | 3 seeds; jitter 1–10%; key swap; rule weights 0.1, 0.3, 1.0 | choosing settings on new layouts |
| WildReceipt | each method, 3 seeds | main results |
| DocILE | each method; RoBERTa and BROS | larger scale, official scorer |
"""),
    ("code", r"""
CORD, WR = ['validation', 'test'], ['validation', 'test']
BEST = dict(jitter=0.05, key_swap=0.5, lambda_arith=0.1, lambda_card=0.1)   # best of each on CORD validation, new layouts
ARMS = {'baseline': {}, 'jitter': {'jitter': BEST['jitter']}, 'key_swap': {'key_swap': BEST['key_swap']},
        'rules': {'lambda_arith': BEST['lambda_arith'], 'lambda_card': BEST['lambda_card']}, 'all': BEST}

experiments = [TrainConfig(data='data/cord', eval_splits=CORD, seed=s) for s in (0, 1, 2)]
experiments += [TrainConfig(data='data/cord', eval_splits=CORD, model=m) for m in ('roberta', 'bros')]
experiments += [TrainConfig(data='data/cord_disjoint', eval_splits=CORD, seed=s) for s in (0, 1, 2)]
experiments += [TrainConfig(data='data/cord_disjoint', eval_splits=CORD, jitter=j) for j in (0.01, 0.03, 0.05, 0.1)]
experiments += [TrainConfig(data='data/cord_disjoint', eval_splits=CORD, key_swap=0.5)]
experiments += [TrainConfig(data='data/cord_disjoint', eval_splits=CORD, lambda_arith=w, lambda_card=w) for w in (0.1, 0.3, 1.0)]
experiments += [TrainConfig(data='data/wildreceipt', eval_splits=WR, seed=s, **a) for a in ARMS.values() for s in (0, 1, 2)]
if RUN_DOCILE:
    experiments += [TrainConfig(data='data/docile', eval_splits=['val'], epochs=6, **a) for a in ARMS.values()]
    experiments += [TrainConfig(data='data/docile', eval_splits=['val'], epochs=6, model=m) for m in ('roberta', 'bros')]

todo = [c for c in experiments if not (RUNS / c.run_name() / 'metrics.json').exists()]
print(f'{len(experiments) - len(todo)} finished, {len(todo)} to train')
if todo and not torch.cuda.is_available():
    print('No GPU: training skipped; the results below use the finished runs.')
else:
    for cfg in todo:
        train(cfg)
"""),
    ("md", "## 17. Results\n\n### 17.1 F1 (%, average ± spread over seeds)\nFor CORD and WildReceipt, *test* holds new layouts; WildReceipt's *validation* holds familiar ones."),
    ("code", r"""
display(results_table(RUNS, 'test'))
results_table(RUNS, 'validation')
"""),
    ("md", "### 17.2 Which documents could skip a person?"),
    ("code", r"""
RUN = RUNS / 'cord_layoutlmv3_j0.0_k0.0_a0.0_c0.0_s0'
if not (RUN / 'routing_test' / 'summary.json').exists():
    route(RUN, 'test', target=0.95)
display(Image(str(RUN / 'routing_test' / 'risk_coverage.png')))
pd.DataFrame(json.loads((RUN / 'routing_test' / 'summary.json').read_text())['signals']).T.round(3)
"""),
    ("code", r"""
coverage = {run.parent.parent.name + ' · ' + run.parent.name.removeprefix('routing_'):
            {k: v['coverage_at_target'] for k, v in json.loads(run.read_text())['signals'].items()}
            for run in sorted(RUNS.glob('*/routing_*/summary.json'))}
print('Share of documents that can skip a person while F1 stays at the target:')
pd.DataFrame(coverage).T.style.format('{:.0%}')
"""),
    ("md", "### 17.3 Two documents up close\nThe riskiest document by the Mahalanobis score, then the safest."),
    ("code", "show(RUN, 'test', pick(RUN, 'test', 'riskiest', 1)[0])"),
    ("code", "show(RUN, 'test', pick(RUN, 'test', 'safest', 1)[0])"),
    ("md", "### 17.4 DocILE official scores\nWith DocILE downloaded, the official scorer compares fields by position, for each layout group."),
    ("code", r"""
if RUN_DOCILE:
    for run in sorted(RUNS.glob('docile_*')):
        report = evaluate_run(run, str(DOCILE_RAW))['val']['docile_kile']
        print(run.name, {k: round(v['f1'], 4) for k, v in report.items() if isinstance(v, dict)})
"""),
    ("md", r"""
## Where everything lives

| Item | Location |
|---|---|
| Code (this notebook is built from it) | github.com/AviDip/doc-extraction, folder `docext/` |
| Finished runs, logs and reports | same repository, branch `results` |
| Tests (23, run without internet) | `tests/` |
| Datasets | CORD (CC BY 4.0), WildReceipt (OpenMMLab), DocILE (token from docile.rossum.ai) |
"""),
]
