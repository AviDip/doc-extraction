# Making Document Extraction Worth Automating

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AviDip/doc-extraction/blob/main/project.ipynb)

Avishek Paul Dipto (a1988440) · Programming for AI and ML 2 · University of Adelaide

Document extraction models read invoices and receipts with over 90% of fields correct, yet most documents are still checked by hand, because nobody knows which outputs are wrong. This project trains models to make fewer mistakes on unfamiliar layouts, and tests which risk scores best decide which documents can skip a person.

**Everything is in [`project.ipynb`](project.ipynb)**: the full code with explanations (Part A) and the experiments (Part B).

## Running it
1. Open the notebook in Colab (button above, or upload the file).
2. Choose a runtime (**Runtime → Change runtime type**); a GPU is needed only to train new runs.
3. **Runtime → Run all.**

The notebook reuses finished runs from this repository's `results` branch, trains anything missing when a GPU is available, and includes DocILE when it finds a token: in the `DOCILE_TOKEN` setting, in Colab's **Secrets** panel as `DOCILE_TOKEN`, or in a file `docile_token.txt`. Datasets come from their public sources:

| Dataset | Source |
|---|---|
| CORD | Hugging Face (`naver-clova-ix/cord-v2`) |
| WildReceipt | OpenMMLab |
| DocILE (optional) | official toolkit, free token from docile.rossum.ai |

| Branch | Contents |
|---|---|
| `main` | the notebook |
| `results` | finished runs, logs and reports |
| `source` | the same code as a Python package, with tests and the script that builds the notebook |

Code: MIT licence. Datasets keep their own licences.
