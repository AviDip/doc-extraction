# Making Document Extraction Worth Automating

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AviDip/doc-extraction/blob/main/project.ipynb)

Avishek Paul Dipto (a1988440) · Programming for AI and ML 2 · University of Adelaide

Document extraction models read invoices and receipts with over 90% of fields correct, yet most documents are still checked by hand, because nobody knows which outputs are wrong. This project trains models to make fewer mistakes on unfamiliar layouts, and tests which risk scores best decide which documents can skip a person.

**Everything is in [`project.ipynb`](project.ipynb)**: the full code with explanations (Part A) and the experiments (Part B).

## Running it
1. Open the notebook in Colab (button above, or upload the file).
2. **Runtime → Change runtime type → A100.**
3. Optional: add your DocILE token (free, docile.rossum.ai) in Colab's **Secrets** panel as `DOCILE_TOKEN`, or in the `DOCILE_TOKEN` setting in section 15. Without it, DocILE is skipped.
4. **Runtime → Run all.** The full run takes about six hours.

Datasets are downloaded from their public sources: CORD (Hugging Face), WildReceipt (OpenMMLab), DocILE (official toolkit).

The same code as a Python package, with tests and the script that builds this notebook, is on the `source` branch.

Code: MIT licence. Datasets keep their own licences.
