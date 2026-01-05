# CARE Pipeline

This repository contains a two-stage pipeline for explanation-aware clinical trait classification using large language models.

- `CARE.py` – training + validation + test evaluation
- `inference.py` – inference on unseen data using a trained checkpoint

---

## Setup

Create an environment and install dependencies:

```bash
pip install torch transformers sentence-transformers peft scikit-learn pandas numpy tqdm matplotlib seaborn
