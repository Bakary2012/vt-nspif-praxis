# VT NSPIF Praxis Code

Source code and execution notebooks for the Doctor of Engineering praxis
*A Vision Transformer and Neuro-Symbolic Framework for Recognizing and Validating
Handwritten Medical Prescription Text toward Structured Electronic Health Record (EHR) Data*,
Bakary Tagognon Sanogo, The George Washington University, 2026.

These files were previously reproduced as listings in the praxis appendices.

## Contents

| Path | Praxis appendix | Description |
|---|---|---|
| `src/vt_nspif_core.py` | Appendix F | Research workflow core: extraction, serialization, scoring |
| `tests/test_vt_nspif_core.py` | Appendix G | 40 deterministic software checks |
| `src/source_label_benchmark.py` | Appendix H | RxHandBD input adapter for the source label benchmark |
| `notebooks/appendix_k_trocr_finetuning.ipynb` | Appendix K.5 | TrOCR fine-tuning protocol (Google Colab) |
| `src/word_level_ablation.py` | Appendix L | Word-level validation ablation |
| `notebooks/appendix_m_enhanced_recognizer_lgc.ipynb` | Appendix M.4 | Enhanced recognizer and lexicon-guided validation (Google Colab) |
| `src/phee_ablation.py`, `src/entity_ablation.py` | Appendix N.3 | Entity-level validation ablation |

## Running the software checks

```bash
PYTHONPATH=src python -m unittest tests/test_vt_nspif_core.py
```

## Data

No datasets are included. RxHandBD, PHEE, and the n2c2 2018 Track 2 corpus must be
obtained from their publishers under their own terms of use.

## Execution environment

Recognition runs used PyTorch 2.8.0 and Transformers 4.57.1 with the TrOCR checkpoint
`microsoft/trocr-base-handwritten` at revision `eaacaf452b06415df8f10bb6fad3a4c11e609406`.
The notebooks were executed on Google Colab GPUs (NVIDIA L4 and A100).
