# Shipment Risk Analyser: optional demo & interview prep (not graded)

Everything here is for demos, practice, and interview prep. The graded submission is `src/`, `tests/`, `tools/generate_dataset.py`, and `DECISIONS.md`.

## Contents

| Path | Purpose |
|------|---------|
| [`WALKTHROUGH.md`](WALKTHROUGH.md) | **Presentation script**: steps 0-12 (setup → README → demo) |
| [`QA.md`](QA.md) | **Q&A and reference**: open when they ask "why?" |
| `app.py` | Streamlit demo |
| `demo_helpers.py` | Shared helpers for app and notebooks |
| `notebooks/` | Demo walkthrough + interview drills |
| `data/` | Generated demo streams |

## Quick start

1. Run setup in [`WALKTHROUGH.md` Step 0](WALKTHROUGH.md#step-0-setup-run-once-before-you-present).
2. Present using steps 1-12 in `WALKTHROUGH.md`.
3. Use `QA.md` only for follow-up questions.

```bash
cd optional && streamlit run app.py
jupyter notebook optional/notebooks/demo_shipment_walkthrough.ipynb
jupyter notebook optional/notebooks/interview_drills.ipynb
```

Core training data stays in `../data/`. Demo shipment data is written to `optional/data/walkthrough/`.
