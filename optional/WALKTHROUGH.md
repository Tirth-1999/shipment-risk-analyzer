# Walkthrough: live presentation script

Follow **steps 0-12** below in order. That is the full walkthrough.

If they ask **why** or want deeper answers, open [`QA.md`](QA.md) separately. Do not mix the two during the first pass.

---

## Steps 0-12

Follow these steps in order. Keep `README.md` open and match each step to a README heading.

**Three phases:** (1) Setup → (2) Walk required files → (3) Demo → (4) Follow-up prep

---

### Step 0: Setup (run once before you present)

```bash
cd "/Users/tirthcshah/Desktop/Tirth Shah/Jobs/FullTime/NextEra Energy/shipment-risk-analyzer"
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[all]'
python tools/generate_dataset.py
python optional/demo_helpers.py --generate-walkthrough --seed 4242
python <<'PY'
import sys; sys.path.insert(0, "src")
from pathlib import Path
from tests.test_solution import train_full_model
train_full_model(Path("artifact"), Path("data"))
PY
pytest -v
```

You should see 33 tests pass, a green **ALL TESTS PASSED** summary, and `artifact/metrics.json` on disk.

---

### Step 1: Scenario (README top)

**Open:** `README.md` (Scenario section)

**Say:**
- Refrigerated trucks send temperature and door events.
- After each event, we predict: incident in the next 6 hours?
- Events are messy (late, duplicate, corrected). Labels arrive later.
- I built train offline + score online, with replay-safe behavior.

**Run:** nothing

---

### Step 2: Input records

**Open:** `src/dispatch_risk/contracts.py`

**Say:**
- `TelemetryEvent` = one sensor message.
- Two times matter: `device_time` (sensor clock) and `received_at` (when we got it).
- We score using `received_at`, not device clock alone.
- Label = 1 if incident happens in the next 6 hours after decision time.

**Run:** nothing

---

### Step 3: Training set (README Required #1)

**Open:** `src/dispatch_risk/solution.py` → search `def build_training_rows`

**Say:**
- Builds one training row per `(shipment_id, decision_time)`.
- Features use only events with `received_at <= decision_time`.
- Label 1 if incident in next 6h; skip row if we don't know the outcome yet.
- Late corrections only count if they had already arrived by decision time.
- Seven features: temp stats, warming rate, reading count, door opens (same as live scoring).

**Run:**

```bash
pytest tests/test_solution.py::TestTrainingLabels -v
```

---

### Step 4: Model training (README Required #2)

**Open:** `solution.py` → `def train` and `artifact/metrics.json`

**Say:**
- Trains logistic regression and gradient boosting; picks the better one on holdout.
- Saves `artifact/model.joblib` (loadable in a fresh process).
- Test split: 80% earlier shipments train, 20% later shipments test (whole trucks, not random rows).

Point at `metrics.json`:
- `evaluation_split`: how we split
- `pr_auc`: ranking quality (incidents are rare)
- `brier_score`: probability calibration
- `baseline_pr_auc`: vs dumb constant baseline
- `slices`: `early_window` and `late_window`

**Run:**

```bash
python -m json.tool artifact/metrics.json
```

---

### Step 5: Online engine (README Required #3)

**Open:** `solution.py` → `class RiskEngine`

**Say:**
- `ingest(event)`: accept stream; ignore exact duplicates; error on conflicts.
- `score(shipment_id, as_of)`: risk 0-1 using only data known at `as_of`.
- `snapshot` / `restore`: save and reload engine memory after crash.
- `reload_model`: swap model file; keep old model if load fails.
- Memory cap: drop oldest shipments when full.
- Same event history always gives the same output bytes.

**Run:**

```bash
pytest tests/test_solution.py::TestIngest tests/test_solution.py::TestScoring tests/test_solution.py::TestReplay -v
```

---

### Step 6: Tests (README Required #4)

**Open:** `tests/test_solution.py` (read the module docstring at the top)

**How pytest runs:**
- Finds every `Test*` class and `test_*` function in that file.
- Each test builds a small scenario, calls your code, checks the result (`assert`).
- `-v` prints each test name so you can say what just passed.

**Say while `pytest -v` runs:**
- "These are not random unit tests. Each group maps to a README requirement."
- "Green means that production failure mode is covered."

| Group | Count | What it proves |
|-------|-------|----------------|
| `TestContract` | 1 | Prediction JSON bytes are stable |
| `TestTrainingLabels` | 6 | README #1: no future data, label window, skip unknown rows |
| `TestIngest` | 5 | README #3: duplicates, conflicts, revisions, wire order |
| `TestScoring` | 6 | README #3: score time, late fixes, degraded mode |
| `TestReplay` | 4 | README #3: same history → same output bytes |
| `TestSnapshotRestore` | 2 | README #3: crash recovery |
| `TestModelReload` | 2 | README #3: safe deploy (keep old model if load fails) |
| `TestMemoryEviction` | 3 | README #3: memory cap |
| `TestOps` | 2 | Thread safety + metrics.json completeness |

**Run all 31:**

```bash
pytest -v
```

At the end you should see:

```
ALL TESTS PASSED (31 checks)
What this run confirmed:
  - Training rows respect received_at and label rules (README #1)
  ...
Nothing failed. Safe to demo and submit.
```

**Run one group while explaining that area:**

```bash
pytest tests/test_solution.py::TestTrainingLabels -v   # Step 3 training rules
pytest tests/test_solution.py::TestIngest -v           # Step 5 ingest
pytest tests/test_solution.py::TestScoring -v          # Step 5 scoring
pytest tests/test_solution.py::TestReplay -v           # replay determinism
```

Each test has a one-line docstring in the file if they ask what a specific name means.

---

### Step 7: Decisions (README Required #5)

**Open:** `DECISIONS.md`

**Say:**
- Documents every important choice and what I skipped in the timebox.
- Explains why I rejected bad advice from the customer notes.

**Run:** nothing

---

### Step 8: Customer notes (README list of 10)

**Open:** `DECISIONS.md` (customer notes section). Full answers: [`QA.md` → Customer notes](QA.md#f7-customer-notes-all-10)

**Say (short version: all 10 are rejected or reinterpreted):**
1. No random row split → whole future shipments
2. No "always newest revision" → only if received by score time
3. No sort entire stream by device time → ingest in delivery order
4. No dedupe on shipment_id alone → track event id + revision
5. Still need snapshots even if Kafka is "exactly once"
6. Don't return 0.0 on bad model load → keep last good model
7. Never put incidents in features → labels only
8. High AUC alone isn't enough → check baseline, slices, calibration
9. Can't keep every shipment forever → LRU memory cap
10. Reload must not wipe event memory

**Run:** nothing

---

### Step 9: Constraints

**Open:** `pyproject.toml` (quick glance)

**Say:**
- No network, no external APIs: sklearn + joblib only.
- Thread-safe (one lock). Works with `max_shipments=32` and 10k+ events.
- No hard-coded sample IDs or row counts.

**Run:** nothing

---

### Step 10: Demo

**Open:** terminal only (optional folder)

**Say:**
- Five demo shipments were never in training.
- Best one to show: `s-demo-correction`: score at hour 8 stays the same after a late fix.

**Run:**

```bash
cd optional && streamlit run app.py
```

In sidebar: **Train model** → **Generate walkthrough** → pick `s-demo-correction`.

Demo shipment order: correction → delayed → warming → door → stable. More detail: [Demo reference](#demo-reference) below.

---

### Step 11: Follow-up interview prep

**Open:** `README.md` (Follow-up section). Prep answers: [`QA.md` → Follow-up tasks](QA.md#f9-follow-up-interview-5-tasks)

**Say: five tasks they may ask:**
1. Score a new event stream (replay must match exactly)
2. Debug one broken invariant (write a failing test, fix, pytest)
3. Change a rule (e.g. 6h → 4h window)
4. Defend evaluation (shipment split, metrics, slices)
5. Small code change without breaking replay

**Run (practice):**

```bash
jupyter notebook optional/notebooks/interview_drills.ipynb
```

Hard questions: [`QA.md`](QA.md)

---

### Step 12: Close

**Say:**

> I hit every README requirement: point-in-time training, full metrics report, online engine with replay and safe reload, tests, and decisions. The model is simple on purpose. The hard part is doing time and streaming correctly.

---

### One-page checklist

| Step | Open | Run |
|------|------|-----|
| 0 Setup | - | block above |
| 1 Scenario | README | - |
| 2 Input | contracts.py | - |
| 3 Training | solution.py → build_training_rows | TestTrainingLabels |
| 4 Model | solution.py → train, metrics.json | json.tool metrics |
| 5 Engine | solution.py → RiskEngine | TestIngest/Scoring/Replay |
| 6 Tests | test_solution.py | pytest -v |
| 7 Decisions | DECISIONS.md | - |
| 8 Customer notes | DECISIONS.md | - |
| 9 Constraints | pyproject.toml | - |
| 10 Demo | Streamlit | streamlit run app.py |
| 11 Follow-up | interview_drills.ipynb | - |
| 12 Close | - | - |

---

## Demo reference

### Streamlit (`cd optional && streamlit run app.py`)

Sidebar: **Train model** → **Generate walkthrough** → pick demo shipment.

| Tab | Show |
|-----|------|
| Training metrics | PR-AUC, Brier, baseline, logreg vs HGB |
| Inference | Scores + reasons at each decision time |
| Event timeline | Temp chart, delays, doors, duplicates |
| Late correction | `s-demo-correction` only: score unchanged at hour 8 |

### Demo notebook (`optional/notebooks/demo_shipment_walkthrough.ipynb`)

**Phase A:** generate data → `build_training_rows` → `train` → metrics  
**Phase B:** five scripted shipments (never trained on):

| Shipment | Story | Point at |
|----------|-------|----------|
| `s-demo-stable` | Steady 3-4°C, duplicates | Idempotent ingest; low risk |
| `s-demo-warming` | Gradual warming | Risk rises before incident |
| `s-demo-correction` | Correction at hour 22 | Score at hour 8 **unchanged** |
| `s-demo-door` | Door opens + warming | `door_open_seen` |
| `s-demo-delayed` | Late `received_at` | Knowledge cutoff |

Decision times: 08:00, 14:00, 20:00 UTC each.

### Interview drills (`optional/notebooks/interview_drills.ipynb`)

| Drill | Practice |
|-------|----------|
| 1 | New event stream + deterministic replay |
| 2 | Late correction invariant |
| 3 | Horizon / split change |
| 4 | Defend evaluation |
| 5 | Code change + determinism (`dense_readings`) |
| 6 | Concurrency + customer notes table |
| 7 | Checklist: pytest, artifact, data, DECISIONS |

---

## Checklist

- [ ] Run Step 0 setup
- [ ] `pytest -v` → 31 passed
- [ ] `artifact/metrics.json` has first-decision-time split + `slices`
- [ ] Can run steps 0-12 end to end
- [ ] Can answer hard questions from `QA.md`
- [ ] Can answer all 10 customer notes (`QA.md`)
- [ ] Can walk five demo shipments (Step 10)
- [ ] Can navigate `solution.py` and read docstrings live
- [ ] `DECISIONS.md` matches your answers

**Files to have open:** `README.md`, `contracts.py`, `solution.py`, `DECISIONS.md`, `artifact/metrics.json`, this file. Keep `QA.md` nearby but closed until Q&A.

---

## Appendix: Flow diagram

```
TRAINING                          SERVING
========                          =======
events.jsonl ──┐                  artifact/model.joblib
labels.jsonl ─┼→ build_training_rows()     ↓
decision_times ┘       ↓              RiskEngine
                  train()                  ↓
                       ↓            ingest (delivery order)
              artifact/                      ↓
                                    score → to_wire()
```
