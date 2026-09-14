# Interview Q&A and reference

Use this when they ask **why** or push on details. For the live presentation script, use [`WALKTHROUGH.md`](WALKTHROUGH.md) only.

---

## Quick reference

### 30-second opening

> I built a system that estimates how likely a refrigerated shipment is to have a temperature problem in the next 6 hours. The hard part was doing it honestly: only use information the company had already received, handle late fixes and duplicate messages safely, give the same answer if you replay the same history, cap memory, and keep serving the old model if a new one fails to load.

**Summary:** Same history → same answer. Only use what we knew at the time. Safe deploys.

### Core rules

1. **When did we know?** Use arrival time (`received_at`), not the sensor clock alone.
2. **Late fixes** improve future scores; they don’t rewrite old scores.
3. **Test on future trucks**, hold out later whole shipments, not random moments from the same route.
4. **Unknown endings**, skip training rows where we don’t know yet if an incident happened; don’t guess “no.”
5. **Duplicate messages**, same message twice: ignore. Same ID but different content: error.
6. **Memory cap**, drop oldest shipments when full; don’t let “already seen” lists grow forever.
7. **Bad model load**, keep the last working model; don’t return “zero risk” as a fake OK.

### Key phrases

- “I score using when the reading *arrived*, not when the sensor *claims* it happened.”
- “A correction that arrives tomorrow doesn’t change yesterday’s report.”
- “I test on shipments from later in time, like next week’s routes, not random boxes from the same truck.”
- “If we don’t know the outcome yet, I skip that example instead of teaching the model the wrong answer.”
- “Same event delivered twice doesn’t get counted twice.”
- “If the new model file fails to load, we keep the old one, we don’t pretend everything is safe.”
- “I kept the code small so I can explain and change it live with you.”

### Summary

> I built a small engine you can test and trust under messy real-world delivery: late data, fixes, duplicates, memory limits, and deploys. The math model is simple on purpose. The important work is the time rules and replay behavior.

---

## Code reference

### File map

| Path | Role |
|------|------|
| `README.md` | Assignment spec, API contract, customer notes |
| `DECISIONS.md` | Official design defense (what you chose/rejected) |
| `src/dispatch_risk/contracts.py` | `TelemetryEvent`, `TrainingRow`, `Prediction` + docstrings |
| `src/dispatch_risk/solution.py` | **Main impl**: `build_training_rows`, `train`, `RiskEngine` |
| `tools/generate_dataset.py` | Training data generator (**do not modify**) |
| `tests/test_solution.py` | All tests, fixtures, and helpers in one file (33 tests, 9 classes) |
| **`optional/`** | **Not graded**. see `optional/README.md` |
| `optional/WALKTHROUGH.md` | Live presentation script (steps 0-12) |
| `optional/QA.md` | This file: Q&A and technical reference |
| `optional/app.py` | Streamlit demo |
| `optional/demo_helpers.py` | Demo/notebook/interview helpers |
| `optional/notebooks/` | Demo walkthrough + interview drills |

**Generated (gitignored, create on first run):** `data/*.jsonl`, `artifact/`, `optional/data/`

### Public API

```python
build_training_rows(events, labels, decision_times)
train(rows, artifact_dir)
RiskEngine(artifact_dir, max_shipments)
```

> `build_training_rows` → point-in-time examples. `train` → fit + save artifact. `RiskEngine` → ingest, score, snapshot, restore, reload.

### The problem (one paragraph)

Operations wants P(incident in next 6h) after every telemetry event. Events arrive at-least-once, late, out-of-order, and corrected. Labels arrive later. The engine must be point-in-time correct, deterministic under replay, thread-safe, memory-bounded, and reload-safe.

| Phase | Question | Updates model? |
|-------|----------|----------------|
| Train | What patterns predict incidents historically? | Yes → `artifact/` |
| Score | What is risk right now? | No → load artifact, ingest, predict |

### Data model

**Two clocks:**

| Field | Meaning | Used for |
|-------|---------|----------|
| `device_time` | When device claims reading happened | Order readings inside known set |
| `received_at` | When platform could use this revision | **Leakage boundary** |

**Label window:** positive if `incident_at ∈ (decision_time, decision_time + 6h]` (left open, right closed).

### Code walkthrough (`solution.py`)

Read top-to-bottom. Every public function has a docstring. Use them live in the interview.

**Training path:**
```
events + labels + decision_times
  → build_training_rows()   # censoring + received_at features + labels
  → _split_by_shipment()    # 80% earlier / 20% later shipments
  → train()                 # logreg vs HGB, slices, save artifact
```

**Serving path:**
```
artifact/model.joblib → RiskEngine
  → ingest(event) × N       # delivery order, idempotent
  → score(shipment_id, as_of) → Prediction.to_wire()  # deterministic bytes
```

**Key functions:**

| Function | Role |
|----------|------|
| `_events_known_at` | `received_at <= as_of`; max revision; sort by `device_time` |
| `_build_features` | 7 features: **same path** in train and score |
| `_incident_label` | Binary label with censoring (`None` = skip row) |
| `_split_by_shipment` | Hold out later shipments by first decision time |
| `RiskEngine.ingest` | Idempotent; conflicting duplicate → `ValueError` |
| `RiskEngine.score` | Point-in-time prediction; degraded if no events |
| `RiskEngine.reload_model` | Swap on success; keep old model on failure |

**Internal state:** `_shipments` (LRU), `_seen_deliveries`, `_delivery_fingerprints`, `_lock` (RLock). On eviction, delivery keys are removed too.

---

## ML pipeline

### Features (7)

| Feature | Meaning |
|---------|---------|
| `latest_temp_c` | Most recent temperature |
| `mean_temp_c` / `max_temp_c` | Aggregate level / peak |
| `temp_slope_c_per_h` | Warming rate |
| `temp_reading_count` | Evidence volume |
| `hours_since_first_reading` | Time span |
| `door_open_count` | Operational risk |

**Reasons (explainability, not inputs):** `temperature_high`, `warming_trend`, `door_open_seen`, `model_high_risk`, `no_events_seen`

### Models

- **Candidates:** LogisticRegression (baseline) vs HistGradientBoostingClassifier
- **Selection:** higher holdout PR-AUC → tie-break lower Brier
- **Fallback:** single-class training → DummyClassifier(prior)
- **Artifact:** `model.joblib` + `metrics.json`

### Evaluation

**Split:** 80/20 by shipment **first decision time** (later shipments held out).

**Metrics:** PR-AUC (rare positives), Brier (calibration), constant baseline, logreg vs HGB comparison.

**Slices (holdout):**

| Slice | Filter |
|-------|--------|
| `early_window` | `hours_since_first_reading < 10` |
| `late_window` | `hours_since_first_reading >= 10` |

---

## Online engine behavior

| Operation | Behavior |
|-----------|----------|
| **Ingest**: exact duplicate | Ignore, return `False` |
| **Ingest**: conflicting duplicate | `ValueError` |
| **Ingest**: higher revision | Store if not stale |
| **Score**: no known events | `degraded=True`, prob 0.0 |
| **Score**: reload failed | Keep previous model (not degraded) |
| **Eviction** | LRU; drop shipment + its delivery keys |
| **Snapshot/restore** | Deterministic sorted JSON |
| **Concurrency** | One RLock on all mutating ops |

---
## Q&A bank

Each entry has a short answer, optional detail, and references to code or tests where relevant.

---

### F0: Big picture

**Q: What did you build?**  
**A:** “I built a system that watches refrigerated truck shipments and answers one question: *Will this shipment have a temperature problem in the next 6 hours?* It gives a number from 0 to 1, like a weather forecast for spoilage. I split the work into two parts: first, learn from past shipments; second, score live shipments as events arrive, without cheating by using information from the future.”  
**Details:** Training writes a saved model file. The live engine loads that file, remembers events per shipment, and outputs a score plus human-readable reasons (like ‘temperature high’ or ‘door was open’).  
**Reference:** `build_training_rows`, `train`, `RiskEngine` in `solution.py`.

**Q: What problem were you really solving?**  
**A:** “Events arrive messy, late, out of order, duplicated, sometimes corrected later. Labels about incidents arrive even later. The hard part isn’t picking a fancy model. It’s making sure that when I score ‘as of 2pm,’ I only use what the company actually knew at 2pm, and that running the same history twice gives the exact same answer.”  
**Details:** That’s why most of my tests are about time, duplicates, corrections, memory limits, and reload, not just accuracy on fake data.

**Q: Walk me through train vs score in one minute.**  
**A:** “Train: read historical events and incident records, build examples at decision times, fit a small model, save it to disk. Score: load the model, ingest live events in delivery order, at each decision time compute features from events we already received, run the model, return probability plus reasons. Training updates the model; scoring does not.”  
**Reference:** Walkthrough appendix flow diagram; demo notebook Phase A vs Phase B.

---

### F1: Two clocks: when did it happen vs when did we know?

**Q: Why `received_at` and not `device_time`?**  
**A:** “The sensor clock says when the fridge *claims* it measured temperature. But the company can only act on a reading once it *arrives* at our system. I score using the arrival time, `received_at`, because that’s when the reading became real for operations. Device time still helps me order readings I already know about.”  
**Note:** “It’s like getting a letter: the postmark is not the same as the day you actually read it. For decisions, I care when we read it.”  
**Details:** Function `_events_known_at` throws away anything with `received_at` after the decision time.  
**Reference:** `test_training_respects_received_at_cutoff`.

**Q: Can you use future events when building features?**  
**A:** “No. If a reading arrives at 3pm, it cannot affect a score at 2pm, even if the sensor says the reading happened at 1pm.”  
**Details:** Same rule in training and in the live engine so they stay in sync.  
**Reference:** `_events_known_at` in `solution.py`; `test_training_and_scoring_use_same_features`.

**Q: Should we sort the incoming stream by device time before ingest?**  
**A:** “No. Messages can arrive in any order on the wire. I ingest in delivery order. When I need a timeline inside one shipment, I sort only the events I already know about, by device time.”  
**Note:** “You don’t reorder your mailbox before opening letters, but once opened, you might sort notes by date on your desk.”

**Q: What if telemetry arrives late?**  
**A:** “Late arrival is normal. Until `received_at` passes, that reading doesn’t exist for scoring. A score at 2pm won’t change when a 1pm reading finally shows up at 4pm, unless I score again at 4pm or later.”  
**Details:** Demo shipment `s-demo-delayed` shows this.

---

### F2: Corrections and duplicate messages

**Q: How do corrections work?**  
**A:** “Each event has an ID and a revision number, like ‘message 5, version 2.’ When I score, I use the newest version that had already arrived by that score time. If a fix arrives later, it improves *future* scores, not scores I already computed for earlier times.”  
**Note:** “You don’t rewrite yesterday’s report when today’s correction email arrives. You update today’s view going forward.”  
**Reference:** `test_late_correction_does_not_change_past_score`; demo `s-demo-correction`.

**Q: What about duplicate deliveries?**  
**A:** “The network may deliver the same message twice. If it’s truly the same event, same ID, same revision, same content, I ignore the second copy. If the same ID and revision show up with *different* content, that’s a data error and I raise an error instead of guessing.”  
**Note:** “Same email twice → delete duplicate. Same subject line but different body → something is wrong, stop and alert.”  
**Reference:** `test_duplicate_ingest_is_idempotent`, `test_conflicting_duplicate_delivery_raises`.

**Q: Show me the late-correction rule works.**  
**A:** “Ingest the original reading, score at hour 8, then ingest a correction that arrives much later, score again at hour 8, the answer bytes are identical. The correction only matters for score times after it arrived.”  
**Reference:** `TestScoring::test_late_correction_does_not_change_past_score`.

**Q: What if an old revision shows up after a newer one?**  
**A:** “I keep the highest revision I’ve seen. A stale lower revision that arrives late gets ignored, it doesn’t overwrite the newer truth.”  
**Reference:** `test_stale_lower_revision_is_ignored`.

---

### F3: Labels: what counts as an incident?

**Q: How is the label defined?**  
**A:** “At each decision time, label = 1 if an incident happens *sometime in the next 6 hours after that moment* (not including the exact decision second). Otherwise label = 0.”  
**Note:** “At noon I ask: will anything go wrong between noon and 6pm? Not ‘right at noon’, that’s the moment I’m standing in.”  
**Reference:** `test_label_window_boundaries`, `test_label_boundary_at_exactly_six_hours`.

**Q: What is label censoring? (skip rows with unknown endings)**  
**A:** “Sometimes we don’t know the answer yet. Maybe the 6-hour window hasn’t finished. Maybe an incident happened but the paperwork isn’t filed yet. I skip those training rows instead of guessing ‘no incident.’ Guessing ‘no’ would teach the model the wrong lesson.”  
**Note:** “You wouldn’t mark a student wrong on a test they haven’t finished yet.”  
**Reference:** `test_label_censoring_skips_unpublished_rows`.

**Q: Do you use incident records as inputs to the model?**  
**A:** “Never. Incidents are the answer key for training only. Using them as inputs would be cheating, like giving the model the test answers while it’s taking the test.”  
**Reference:** Customer note #7; `build_training_rows` never reads incidents into features.

**Q: Incident exactly at decision time, positive?**  
**A:** “No. the window starts *after* the decision moment. Incident exactly 6 hours later *does* count as positive.”

---

### F4: Features and model (keep it simple)

**Q: What features do you use?**  
**A:** “Seven numbers from temperature readings and door events: latest temp, average, max, warming speed, how many readings, how long we’ve been watching, and how many times the door opened. All computed only from events we already received.”  
**Details:** Same recipe in training and scoring, one function `_build_features`.

**Q: Why such simple features?**  
**A:** “I wanted something I can explain on a whiteboard and debug live. Fancy features don’t help if I can’t prove I didn’t use future information. Simple + shared between train and score was the goal.”

**Q: Why two model types (logistic regression vs gradient boosting)?**  
**A:** “I train both on the same data and pick the winner on held-out shipments. Logistic regression is the simple straight-line baseline. Gradient boosting can catch curved patterns, like ‘dangerous only when temp is high *and* door opened.’ I pick whichever ranks incidents better on the test set; if tied, whichever’s probability numbers are more honest.”  
**Details:** Saved in `metrics.json` as `model_comparison`.

**Q: Why care about ‘ranking incidents’ more than overall accuracy?**  
**A:** “Incidents are rare, most shipments are fine. A model that always says ‘safe’ looks accurate but useless. I measure how well it puts bad shipments above good ones on the test set, and I compare against a dumb baseline that always predicts the average rate.”  
**Details:** That’s PR-AUC in the metrics file, “precision-recall area,” but you can just say “ranking quality when positives are rare.”

**Q: What is the Brier score?**  
**A:** “It checks whether predicted probabilities match reality, like saying ‘70% chance’ should mean about 7 out of 10 similar cases happen. Lower is better.”

---

### F5: Evaluation: how do you know the model works?

**Q: How do you split train vs test?**  
**A:** “I sort shipments by when we first scored them. The earliest 80% of shipments go to training; the latest 20% go to testing. Whole shipments stay on one side, I never split rows from the same truck across both sets.”  
**Note:** “Train on last month’s routes; test on next week’s routes, not random boxes from the same truck in both piles.”  
**Reference:** `_split_by_shipment`; metrics say “80/20 by first decision time.”

**Q: Why not random 80/20 on rows?**  
**A:** “Random rows from the same shipment leak information. The model would see early hours of a route in training and later hours in testing, that’s not how deployment works. In production we score whole new shipments we haven’t seen.”

**Q: What are slices?**  
**A:** “Overall score can hide problems in subgroups. I also check shipments where the door opened, and shipments with no temperature readings yet. If the model fails there, ops needs to know even if the average looks fine.”  
**Reference:** `slices` in `metrics.json`, `early_window`, `late_window`.

**Q: Is 0.98 score enough to launch?**  
**A:** “No. This is synthetic data. I’d need real-world validation, check against a simple baseline, look at subgroups, tune the threshold based on cost of false alarms vs missed incidents, and monitor after launch.”

**Q: Your metrics file looks out of date?**  
**A:** “Fair. I regenerate the model artifact before demos so the metrics file matches the current code. Run Walkthrough Step 0 setup.”

---

### F6: Live engine behavior

**Q: What happens if we score a shipment with no events?**  
**A:** “I return probability 0 with a flag saying ‘degraded, no events seen.’ That means ‘I have no evidence,’ not ‘definitely safe.’ It’s different from a broken model.”  
**Reference:** `test_score_without_events_is_degraded`.

**Q: What if loading a new model fails during deploy?**  
**A:** “Keep serving the old model. Don’t pretend the risk is zero, that would lie to operations. Report that reload failed; previous model still runs.”  
**Reference:** `test_reload_failure_keeps_old_model`.

**Q: What if reload succeeds?**  
**A:** “Swap to the new model file. Shipment memory stays, reload changes the brain, not the notebook of events.”  
**Reference:** `test_reload_success_swaps_model`, `test_restored_snapshot_scores_after_model_reload`.

**Q: How does the memory limit (`max_shipments`) work?**  
**A:** “I can’t remember every shipment forever. I keep the most recently used ones up to a cap. When full, I drop the oldest shipment’s memory to make room, like clearing the oldest folder off a desk.”  
**Details:** I also drop that shipment’s ‘already seen message’ IDs so duplicate-tracking doesn’t grow forever.  
**Reference:** `test_max_shipments_eviction`, `test_max_shipments_caps_active_state_under_load`.

**Q: What if a dropped shipment comes back?**  
**A:** “It starts with a clean slate, we lost history when evicted. That’s the tradeoff for bounded memory. In production I’d maybe persist hot shipments or raise the cap.”

**Q: Why snapshots if the message queue is ‘exactly once’?**  
**A:** “Even perfect delivery doesn’t replace app memory. Snapshots save ‘what the engine knows’ so after a crash or restart I can resume without reprocessing everything from scratch, and duplicates still won’t double-count.”  
**Reference:** `test_snapshot_bytes_are_deterministic`, `test_restore_preserves_idempotency`.

**Q: Multiple threads hitting the engine at once?**  
**A:** “One lock wraps ingest, score, snapshot, restore, and reload so they don’t step on each other. Simple and safe; not the fastest possible design.”  
**Reference:** `test_concurrent_ingest_and_score`.

**Q: What is replay determinism?**  
**A:** “Feed the same events in the same order twice from a clean start, you get byte-identical outputs and identical snapshot files. No randomness, no hidden state. That makes bugs reproducible and audits possible.”  
**Reference:** `test_replay_is_deterministic`, `test_large_replay_over_ten_thousand_events`.

**Q: What are ‘reasons’ on a prediction?**  
**A:** “Human-readable tags explaining why the score is high or low, like ‘temperature high,’ ‘warming trend,’ ‘door open seen.’ They don’t change the math; they help a human trust or challenge the number.”

---

### F7: Customer notes (all 10)

| # | Customer says | Response |
|---|---------------|------------------|
| 1 | Random 80/20 row split | “No. Test on whole future shipments, not random moments from the same truck.” |
| 2 | Always use newest revision | “No. Only revisions that had already arrived by score time.” |
| 3 | Sort everything by device time | “Only inside what we already know; don’t reorder the incoming wire.” |
| 4 | Deduplicate on shipment_id | “No. We track events per shipment, not one blob per truck.” |
| 5 | Kafka is exactly-once, skip snapshots | “No. The app still needs memory rules and save/restore.” |
| 6 | Return 0.0 if model can’t load | “No. Keep last good model; zero would fake ‘safe.’” |
| 7 | Use incident table in features | “No. That’s the answer key, not an input.” |
| 8 | AUC > 0.90 means ship it | “No. Check baselines, calibration, subgroups, real data, business costs.” |
| 9 | Keep every shipment in memory forever | “No. Cap with LRU; drop oldest when full.” |
| 10 | Reload may wipe memory | “No. Reload swaps model file only; event memory stays.” |

Full written defense: `DECISIONS.md`.

---

### F8: Design choices, weaknesses, production next steps

**Q: Why keep the code small?**  
**A:** “So I can explain every rule in an interview and change one thing without breaking five others. The value here is correct time behavior under messy streams, not a huge framework.”

**Q: What would you add in production?**  
**A:** “Stronger validation of saved files, more subgroups in evaluation, persist important shipments before eviction, clearer ‘why degraded’ messages, automated retraining, and picking alert thresholds based on dollar cost, not just model score.”

**Q: Weaknesses you’re aware of?**  
**A:** “Simple temperature features; evicted shipments lose history; one lock limits speed; trained on fake data; saved model format is Python-specific. I chose these to fit the timebox and documented the next steps.”

**Q: What makes this submission defensible?**  
**A:** “Every scary streaming bug I could think of has a test: late data, duplicates, corrections, memory cap, reload failure, threading, big replays. The model is boring on purpose; the engineering rules are not.”

---

### F9: Follow-up interview (5 tasks)

**Task 1: New event stream**  
**A:** “Load the saved model, don’t retrain unless asked. Feed new events in arrival order, don’t resort by sensor clock. Score at the given times. Run the whole thing twice; outputs must match exactly.”

**Task 2: Something broke**  
**A:** “Reproduce it, figure out which category, time, duplicate, snapshot, reload, memory, write the smallest test that fails, fix the smallest code path, run pytest.”

**Task 3: Change a rule**  
**A:** “Example: prediction window 6 hours → 4 hours. Change the constant, update labels, retrain if needed, add a test for the new boundary, run pytest.”

**Task 4: Defend your evaluation**  
**A:** “Hold out later whole shipments; skip rows where we don’t know the outcome yet; report ranking quality and probability honesty vs a dumb baseline; check door-open and no-reading subgroups; admit synthetic data isn’t enough to launch.”

**Task 5: Small code change without breaking replay**  
**A:** “Safe example: add a new reason tag when we have many temperature readings, without changing probabilities or the feature fingerprint. Replay must still match byte-for-byte.”  
**Reference:** `INTERVIEW_DENSE_READINGS_PATCH` in `optional/demo_helpers.py`.

---

### F10: Quick reference

**Q: What does the probability number mean?**  
**A:** “It’s the model’s best guess of how likely a temperature incident is in the next 6 hours, based on what we know right now, not a guarantee.”

**Q: What’s the hardest bug you prevented?**  
**A:** “Using tomorrow’s information to answer yesterday’s question. That makes offline tests look amazing and production fail.”

**Q: What would you demo in 2 minutes?**  
**A:** “Shipment `s-demo-correction`: score at hour 8, then a fix arrives later, hour 8 score unchanged. Shows we respect ‘what was known when.’”

**Q: One sentence on why you’d hire your approach?**  
**A:** “I optimized for rules operations can trust, same input history, same output, honest about missing data, safe deploys, not for the fanciest model name.”

---

## Tests reference

See the module docstring at the top of `tests/test_solution.py` for how pytest runs and what each group maps to.

**33 tests** in 9 groups. Run: `pytest -v`

| Group | Tests | README | Proves |
|-------|-------|--------|--------|
| `TestContract` | 1 | API | Stable prediction JSON bytes |
| `TestTrainingLabels` | 6 | Required #1 | received_at cutoff, labels, train/score feature parity |
| `TestIngest` | 5 | Required #3 | Idempotent ingest, conflicts, revisions |
| `TestScoring` | 6 | Required #3 | Late corrections, degraded, UTC normalization |
| `TestReplay` | 4 | Required #3 | Deterministic replay, 10k+ events |
| `TestSnapshotRestore` | 2 | Required #3 | Restore keeps dedupe state |
| `TestModelReload` | 2 | Required #3 | Failed reload keeps old model |
| `TestMemoryEviction` | 3 | Required #3 | LRU cap under load |
| `TestOps` | 2 | Constraints | Thread safety, metrics.json fields |

Every `test_*` function has a one-line docstring explaining that specific check.

