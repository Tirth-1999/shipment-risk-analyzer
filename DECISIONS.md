# Decision record

## Event-time and knowledge-time policy

- I use `received_at` as the knowledge cutoff. An event is only available at `as_of` when `received_at <= as_of`.
- I store every revision. At score time I pick the highest revision with `received_at <= as_of`.
- If a correction arrives after a decision, it can change future scores but not past ones when I re-score the same `as_of`.
- I use `device_time` only to order readings inside the known set. It does not override the knowledge cutoff.

## Label construction and leakage controls

- Label is `1` when `incident_at` falls in `(decision_time, decision_time + 6 hours]`, else `0`.
- **Label censoring (simple rule):** skip a row unless (a) the 6-hour window has finished and (b) any incident label in that window has `label_available_at <= observation_cutoff`. Default cutoff is the latest label publish time in the dataset.
- Features come only from telemetry known at `decision_time`.
- I never use incident tables or `label_available_at` in features.
- **Evaluation split:** 80/20 by shipment **first decision time** (later shipments held out), not random rows and not sorted shipment IDs.

## Features and model

- Features: latest/mean/max temperature, warming slope, reading count, hours since first reading, door-open count.
- Candidates: `LogisticRegression` (linear baseline) and `HistGradientBoostingClassifier` (non-linear tabular model).
- Selection: higher PR-AUC on the held-out shipment split; tie-break on lower Brier score.
- Why HGB: temperature risk is driven by non-linear combinations (slope + door opens + max temp). Trees handle that without extra dependencies.
- Metrics: PR-AUC, Brier score, constant baseline, side-by-side candidate comparison, plus simple **slices** on the holdout set (`early_window`, `late_window`).
- Slice note: sample door-open events arrive after incidents, so a `door_open_seen` holdout slice is empty on this generator. I slice by trip age instead.
- PR-AUC on an eval split with zero positives is defined as `0.0`.
- Artifact: `model.joblib` stores the winning pipeline, feature names, model version (e.g. `logreg-v1`), and metrics.

## State, idempotency, and eviction

- Exact duplicate `(event_id, revision)` deliveries are ignored.
- **Conflicting** redelivery of the same `(event_id, revision)` with different content raises `ValueError`.
- I store every revision. At score time I pick the highest revision with `received_at <= as_of` (so a correction delivered before the original still leaves earlier `as_of` scores correct).
- When active shipments exceed `max_shipments`, I evict the least recently used shipment.
- Snapshots keep shipment LRU order so restore continues eviction correctly.
- On eviction I also drop that shipment's deliveries from `_seen_deliveries`, so memory stays bounded under `max_shipments`.
- If a correction arrives after eviction, that shipment starts fresh when it returns.

## Concurrency and model reload

- One re-entrant lock protects ingest, score, snapshot, restore, reload, and stats.
- On reload I validate the candidate artifact first and swap only on success.
- If reload fails I return `False` and keep serving the previous model.

## Customer notes I rejected or reinterpreted

| Note | My decision |
|------|-------------|
| 1 random 80/20 row split | Rejected. I hold out later whole shipments by first decision time. |
| 2 apply newest revision to old decisions | Rejected. I use the received_at policy above. |
| 3 sort by device time | Accepted for feature ordering only, after the received_at filter. |
| 4 deduplicate on shipment_id | Rejected. Each shipment keeps its own event state. |
| 5 Kafka exactly-once removes snapshot need | Rejected. I still need idempotent ingest and snapshots. |
| 6 return 0.0 when model cannot load | Rejected. I keep the previous model; degraded only when no events exist. |
| 7 use full incident table in features | Rejected. Incidents are labels only. |
| 8 AUC above 0.90 is enough | Rejected. I report PR-AUC, calibration, baseline, and slices. |
| 9 keep every shipment in memory | Rejected. I enforce `max_shipments` with LRU eviction. |
| 10 reload may clear in-memory state | Rejected. Reload swaps the model only. |

## Known limitations

- Simple hand-built features only.
- Evicted shipments lose prior in-memory history (by design).
- No automated retraining pipeline.

## Testing

- 33 pytest cases in `tests/test_solution.py` (grouped by concern).
- `pyproject.toml` sets `pythonpath = ["src"]` so tests import `dispatch_risk` without extra env setup.

## Reproduction

```bash
python tools/generate_dataset.py
pip install -e '.[dev]'
pytest -v
```

Optional demo (not graded): see `optional/README.md`. Streamlit, notebooks, and interview prep live under `optional/`.
