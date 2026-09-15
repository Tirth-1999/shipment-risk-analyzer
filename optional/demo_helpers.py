"""Demo and interview helpers for notebooks, Streamlit, and drills (not graded).

Provides: JSONL loaders, walkthrough shipment generation, holdout plots,
deterministic replay checks, interview drill utilities (see ``interview_drills.ipynb``).
Core scoring logic remains in ``src/dispatch_risk/solution.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import pandas as pd

    from dispatch_risk.contracts import TelemetryEvent, TrainingRow

OPTIONAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OPTIONAL_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
WALKTHROUGH_DIR = OPTIONAL_DIR / "data" / "walkthrough"
INTERVIEW_STREAM_DIR = OPTIONAL_DIR / "data" / "interview_stream"
ARTIFACT_DIR = PROJECT_ROOT / "artifact"

# One temperature reading per hour for a full 24h demo stream.
DEMO_STREAM_HOURS = 24


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Read one JSON Lines file into a list of dicts."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_event(raw: dict[str, object]) -> "TelemetryEvent":
    """Convert one JSON dict into a TelemetryEvent."""
    from dispatch_risk.contracts import TelemetryEvent

    return TelemetryEvent(
        event_id=str(raw["event_id"]),
        revision=int(raw["revision"]),
        shipment_id=str(raw["shipment_id"]),
        device_time=_parse_time(str(raw["device_time"])),
        received_at=_parse_time(str(raw["received_at"])),
        kind=str(raw["kind"]),
        value=raw["value"],  # type: ignore[assignment]
        source=str(raw["source"]),
        payload=dict(raw["payload"]),  # type: ignore[arg-type]
    )


def load_events(data_dir: Path = DATA_DIR) -> list["TelemetryEvent"]:
    """Load all events from a data folder's events.jsonl file."""
    return [parse_event(row) for row in load_jsonl(data_dir / "events.jsonl")]


def load_labels(data_dir: Path = DATA_DIR) -> list[dict[str, object]]:
    """Load incident labels from labels.jsonl."""
    return load_jsonl(data_dir / "labels.jsonl")


def load_decision_times(data_dir: Path = DATA_DIR) -> list[tuple[str, datetime]]:
    """Load score times from decision_times.jsonl."""
    rows = load_jsonl(data_dir / "decision_times.jsonl")
    return [(str(row["shipment_id"]), _parse_time(str(row["decision_time"]))) for row in rows]


def events_for_shipment(events: list["TelemetryEvent"], shipment_id: str) -> list["TelemetryEvent"]:
    """Keep events for one shipment, in delivery order."""
    return [event for event in events if event.shipment_id == shipment_id]


def decisions_for_shipment(
    decision_times: Sequence[tuple[str, datetime]], shipment_id: str
) -> list[tuple[str, datetime]]:
    """Keep decision times for one shipment."""
    return [(sid, as_of) for sid, as_of in decision_times if sid == shipment_id]


def incident_time_for_shipment(labels: list[dict[str, object]], shipment_id: str) -> datetime | None:
    """Return incident_at for a shipment, or None if there is no label."""
    for label in labels:
        if str(label["shipment_id"]) == shipment_id:
            return _parse_time(str(label["incident_at"]))
    return None


def load_walkthrough_manifest(data_dir: Path = WALKTHROUGH_DIR) -> dict[str, object]:
    """Load MANIFEST.json from a walkthrough folder."""
    path = data_dir / "MANIFEST.json"
    if not path.exists():
        return {"shipments": []}
    return json.loads(path.read_text(encoding="utf-8"))


def walkthrough_shipment_options(data_dir: Path = WALKTHROUGH_DIR) -> list[dict[str, object]]:
    """Return shipment entries from the walkthrough manifest."""
    manifest = load_walkthrough_manifest(data_dir)
    shipments = manifest.get("shipments")
    if isinstance(shipments, list):
        return shipments
    if "shipment_id" in manifest:
        return [dict(manifest)]  # type: ignore[list-item]
    return []


def label_in_next_6h(decision_time: datetime, incident_at: datetime | None) -> int:
    """Return 1 if incident_at falls in (decision_time, decision_time + 6 hours]."""
    if incident_at is None:
        return 0
    decision_time = decision_time.astimezone(timezone.utc)
    incident_at = incident_at.astimezone(timezone.utc)
    return int(decision_time < incident_at <= decision_time + timedelta(hours=6))


def holdout_evaluation(
    rows: Sequence["TrainingRow"],
    artifact_dir: Path,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Score the held-out shipment split and return arrays for charts."""
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        precision_recall_curve,
    )

    from dispatch_risk.solution import (
        _positive_class_probabilities,
        _rows_to_xy,
        _split_by_shipment,
    )

    train_rows, test_rows = _split_by_shipment(rows)
    if not test_rows:
        test_rows = list(train_rows)

    bundle = joblib.load(artifact_dir / "model.joblib")
    model = bundle["model"]
    x_test, y_test = _rows_to_xy(test_rows)
    probabilities = _positive_class_probabilities(model, x_test)
    predictions = [1 if prob >= threshold else 0 for prob in probabilities]

    precision, recall, pr_thresholds = precision_recall_curve(y_test, probabilities)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    return {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "y_true": y_test.tolist(),
        "y_prob": probabilities,
        "y_pred": predictions,
        "pr_curve": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def generate_fresh_stream(
    output_dir: Path,
    *,
    seed: int = 9999,
    shipments: int = 120,
) -> dict[str, object]:
    """Generate a new dataset with tools/generate_dataset.py."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "generate_dataset.py"),
            "--seed",
            str(seed),
            "--shipments",
            str(shipments),
            "--output",
            str(output_dir),
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    return json.loads((output_dir / "MANIFEST.json").read_text(encoding="utf-8"))


def replay_wire_outputs(
    engine: "RiskEngine",
    events: Sequence["TelemetryEvent"],
    decision_times: Sequence[tuple[str, datetime]],
) -> list[bytes]:
    """Ingest events and return serialized predictions."""
    for event in events:
        engine.ingest(event)
    return [engine.score(shipment_id, as_of).to_wire() for shipment_id, as_of in decision_times]


def verify_deterministic_replay(
    artifact_dir: Path,
    events: Sequence["TelemetryEvent"],
    decision_times: Sequence[tuple[str, datetime]],
    *,
    max_shipments: int = 32,
) -> dict[str, object]:
    """Run replay twice; predictions and snapshots must match byte-for-byte."""
    from dispatch_risk.solution import RiskEngine

    def run_once() -> tuple[list[bytes], bytes]:
        engine = RiskEngine(artifact_dir, max_shipments=max_shipments)
        wires = replay_wire_outputs(engine, events, decision_times)
        snap_path = artifact_dir / "_replay_snap.json"
        engine.snapshot(snap_path)
        snap_bytes = snap_path.read_bytes()
        snap_path.unlink(missing_ok=True)
        return wires, snap_bytes

    first_wires, first_snap = run_once()
    second_wires, second_snap = run_once()
    return {
        "prediction_runs_match": first_wires == second_wires,
        "snapshot_bytes_match": first_snap == second_snap,
        "prediction_count": len(first_wires),
        "first_digest": hashlib.sha256(b"".join(first_wires)).hexdigest()[:16],
    }


INTERVIEW_DENSE_READINGS_PATCH = """\
# Add inside _prediction_reasons in src/dispatch_risk/solution.py
# after the existing reason checks:

    if int(features["temp_reading_count"]) >= 8:
        reasons.append("dense_readings")
"""


@contextmanager
def patched_prediction_reasons(
    patch_fn: Callable[[list[str], Mapping[str, float | int | str | None], float], None],
):
    """Temporarily extend _prediction_reasons (notebook sandbox for interview drill 5)."""
    from dispatch_risk import solution

    original = solution._prediction_reasons

    def wrapped(
        features: Mapping[str, float | int | str | None],
        probability: float,
    ) -> tuple[str, ...]:
        reasons = list(original(features, probability))
        patch_fn(reasons, features, probability)
        return tuple(reasons)

    solution._prediction_reasons = wrapped
    try:
        yield
    finally:
        solution._prediction_reasons = original


@contextmanager
def apply_dense_readings_reason_patch():
    """Sandbox the interview patch: dense_readings when temp_reading_count >= 8."""

    def _add_dense_readings(
        reasons: list[str],
        features: Mapping[str, float | int | str | None],
        _probability: float,
    ) -> None:
        if int(features["temp_reading_count"]) >= 8 and "dense_readings" not in reasons:
            reasons.append("dense_readings")

    with patched_prediction_reasons(_add_dense_readings):
        yield


def score_decisions_table(
    artifact_dir: Path,
    events: Sequence["TelemetryEvent"],
    decision_times: Sequence[tuple[str, datetime]],
    *,
    max_shipments: int = 32,
) -> "pd.DataFrame":
    """Score each decision time and return a compact comparison table."""
    import pandas as pd

    from dispatch_risk.solution import RiskEngine

    engine = RiskEngine(artifact_dir, max_shipments=max_shipments)
    for event in events:
        engine.ingest(event)
    rows: list[dict[str, object]] = []
    for shipment_id, as_of in decision_times:
        pred = engine.score(shipment_id, as_of)
        rows.append(
            {
                "shipment_id": shipment_id,
                "as_of": as_of.strftime("%Y-%m-%d %H:%M UTC"),
                "probability": round(pred.probability, 4),
                "feature_digest": pred.feature_digest,
                "reasons": ", ".join(pred.reasons) or "(none)",
                "wire_digest": hashlib.sha256(pred.to_wire()).hexdigest()[:12],
            }
        )
    return pd.DataFrame(rows)


def compare_reason_patch(
    artifact_dir: Path,
    events: Sequence["TelemetryEvent"],
    decision_times: Sequence[tuple[str, datetime]],
    *,
    max_shipments: int = 32,
) -> dict[str, object]:
    """Show how a reasons-only patch affects wire output while keeping model scores stable."""
    import pandas as pd

    before_df = score_decisions_table(
        artifact_dir, events, decision_times, max_shipments=max_shipments
    )
    with apply_dense_readings_reason_patch():
        after_df = score_decisions_table(
            artifact_dir, events, decision_times, max_shipments=max_shipments
        )
        replay = verify_deterministic_replay(
            artifact_dir,
            events,
            decision_times,
            max_shipments=max_shipments,
        )

    merged = before_df.merge(after_df, on=["shipment_id", "as_of"], suffixes=("_before", "_after"))
    merged["prob_changed"] = merged["probability_before"] != merged["probability_after"]
    merged["digest_changed"] = merged["feature_digest_before"] != merged["feature_digest_after"]
    merged["reasons_changed"] = merged["reasons_before"] != merged["reasons_after"]
    merged["added_dense_readings"] = merged["reasons_after"].str.contains("dense_readings", na=False)

    return {
        "comparison": merged,
        "rows_with_new_reason": int(merged["added_dense_readings"].sum()),
        "probability_unchanged": not bool(merged["prob_changed"].any()),
        "feature_digest_unchanged": not bool(merged["digest_changed"].any()),
        "replay_after_patch": replay,
    }


def run_determinism_pytests(project_root: Path) -> subprocess.CompletedProcess[str]:
    """Run the three replay determinism tests used in the interview notebook."""
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_solution.py::TestReplay::test_replay_is_deterministic",
            "tests/test_solution.py::TestScoring::test_late_correction_does_not_change_past_score",
            "tests/test_solution.py::TestReplay::test_snapshot_bytes_are_deterministic",
            "-q",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )


CUSTOMER_REQUIREMENT_REJECTIONS: list[dict[str, str]] = [
    {
        "customer_note": "Random 80/20 row split",
        "decision": "Rejected",
        "alternative": "Hold out later whole shipments by first decision time so the same shipment does not appear in both train and test.",
    },
    {
        "customer_note": "Apply newest revision to old decisions",
        "decision": "Rejected",
        "alternative": "At each score time, use only the highest revision whose received_at timestamp was already known.",
    },
    {
        "customer_note": "Sort by device time for knowledge cutoff",
        "decision": "Reinterpreted",
        "alternative": "Process delivery order as received; use device_time only to order readings after the received_at cutoff.",
    },
    {
        "customer_note": "Deduplicate on shipment_id",
        "decision": "Rejected",
        "alternative": "Keep per-shipment event history keyed by event_id and revision because one shipment has many valid events.",
    },
    {
        "customer_note": "Kafka exactly-once removes snapshot need",
        "decision": "Rejected",
        "alternative": "Make ingest idempotent in application code and persist deterministic snapshots for replay and recovery.",
    },
    {
        "customer_note": "Return 0.0 when model cannot load",
        "decision": "Rejected",
        "alternative": "Validate a candidate model before swap; if loading fails, keep serving the previous valid model.",
    },
    {
        "customer_note": "Use full incident table in features",
        "decision": "Rejected",
        "alternative": "Use incident records only to build labels; never feed future incident information into features.",
    },
    {
        "customer_note": "AUC above 0.90 is enough for launch",
        "decision": "Rejected",
        "alternative": "Report PR-AUC, Brier score, constant-baseline comparison, and operational slices before discussing launch.",
    },
    {
        "customer_note": "Keep every shipment in memory",
        "decision": "Rejected",
        "alternative": "Enforce max_shipments with LRU eviction and remove that shipment's idempotency keys from auxiliary state.",
    },
    {
        "customer_note": "Reload may clear in-memory state",
        "decision": "Rejected",
        "alternative": "Reload swaps only the validated model object; retained shipment event state stays untouched.",
    },
]


def customer_requirement_table() -> "pd.DataFrame":
    """Return the 10 customer notes and how this solution handles them."""
    import pandas as pd

    return pd.DataFrame(CUSTOMER_REQUIREMENT_REJECTIONS)


INTERVIEW_FEATURE_PATCH = """\
# Example: add a feature in src/dispatch_risk/solution.py

# 1) FEATURE_NAMES tuple:
#    "compressor_on_count",

# 2) Inside _build_features():
#    compressor_on_count = sum(
#        1 for event in events
#        if event.kind == "compressor_status" and event.value in (1, 1.0, "on", True)
#    )

# 3) Add to the returned feature dict, retrain, redeploy artifact.
# Snapshots from before the change still load; scores change only after retrain.
"""


def compare_time_based_split(
    rows: Sequence["TrainingRow"],
) -> dict[str, object]:
    """Contrast shipment holdout with a time-ordered row split (also leaky)."""
    from sklearn.metrics import average_precision_score

    from dispatch_risk.solution import (
        _make_candidate_model,
        _positive_class_probabilities,
        _rows_to_xy,
        _split_by_shipment,
    )

    train_ship, test_ship = _split_by_shipment(rows)
    eval_ship = test_ship if test_ship else list(rows)
    x_train, y_train = _rows_to_xy(train_ship)
    x_eval, y_eval = _rows_to_xy(eval_ship)
    ship_model = _make_candidate_model("logreg", y_train)
    ship_model.fit(x_train, y_train)
    shipment_pr_auc = float(
        average_precision_score(y_eval, _positive_class_probabilities(ship_model, x_eval))
    )

    sorted_rows = sorted(rows, key=lambda row: row.decision_time)
    split_at = max(1, int(len(sorted_rows) * 0.8))
    train_time = sorted_rows[:split_at]
    test_time = sorted_rows[split_at:] or sorted_rows
    train_ids = {row.shipment_id for row in train_time}
    leaked = sum(1 for row in test_time if row.shipment_id in train_ids)
    x_train_t, y_train_t = _rows_to_xy(train_time)
    x_test_t, y_test_t = _rows_to_xy(test_time)
    time_model = _make_candidate_model("logreg", y_train_t)
    time_model.fit(x_train_t, y_train_t)
    time_pr_auc = float(
        average_precision_score(y_test_t, _positive_class_probabilities(time_model, x_test_t))
    )

    return {
        "shipment_split_pr_auc": shipment_pr_auc,
        "time_split_pr_auc": time_pr_auc,
        "test_rows_leaking_shipments": leaked,
        "test_row_count": len(test_time),
    }


def ingest_delay_summary(events: Sequence["TelemetryEvent"]) -> dict[str, float]:
    """Summarize received_at minus device_time delays in minutes."""
    delays: list[float] = []
    for event in events:
        delay_min = (
            event.received_at.astimezone(timezone.utc) - event.device_time.astimezone(timezone.utc)
        ).total_seconds() / 60.0
        delays.append(delay_min)
    if not delays:
        return {"count": 0.0, "p50_min": 0.0, "p95_min": 0.0, "max_min": 0.0}
    delays.sort()
    p95_index = max(0, int(len(delays) * 0.95) - 1)
    return {
        "count": float(len(delays)),
        "p50_min": round(delays[len(delays) // 2], 1),
        "p95_min": round(delays[p95_index], 1),
        "max_min": round(delays[-1], 1),
    }


def demo_concurrency_smoke(
    artifact_dir: Path,
    *,
    threads: int = 8,
    events_per_thread: int = 40,
    max_shipments: int = 32,
) -> dict[str, object]:
    """Mirror tests/test_solution.py::TestOps concurrent ingest and score under load."""
    import threading

    from dispatch_risk.contracts import TelemetryEvent
    from dispatch_risk.solution import RiskEngine

    engine = RiskEngine(artifact_dir, max_shipments=max_shipments)
    errors: list[str] = []

    def worker(offset: int) -> None:
        try:
            for idx in range(events_per_thread):
                event = TelemetryEvent(
                    event_id=f"evt-{offset}-{idx}",
                    revision=1,
                    shipment_id=f"s-{offset}",
                    device_time=datetime(2026, 6, 1, idx % 12, tzinfo=timezone.utc),
                    received_at=datetime(2026, 6, 1, idx % 12, 5, tzinfo=timezone.utc),
                    kind="temperature_c",
                    value=4.0 + idx * 0.01,
                    source="sensor-north",
                    payload={},
                )
                engine.ingest(event)
                engine.score(f"s-{offset}", datetime(2026, 6, 1, 12, tzinfo=timezone.utc))
        except Exception as exc:  # pragma: no cover - surfaced via errors list
            errors.append(str(exc))

    thread_objs = [threading.Thread(target=worker, args=(offset,)) for offset in range(threads)]
    for thread in thread_objs:
        thread.start()
    for thread in thread_objs:
        thread.join()

    stats = engine.stats()
    return {
        "threads": threads,
        "events_per_thread": events_per_thread,
        "passed": not errors,
        "errors": errors,
        "active_shipments": int(stats["active_shipments"]),
        "ingest_count": int(stats["ingest_count"]),
        "score_count": int(stats["score_count"]),
        "model_version": str(stats["model_version"]),
    }


def run_interview_checklist(project_root: Path) -> dict[str, object]:
    """Run the pre-interview sanity checks (Drill 7 in interview_drills.ipynb)."""
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    artifact = project_root / "artifact" / "model.joblib"
    data = project_root / "data" / "events.jsonl"
    decisions = project_root / "DECISIONS.md"
    return {
        "pytest_passed": pytest_result.returncode == 0,
        "pytest_summary": pytest_result.stdout.strip().splitlines()[-1] if pytest_result.stdout else "",
        "artifact_ready": artifact.exists(),
        "training_data_ready": data.exists(),
        "decisions_doc": decisions.exists(),
    }


def events_known_at_wrong_policy(
    events: Sequence["TelemetryEvent"],
    as_of: datetime,
) -> list["TelemetryEvent"]:
    """Broken policy: highest revision wins even when received_at is after as_of."""
    best: dict[str, TelemetryEvent] = {}
    for event in events:
        current = best.get(event.event_id)
        if current is None or event.revision > current.revision:
            best[event.event_id] = event
    as_of = as_of.astimezone(timezone.utc)
    return sorted(
        best.values(),
        key=lambda item: (item.device_time.astimezone(timezone.utc), item.event_id),
    )


def compare_evaluation_splits(
    rows: Sequence["TrainingRow"],
    *,
    random_seed: int = 1729,
) -> dict[str, object]:
    """Contrast chronological shipment holdout (correct) with a random row split (leaky)."""
    import random

    from sklearn.metrics import average_precision_score

    from dispatch_risk.solution import (
        _make_candidate_model,
        _positive_class_probabilities,
        _rows_to_xy,
        _split_by_shipment,
    )

    train_rows, test_rows = _split_by_shipment(rows)
    eval_rows = test_rows if test_rows else list(train_rows)
    x_train, y_train = _rows_to_xy(train_rows)
    x_eval, y_eval = _rows_to_xy(eval_rows)
    shipment_model = _make_candidate_model("logreg", y_train)
    shipment_model.fit(x_train, y_train)
    shipment_probs = _positive_class_probabilities(shipment_model, x_eval)
    shipment_pr_auc = float(average_precision_score(y_eval, shipment_probs))

    shuffled = list(rows)
    random.Random(random_seed).shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * 0.8))
    row_train = shuffled[:split_at]
    row_test = shuffled[split_at:] or shuffled
    train_ids = {row.shipment_id for row in row_train}
    leaked_test_rows = sum(1 for row in row_test if row.shipment_id in train_ids)
    x_row_train, y_row_train = _rows_to_xy(row_train)
    x_row_test, y_row_test = _rows_to_xy(row_test)
    row_model = _make_candidate_model("logreg", y_row_train)
    row_model.fit(x_row_train, y_row_train)
    row_probs = _positive_class_probabilities(row_model, x_row_test)
    row_pr_auc = float(average_precision_score(y_row_test, row_probs))

    return {
        "shipment_split_pr_auc": shipment_pr_auc,
        "row_split_pr_auc": row_pr_auc,
        "train_shipments": len({row.shipment_id for row in train_rows}),
        "test_shipments": len({row.shipment_id for row in test_rows}),
        "test_rows_leaking_shipments": leaked_test_rows,
        "test_row_count": len(row_test),
    }


def label_horizon_summary(
    labels: Sequence[Mapping[str, object]],
    decision_times: Sequence[tuple[str, datetime]],
    *,
    old_hours: int = 6,
    new_hours: int = 4,
) -> dict[str, object]:
    """Count labels that change when the incident window shrinks."""

    def _label_at(hours: int, shipment_id: str, decision_time: datetime) -> int:
        decision_time = decision_time.astimezone(timezone.utc)
        window_end = decision_time + timedelta(hours=hours)
        for label in labels:
            if str(label["shipment_id"]) != shipment_id:
                continue
            incident_at = _parse_time(str(label["incident_at"]))
            if decision_time < incident_at <= window_end:
                return 1
        return 0

    flips: list[dict[str, object]] = []
    for shipment_id, decision_time in decision_times:
        old = _label_at(old_hours, shipment_id, decision_time)
        new = _label_at(new_hours, shipment_id, decision_time)
        if old != new:
            flips.append(
                {
                    "shipment_id": shipment_id,
                    "decision_time": decision_time.strftime("%Y-%m-%d %H:%M UTC"),
                    f"label_{old_hours}h": old,
                    f"label_{new_hours}h": new,
                }
            )
    positives_old = sum(
        _label_at(old_hours, shipment_id, decision_time) for shipment_id, decision_time in decision_times
    )
    positives_new = sum(
        _label_at(new_hours, shipment_id, decision_time) for shipment_id, decision_time in decision_times
    )
    return {
        "decision_times": len(decision_times),
        f"positives_{old_hours}h": positives_old,
        f"positives_{new_hours}h": positives_new,
        "flipped_rows": flips,
    }


def carrier_for_shipment(events: Sequence["TelemetryEvent"], shipment_id: str) -> str:
    """Map a shipment to north / central / coast from its sensor source."""
    for event in events:
        if event.shipment_id != shipment_id:
            continue
        source = event.source.removeprefix("sensor-")
        if source in {"north", "central", "coast"}:
            return source
    return "unknown"


def holdout_carrier_slices(
    rows: Sequence["TrainingRow"],
    events: Sequence["TelemetryEvent"],
    artifact_dir: Path,
) -> list[dict[str, object]]:
    """PR-AUC on the held-out split, broken out by carrier lane."""
    import joblib
    from sklearn.metrics import average_precision_score

    from dispatch_risk.solution import (
        _positive_class_probabilities,
        _rows_to_xy,
        _split_by_shipment,
    )

    _, test_rows = _split_by_shipment(rows)
    if not test_rows:
        test_rows = list(rows)
    bundle = joblib.load(artifact_dir / "model.joblib")
    model = bundle["model"]
    x_test, y_test = _rows_to_xy(test_rows)
    probabilities = _positive_class_probabilities(model, x_test)

    by_carrier: dict[str, list[tuple[int, float]]] = {}
    for row, label, prob in zip(test_rows, y_test.tolist(), probabilities, strict=True):
        carrier = carrier_for_shipment(events, row.shipment_id)
        by_carrier.setdefault(carrier, []).append((label, prob))

    slices: list[dict[str, object]] = []
    for carrier, pairs in sorted(by_carrier.items()):
        labels = [label for label, _ in pairs]
        probs = [prob for _, prob in pairs]
        positive_rate = sum(labels) / max(len(labels), 1)
        pr_auc = float(average_precision_score(labels, probs)) if len(set(labels)) > 1 else None
        slices.append(
            {
                "carrier": carrier,
                "rows": len(pairs),
                "positive_rate": round(positive_rate, 3),
                "pr_auc": None if pr_auc is None else round(pr_auc, 3),
            }
        )
    return slices


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _emit_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _temp_event(
    shipment_id: str,
    hour: int,
    start: datetime,
    temp: float,
    received_offset_min: int,
    revision: int = 1,
    payload: dict | None = None,
) -> dict:
    device_time = start + timedelta(hours=hour)
    return {
        "device_time": _iso(device_time),
        "event_id": f"{shipment_id}-temp-{hour:02d}",
        "kind": "temperature_c",
        "payload": payload or {"demo": "walkthrough"},
        "received_at": _iso(device_time + timedelta(minutes=received_offset_min)),
        "revision": revision,
        "shipment_id": shipment_id,
        "source": "sensor-coast",
        "value": round(temp, 3),
    }


def _door_event(
    shipment_id: str,
    hour: int,
    start: datetime,
    offset_min: int = 10,
    *,
    minute: int = 0,
    seq: int = 0,
) -> dict:
    device_time = start + timedelta(hours=hour, minutes=minute)
    suffix = f"{hour:02d}" if minute == 0 and seq == 0 else f"{hour:02d}m{minute:02d}s{seq}"
    return {
        "device_time": _iso(device_time),
        "event_id": f"{shipment_id}-door-{suffix}",
        "kind": "door_open",
        "payload": {"demo": "walkthrough"},
        "received_at": _iso(device_time + timedelta(minutes=offset_min)),
        "revision": 1,
        "shipment_id": shipment_id,
        "source": "ops-console",
        "value": 1.0,
    }


def _random_ingest_delay_min(rng: random.Random, *, late: bool = False) -> int:
    if late:
        return rng.choice([120, 150, 180, 210, 240, 300])
    return rng.choice([0, 1, 2, 4, 7, 11, 18, 25, 35, 48, 62, 90])


def _duplicate_delivery(event: dict, rng: random.Random, *, extra_delay_min: int | None = None) -> dict:
    """Exact redelivery of the same record (identical content, including received_at).

    Matches the graded generator: same (event_id, revision) must be byte-identical
    or RiskEngine treats it as a conflicting redelivery.
    """
    del rng, extra_delay_min  # kept for call-site compatibility
    return dict(event)


def _inject_duplicate_deliveries(
    events: list[dict],
    rng: random.Random,
    *,
    lo: int,
    hi: int,
    kinds: tuple[str, ...] = ("temperature_c",),
    skip_event_ids: set[str] | None = None,
) -> None:
    skip = skip_event_ids or set()
    pool = [
        event
        for event in events
        if event["kind"] in kinds and int(event["revision"]) == 1 and str(event["event_id"]) not in skip
    ]
    if not pool:
        return
    count = rng.randint(lo, min(hi, len(pool)))
    for source in rng.sample(pool, k=count):
        extra = rng.randint(45, 240) if source.get("payload", {}).get("delayed") else None
        events.append(_duplicate_delivery(source, rng, extra_delay_min=extra))


def _inject_door_opens(
    events: list[dict],
    shipment_id: str,
    start: datetime,
    rng: random.Random,
    hours: list[int],
) -> None:
    for hour in hours:
        events.append(
            _door_event(
                shipment_id,
                hour,
                start,
                offset_min=rng.randint(3, 55),
                minute=rng.choice([0, 10, 20, 30]),
                seq=rng.randint(0, 1),
            )
        )


def _demo_label(shipment_id: str, incident_at: datetime) -> dict:
    return {
        "incident_at": _iso(incident_at),
        "incident_id": f"inc-{shipment_id}",
        "label_available_at": _iso(incident_at + timedelta(hours=12)),
        "severity": 2,
        "shipment_id": shipment_id,
    }


def _demo_hour_range() -> range:
    return range(DEMO_STREAM_HOURS)


def _late_stream_hour() -> int:
    """When delayed-ingest simulation starts (~62% through the day)."""
    return max(1, int(DEMO_STREAM_HOURS * 0.625))


def _demo_decisions(shipment_id: str, start: datetime, hours: tuple[int, ...] = (8, 14, 20)) -> list[dict]:
    return [{"decision_time": _iso(start + timedelta(hours=h)), "shipment_id": shipment_id} for h in hours]


def _scenario_stable(start: datetime, rng: random.Random) -> tuple[list[dict], list[dict], list[dict], dict]:
    sid = "s-demo-stable"
    events = [
        _temp_event(sid, h, start, 3.5 + rng.gauss(0, 0.22), _random_ingest_delay_min(rng))
        for h in _demo_hour_range()
    ]
    _inject_duplicate_deliveries(events, rng, lo=4, hi=8)
    meta = {
        "shipment_id": sid,
        "use_case": "Stable cold chain",
        "talk_about": "Steady 3-4 C with duplicate deliveries. Repeats are ignored by event_id and revision.",
        "has_incident": False,
    }
    return events, [], _demo_decisions(sid, start), meta


def _scenario_warming(start: datetime, rng: random.Random) -> tuple[list[dict], list[dict], list[dict], dict]:
    sid = "s-demo-warming"
    incident_at = start + timedelta(hours=19)
    events = []
    for hour in _demo_hour_range():
        temp = 3.8 + hour * 0.22 + rng.gauss(0, 0.18)
        events.append(_temp_event(sid, hour, start, temp, _random_ingest_delay_min(rng)))
    _inject_duplicate_deliveries(events, rng, lo=2, hi=5)
    meta = {
        "shipment_id": sid,
        "use_case": "Gradual warming",
        "talk_about": "Temperature drifts up. Incident within 6 hours at the 14:00 score.",
        "has_incident": True,
        "incident_at": _iso(incident_at),
    }
    return events, [_demo_label(sid, incident_at)], _demo_decisions(sid, start), meta


def _scenario_correction(start: datetime, rng: random.Random) -> tuple[list[dict], list[dict], list[dict], dict]:
    sid = "s-demo-correction"
    incident_at = start + timedelta(hours=20)
    events = []
    for hour in _demo_hour_range():
        temp = 3.6 + hour * 0.05 if hour < 12 else 4.0 + (hour - 11) * 0.55
        events.append(_temp_event(sid, hour, start, temp + rng.gauss(0, 0.2), _random_ingest_delay_min(rng)))
    _inject_duplicate_deliveries(
        events,
        rng,
        lo=2,
        hi=4,
        skip_event_ids={f"{sid}-temp-08"},
    )
    events.append(
        _temp_event(
            sid,
            8,
            start,
            2.0,
            received_offset_min=0,
            revision=2,
            payload={"correction": "calibrated", "demo": "walkthrough"},
        )
    )
    events[-1]["received_at"] = _iso(start + timedelta(hours=22))
    meta = {
        "shipment_id": sid,
        "use_case": "Late correction",
        "talk_about": "Correction arrives at hour 22 but score at hour 8 must stay the same.",
        "has_incident": True,
        "incident_at": _iso(incident_at),
    }
    return events, [_demo_label(sid, incident_at)], _demo_decisions(sid, start), meta


def _scenario_door(start: datetime, rng: random.Random) -> tuple[list[dict], list[dict], list[dict], dict]:
    sid = "s-demo-door"
    incident_at = start + timedelta(hours=17)
    events = []
    door_hours = sorted(set(rng.sample(list(range(7, 21)), k=rng.randint(4, 6)) + [12]))
    for hour in _demo_hour_range():
        temp = 4.0 + rng.gauss(0, 0.15)
        if hour >= 12:
            temp += (hour - 11) * 0.85
        if hour in door_hours:
            temp += rng.uniform(0.4, 1.2)
        events.append(_temp_event(sid, hour, start, temp, _random_ingest_delay_min(rng)))
    _inject_door_opens(events, sid, start, rng, door_hours)
    _inject_duplicate_deliveries(events, rng, lo=2, hi=5)
    meta = {
        "shipment_id": sid,
        "use_case": "Door open event",
        "talk_about": (
            f"{len(door_hours)} door-open events and warming. door_open_seen raises the score."
        ),
        "has_incident": True,
        "incident_at": _iso(incident_at),
    }
    return events, [_demo_label(sid, incident_at)], _demo_decisions(sid, start), meta


def _scenario_delayed(start: datetime, rng: random.Random) -> tuple[list[dict], list[dict], list[dict], dict]:
    sid = "s-demo-delayed"
    incident_at = start + timedelta(hours=18)
    late_from = _late_stream_hour()
    events = []
    for hour in _demo_hour_range():
        temp = 4.2 + hour * 0.18 + rng.gauss(0, 0.18)
        delay = _random_ingest_delay_min(rng, late=hour >= late_from)
        payload = {"demo": "walkthrough", "delayed": hour >= late_from}
        events.append(
            _temp_event(
                sid,
                hour,
                start,
                temp,
                delay,
                payload=payload,
            )
        )
    _inject_duplicate_deliveries(events, rng, lo=3, hi=6)
    meta = {
        "shipment_id": sid,
        "use_case": "Delayed telemetry",
        "talk_about": (
            f"Readings from hour {late_from}+ arrive 3 hours late. Only received_at counts when scoring."
        ),
        "has_incident": True,
        "incident_at": _iso(incident_at),
    }
    return events, [_demo_label(sid, incident_at)], _demo_decisions(sid, start), meta


def generate_walkthrough(seed: int) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Build five scripted demo shipments for inference (not used in training)."""
    rng = random.Random(seed)
    start = datetime(2026, 3, 15, 0, 0, tzinfo=timezone.utc)

    builders = (_scenario_stable, _scenario_warming, _scenario_correction, _scenario_door, _scenario_delayed)
    all_events: list[dict] = []
    all_labels: list[dict] = []
    all_decisions: list[dict] = []
    shipments: list[dict] = []

    for builder in builders:
        events, labels, decisions, meta = builder(start, rng)
        all_events.extend(events)
        all_labels.extend(labels)
        all_decisions.extend(decisions)
        meta["events"] = len(events)
        meta["decision_times"] = len(decisions)
        shipments.append(meta)

    all_events.sort(key=lambda row: (row["received_at"], row["event_id"], row["revision"]))

    summary = {
        "seed": seed,
        "shipments": shipments,
        "shipment_count": len(shipments),
        "events": len(all_events),
        "labels": len(all_labels),
        "decision_times": len(all_decisions),
        "purpose": "Five demo shipments for inference (not used in training)",
        "stream_hours": DEMO_STREAM_HOURS,
    }
    return all_events, all_labels, all_decisions, summary


# ---------------------------------------------------------------------------
# Notebook visualization (matplotlib + styled tables, matches Streamlit UI)
# ---------------------------------------------------------------------------

SHIPMENT_COLORS = {
    "s-demo-stable": "#2563eb",
    "s-demo-warming": "#dc2626",
    "s-demo-correction": "#9333ea",
    "s-demo-door": "#ea580c",
    "s-demo-delayed": "#0891b2",
}

_RISK_LOW = "#22a06b"
_RISK_MID = "#e06c00"
_RISK_HIGH = "#c9372c"
_PRIMARY = "#1f5c99"
_MUTED = "#cbd5e1"


def configure_notebook_style() -> None:
    """Apply chart defaults shared with the Streamlit dashboard."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#fafbfc",
            "axes.edgecolor": "#c8d0da",
            "axes.labelcolor": "#1c2430",
            "axes.titlecolor": "#1c2430",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "font.family": "sans-serif",
            "grid.color": "#e2e8f0",
            "grid.alpha": 0.45,
        }
    )


def _risk_color(probability: float) -> str:
    if probability < 0.3:
        return _RISK_LOW
    if probability < 0.7:
        return _RISK_MID
    return _RISK_HIGH


def _event_flags(event, seen: set[tuple[str, int]]) -> tuple[list[str], bool]:
    key = (str(event.event_id), int(event.revision))
    duplicate = key in seen
    if not duplicate:
        seen.add(key)
    delay_min = (event.received_at.astimezone(timezone.utc) - event.device_time.astimezone(timezone.utc)).total_seconds() / 60.0
    flags: list[str] = []
    if duplicate:
        flags.append("duplicate")
    if int(event.revision) > 1:
        flags.append("correction")
    if str(event.kind) == "door_open":
        flags.append("door")
    if delay_min >= 60:
        flags.append("delayed")
    return flags, duplicate


def events_to_display_df(events: Sequence[object]) -> "pd.DataFrame":
    """Build a notebook-friendly event log with delivery flags."""
    import pandas as pd

    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, object]] = []
    for idx, event in enumerate(events):
        flags, _ = _event_flags(event, seen)
        delay_min = (
            event.received_at.astimezone(timezone.utc) - event.device_time.astimezone(timezone.utc)
        ).total_seconds() / 60.0
        rows.append(
            {
                "#": idx,
                "event_id": event.event_id,
                "kind": event.kind,
                "value": event.value,
                "device_time": event.device_time.strftime("%Y-%m-%d %H:%M"),
                "received_at": event.received_at.strftime("%Y-%m-%d %H:%M"),
                "revision": event.revision,
                "flags": ", ".join(flags) if flags else "normal",
                "delay_min": round(delay_min, 1),
            }
        )
    return pd.DataFrame(rows)


def style_manifest_table(df: "pd.DataFrame") -> "pd.io.formats.style.Styler":
    def _incident(val: object) -> str:
        if val is True:
            return "background-color: #fee2e2; color: #991b1b"
        if val is False:
            return "background-color: #ecfdf5; color: #166534"
        return ""

    return (
        df.style.hide(axis="index")
        .set_properties(**{"font-size": "0.92rem", "line-height": "1.45"})
        .map(_incident, subset=["has_incident"])
        .set_table_styles(
            [
                {"selector": "th", "props": [("background-color", "#f1f5f9"), ("color", "#334155"), ("font-weight", "600")]},
                {"selector": "td", "props": [("padding", "0.45rem 0.65rem")]},
            ]
        )
    )


def style_events_table(df: "pd.DataFrame") -> "pd.io.formats.style.Styler":
    def _row(row: "pd.Series") -> list[str]:
        flag = str(row["flags"])
        if "correction" in flag:
            return ["background-color: #f3e8ff"] * len(row)
        if "duplicate" in flag:
            return ["background-color: #fef3c7"] * len(row)
        if "door" in flag:
            return ["background-color: #ffedd5"] * len(row)
        if "delayed" in flag:
            return ["background-color: #e0f2fe"] * len(row)
        return [""] * len(row)

    return (
        df.style.apply(_row, axis=1)
        .set_properties(**{"font-size": "0.88rem"})
        .set_table_styles([{"selector": "th", "props": [("background-color", "#f1f5f9"), ("font-weight", "600")]}])
    )


def style_predictions_table(df: "pd.DataFrame") -> "pd.io.formats.style.Styler":
    def _risk_cell(val: object) -> str:
        try:
            prob = float(val)
        except (TypeError, ValueError):
            return ""
        return f"background-color: {_risk_color(prob)}22; color: #1c2430; font-weight: 600"

    def _label_cell(val: object) -> str:
        if val in (1, True, "1"):
            return "background-color: #fee2e2; color: #991b1b; font-weight: 600"
        return "background-color: #ecfdf5; color: #166534"

    styler = df.style.hide(axis="index").set_properties(**{"font-size": "0.92rem"})
    if "probability" in df.columns:
        styler = styler.map(_risk_cell, subset=["probability"])
    if "true_label_next_6h" in df.columns:
        styler = styler.map(_label_cell, subset=["true_label_next_6h"])
    return styler.set_table_styles(
        [{"selector": "th", "props": [("background-color", "#f1f5f9"), ("font-weight", "600")]}]
    )


def plot_model_comparison(comparison: dict[str, dict[str, float]], selected: str):
    """Bar chart: logreg vs HGB (PR-AUC and Brier)."""
    import matplotlib.pyplot as plt

    short = {"logreg": "Logistic reg.", "hgb": "Grad. boosting"}
    keys = list(comparison.keys())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric, ylab in zip(
        axes,
        ["pr_auc", "brier_score"],
        ["PR-AUC ↑", "Brier ↓"],
        strict=True,
    ):
        vals = [comparison[k][metric] for k in keys]
        colors = [_PRIMARY if k == selected else _MUTED for k in keys]
        ax.bar([short.get(k, k) for k in keys], vals, color=colors, width=0.55)
        ax.set_ylabel(ylab)
        ax.grid(axis="y")
    fig.suptitle(f"Model comparison · winner: {short.get(selected, selected)}", fontsize=11)
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.18)
    return fig


def plot_holdout_panels(holdout: dict[str, object]):
    """Confusion matrix + PR curve for the held-out split."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay

    matrix = np.array(holdout["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["No incident", "Incident"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False, text_kw={"fontsize": 12}
    )
    correct = int(matrix[0, 0] + matrix[1, 1])
    axes[0].set_title(f"Confusion matrix · {correct}/{int(matrix.sum())} correct", pad=10)

    pr = holdout["pr_curve"]
    axes[1].plot(pr["recall"], pr["precision"], linewidth=2.2, color=_PRIMARY, label=f"Model {holdout['pr_auc']:.3f}")
    baseline = sum(holdout["y_true"]) / max(len(holdout["y_true"]), 1)
    axes[1].axhline(y=max(baseline, 0.01), color="#94a3b8", linestyle="--", label="Baseline")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="lower left")
    axes[1].grid(True)
    axes[1].set_title(f"PR curve · AUC {holdout['pr_auc']:.3f}", pad=10)
    fig.subplots_adjust(wspace=0.28, bottom=0.14, top=0.88)
    return fig


def plot_risk_scores(pred_df: "pd.DataFrame", shipment_id: str):
    """Bar chart of risk probability at each decision time."""
    import matplotlib.pyplot as plt

    labels = pred_df["as_of"].str[-5:].tolist() if "as_of" in pred_df.columns else list(range(len(pred_df)))
    values = pred_df["probability"].tolist()
    colors = [_risk_color(v) for v in values]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(labels, values, color=colors, width=0.5)
    ax.axhline(0.5, color="#64748b", linestyle="--", linewidth=1.2, label="0.5 threshold")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Risk probability")
    ax.set_title(f"Risk scores · {shipment_id}")
    ax.grid(axis="y")
    fig.subplots_adjust(bottom=0.14, top=0.88)
    return fig


def plot_shipment_timeline(events: Sequence[object], shipment_id: str, *, time_col: str = "received_at"):
    """Single-shipment temperature timeline (matches dashboard markers)."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    color = SHIPMENT_COLORS.get(shipment_id, "#64748b")
    seen: set[tuple[str, int]] = set()
    temps: list[tuple[datetime, float, str]] = []
    doors: list[datetime] = []
    for event in events:
        flags, _ = _event_flags(event, seen)
        flag_str = ", ".join(flags) if flags else "normal"
        tstamp = event.received_at if time_col == "received_at" else event.device_time
        tstamp = tstamp.astimezone(timezone.utc)
        if str(event.kind) == "temperature_c":
            temps.append((tstamp, float(event.value), flag_str))
        elif str(event.kind) == "door_open":
            doors.append(tstamp)

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    if temps:
        times = [t for t, _, _ in temps]
        values = [v for _, v, _ in temps]
        ax.plot(times, values, color=color, linewidth=1.8, alpha=0.35, zorder=1)
        for tstamp, value, flag_str in temps:
            if "duplicate" in flag_str:
                ax.scatter(tstamp, value, marker="o", s=68, facecolors="none", edgecolors="#1c2430", linewidths=1.6, zorder=3)
            elif "correction" in flag_str:
                ax.scatter(tstamp, value, marker="D", s=88, color=color, edgecolors="#1c2430", linewidths=0.8, zorder=4)
            else:
                ax.scatter(tstamp, value, color=color, s=38, zorder=2)
        max_temp = max(values + [8.5])
    else:
        max_temp = 8.5

    if doors:
        ax.scatter(doors, [max_temp + 0.55] * len(doors), marker="v", s=90, color=color, edgecolors="#1c2430", zorder=5)

    ax.axhline(8.0, color="#94a3b8", linestyle="--", linewidth=1.2, label="8 °C guide")
    ax.set_ylabel("Temperature (°C)")
    axis_label = "Received at" if time_col == "received_at" else "Device time"
    ax.set_xlabel(axis_label)
    ax.set_title(f"Temperature timeline · {shipment_id}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.grid(True)
    ax.set_ylim(0, max_temp + 1.2)
    ax.margins(x=0.02)
    fig.subplots_adjust(bottom=0.18, top=0.88)
    return fig


def demo_late_correction(
    artifact_dir: Path = ARTIFACT_DIR,
    data_dir: Path = WALKTHROUGH_DIR,
    shipment_id: str = "s-demo-correction",
) -> tuple[object, object, datetime]:
    """Score before/after a late correction on the dedicated demo shipment."""
    from dispatch_risk.solution import RiskEngine

    events = events_for_shipment(load_events(data_dir), shipment_id)
    decisions = decisions_for_shipment(load_decision_times(data_dir), shipment_id)
    correction = next((e for e in events if e.revision > 1), None)
    if correction is None:
        raise ValueError(f"{shipment_id} has no late correction event (revision > 1)")

    corr_hour = correction.device_time.astimezone(timezone.utc).hour
    as_of_past = next((t for _, t in decisions if t.astimezone(timezone.utc).hour == corr_hour), None)
    if as_of_past is None:
        raise ValueError(f"No decision time at hour {corr_hour:02d} for {shipment_id}")

    engine = RiskEngine(artifact_dir, max_shipments=10)
    for event in events:
        if event is correction:
            break
        engine.ingest(event)
    before = engine.score(shipment_id, as_of_past)
    engine.ingest(correction)
    after = engine.score(shipment_id, as_of_past)
    return before, after, as_of_past


def plot_correction_before_after(before_prob: float, after_prob: float, *, as_of_label: str):
    """Before/after bar chart for late correction demo."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    labels = ["Before correction", "After correction"]
    values = [before_prob, after_prob]
    colors = [_risk_color(v) for v in values]
    ax.bar(labels, values, color=colors, width=0.55)
    ax.axhline(0.5, color="#64748b", linestyle="--")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Risk probability")
    unchanged = "unchanged ✓" if before_prob == after_prob else "changed"
    ax.set_title(f"Late correction @ {as_of_label} · score {unchanged}")
    ax.grid(axis="y")
    fig.subplots_adjust(bottom=0.14, top=0.82)
    return fig


def plot_stream_compare(wide_df: "pd.DataFrame", title: str = "Same model · two demo seeds"):
    """Grouped bar chart comparing risk scores across streams."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(wide_df.index))
    width = 0.35
    cols = list(wide_df.columns)
    for idx, col in enumerate(cols):
        offset = (idx - (len(cols) - 1) / 2) * width
        vals = wide_df[col].astype(float).tolist()
        ax.bar(x + offset, vals, width=width, label=col, color=_PRIMARY if idx == 0 else "#64748b")
    ax.set_xticks(x)
    ax.set_xticklabels(wide_df.index.tolist())
    ax.set_ylabel("Risk probability")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="#64748b", linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y")
    fig.subplots_adjust(bottom=0.14, top=0.86)
    return fig


def write_walkthrough_dataset(output: Path = WALKTHROUGH_DIR, seed: int = 4242) -> dict[str, object]:
    """Write demo walkthrough JSONL files and return the manifest summary."""
    events, labels, decisions, summary = generate_walkthrough(seed)
    _emit_jsonl(output / "events.jsonl", events)
    _emit_jsonl(output / "labels.jsonl", labels)
    _emit_jsonl(output / "decision_times.jsonl", decisions)
    (output / "MANIFEST.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo walkthrough shipments for notebook/UI.")
    parser.add_argument("--generate-walkthrough", action="store_true", help="Write 5 demo shipments to disk")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output", type=Path, default=WALKTHROUGH_DIR)
    args = parser.parse_args()
    if not args.generate_walkthrough:
        parser.error("use --generate-walkthrough to write demo shipment files")
    summary = write_walkthrough_dataset(args.output, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
