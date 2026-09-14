"""Tests for dispatch_risk.

How pytest runs these:
    pytest -v
        Discovers every class named Test* and every function named test_* in this file.
        Runs each test in isolation. Green = assertion passed.

    pytest tests/test_solution.py::TestIngest -v
        Runs only one group (useful during a walkthrough).

Fixtures (reused setup):
    data_dir       - loads generated data/ (skips if you forgot generate_dataset.py)
    artifact_dir   - trains a small model in a temp folder for engine tests
    fresh_stream_dir - generates a new 120-shipment stream for replay tests

What each group proves (maps to README):
    TestContract         - API output format is stable JSON bytes
    TestTrainingLabels   - Required #1: training rows respect time and labels
    TestIngest           - Required #3: duplicate, revision, out-of-order ingest
    TestScoring          - Required #3: score time, corrections, degraded mode
    TestReplay           - Required #3: same history gives same output bytes
    TestSnapshotRestore  - Required #3: crash recovery keeps duplicate tracking
    TestModelReload      - Required #3: safe model swap on deploy
    TestMemoryEviction   - Required #3: memory cap under load
    TestOps              - threading + artifact/metrics completeness
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Must run before joblib/sklearn are imported.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import pytest

from dispatch_risk.contracts import Prediction, TelemetryEvent
from dispatch_risk.solution import (
    RiskEngine,
    _feature_digest,
    build_training_rows,
    train,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------


def parse_event(raw: dict[str, object]) -> TelemetryEvent:
    """Parse one JSONL event dict into a ``TelemetryEvent``."""
    return TelemetryEvent(
        event_id=str(raw["event_id"]),
        revision=int(raw["revision"]),
        shipment_id=str(raw["shipment_id"]),
        device_time=datetime.fromisoformat(str(raw["device_time"]).replace("Z", "+00:00")),
        received_at=datetime.fromisoformat(str(raw["received_at"]).replace("Z", "+00:00")),
        kind=str(raw["kind"]),
        value=raw["value"],
        source=str(raw["source"]),
        payload=dict(raw["payload"]),  # type: ignore[arg-type]
    )


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load a JSON Lines file into a list of dicts."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_temp_event(
    *,
    event_id: str = "evt-1",
    revision: int = 1,
    shipment_id: str = "s-1",
    device_time: datetime,
    received_at: datetime,
    kind: str = "temperature_c",
    value: float = 4.0,
    source: str = "sensor-north",
) -> TelemetryEvent:
    """Factory for temperature events in unit tests."""
    return TelemetryEvent(
        event_id=event_id,
        revision=revision,
        shipment_id=shipment_id,
        device_time=device_time,
        received_at=received_at,
        kind=kind,
        value=value,
        source=source,
        payload={},
    )


def train_minimal_model(artifact_dir: Path, data_dir: Path | None = None) -> None:
    """Train on a small subset of ``data/`` for fast isolated engine tests."""
    root = data_dir or Path(__file__).parents[1] / "data"
    events = [parse_event(row) for row in load_jsonl(root / "events.jsonl")[:500]]
    labels = load_jsonl(root / "labels.jsonl")
    decision_times = [
        (str(row["shipment_id"]), datetime.fromisoformat(str(row["decision_time"]).replace("Z", "+00:00")))
        for row in load_jsonl(root / "decision_times.jsonl")[:40]
    ]
    train(build_training_rows(events, labels, decision_times), artifact_dir)


def train_full_model(artifact_dir: Path, data_dir: Path) -> None:
    """Train on the full generated dataset (used to refresh ``artifact/``)."""
    events = [parse_event(row) for row in load_jsonl(data_dir / "events.jsonl")]
    labels = load_jsonl(data_dir / "labels.jsonl")
    decision_times = [
        (str(row["shipment_id"]), datetime.fromisoformat(str(row["decision_time"]).replace("Z", "+00:00")))
        for row in load_jsonl(data_dir / "decision_times.jsonl")
    ]
    train(build_training_rows(events, labels, decision_times), artifact_dir)


def replay_engine(
    engine: RiskEngine,
    events: list[TelemetryEvent],
    decision_times: list[tuple[str, datetime]],
) -> list[bytes]:
    """Ingest events then return ``Prediction.to_wire()`` bytes for each decision."""
    outputs: list[bytes] = []
    for event in events:
        engine.ingest(event)
    for shipment_id, as_of in decision_times:
        outputs.append(engine.score(shipment_id, as_of).to_wire())
    return outputs


@pytest.fixture
def data_dir() -> Path:
    """Project ``data/`` directory; skips if ``events.jsonl`` has not been generated."""
    path = Path(__file__).parents[1] / "data"
    if not (path / "events.jsonl").exists():
        pytest.skip("generated dataset missing")
    return path


@pytest.fixture
def artifact_dir(tmp_path: Path, data_dir: Path) -> Path:
    """Minimal trained model in a temp dir for engine tests."""
    out = tmp_path / "artifact"
    train_minimal_model(out, data_dir)
    return out


@pytest.fixture
def fresh_stream_dir(tmp_path: Path) -> Path:
    """New 120-shipment stream (seed 4242) for generalization / replay tests."""
    out = tmp_path / "fresh_data"
    subprocess.run(
        [sys.executable, "tools/generate_dataset.py", "--seed", "4242", "--shipments", "120", "--output", str(out)],
        check=True,
        cwd=Path(__file__).parents[1],
    )
    return out


# ---------------------------------------------------------------------------
# Contract & wire format
# ---------------------------------------------------------------------------


class TestContract:
    """Prediction.to_wire() JSON is sorted and stable (needed for replay checks)."""

    def test_prediction_wire_format_is_canonical(self) -> None:
        """Same prediction always serializes to the same byte string."""
        prediction = Prediction(
            shipment_id="s-1",
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
            probability=0.25,
            model_version="m-1",
            feature_digest="abc",
            degraded=False,
            reasons=("temperature_high",),
        )
        assert prediction.to_wire() == (
            b'{"as_of":"2026-01-01T00:00:00+00:00","degraded":false,'
            b'"feature_digest":"abc","model_version":"m-1","probability":0.25,'
            b'"reasons":["temperature_high"],"shipment_id":"s-1"}'
        )


# ---------------------------------------------------------------------------
# Training rows & labels
# ---------------------------------------------------------------------------


class TestTrainingLabels:
    """build_training_rows: no future data, correct labels, skip unknown outcomes."""

    def test_training_respects_received_at_cutoff(self) -> None:
        """Events with received_at after decision_time are excluded from features."""
        decision_time = datetime(2026, 1, 1, 10, tzinfo=UTC)
        early = TelemetryEvent(
            event_id="evt-early",
            revision=1,
            shipment_id="s-1",
            device_time=datetime(2026, 1, 1, 8, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 9, tzinfo=UTC),
            kind="temperature_c",
            value=4.0,
            source="sensor-north",
            payload={},
        )
        late = TelemetryEvent(
            event_id="evt-late",
            revision=1,
            shipment_id="s-1",
            device_time=datetime(2026, 1, 1, 9, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 11, tzinfo=UTC),
            kind="temperature_c",
            value=12.0,
            source="sensor-north",
            payload={},
        )
        rows = build_training_rows([early, late], [], [("s-1", decision_time)])
        assert rows[0].metadata["known_event_count"] == 1
        assert rows[0].features["latest_temp_c"] == 4.0

    def test_label_window_boundaries(self) -> None:
        """Label window is (decision_time, decision_time + 6h], not inclusive at start."""
        decision_time = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
        labels = [
            {"shipment_id": "s-a", "incident_at": decision_time},
            {"shipment_id": "s-b", "incident_at": decision_time + timedelta(seconds=1)},
            {"shipment_id": "s-c", "incident_at": decision_time + timedelta(hours=6)},
            {"shipment_id": "s-d", "incident_at": decision_time + timedelta(hours=6, seconds=1)},
        ]
        rows = build_training_rows(
            [],
            labels,
            [
                ("s-a", decision_time),
                ("s-b", decision_time),
                ("s-c", decision_time),
                ("s-d", decision_time),
            ],
        )
        by_shipment = {row.shipment_id: row.label for row in rows}
        assert by_shipment["s-a"] == 0
        assert by_shipment["s-b"] == 1
        assert by_shipment["s-c"] == 1
        assert by_shipment["s-d"] == 0

    def test_label_censoring_skips_unpublished_rows(self) -> None:
        """Skip training row when label is not published yet (do not guess label=0)."""
        decision_time = datetime(2026, 12, 1, 12, 0, tzinfo=UTC)
        labels = [
            {
                "shipment_id": "s-late-label",
                "incident_at": decision_time + timedelta(hours=2),
                "label_available_at": decision_time + timedelta(days=7),
            }
        ]
        rows = build_training_rows(
            [],
            labels,
            [("s-late-label", decision_time)],
            observation_cutoff=decision_time + timedelta(hours=7),
        )
        assert rows == []

    def test_label_boundary_at_exactly_six_hours(self) -> None:
        """Incident exactly 6 hours after decision time counts as positive."""
        decision_time = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
        labels = [{"shipment_id": "s-edge", "incident_at": decision_time + timedelta(hours=6)}]
        rows = build_training_rows([], labels, [("s-edge", decision_time)])
        assert rows[0].label == 1

    def test_training_row_with_single_event(self) -> None:
        """One known event produces one training row with expected feature counts."""
        decision_time = datetime(2026, 11, 3, 10, tzinfo=UTC)
        event = make_temp_event(
            event_id="evt-only",
            device_time=datetime(2026, 11, 3, 8, tzinfo=UTC),
            received_at=datetime(2026, 11, 3, 9, tzinfo=UTC),
            value=4.5,
        )
        rows = build_training_rows([event], [], [("s-1", decision_time)])
        assert len(rows) == 1
        assert rows[0].metadata["known_event_count"] == 1
        assert rows[0].features["temp_reading_count"] == 1
        assert rows[0].features["latest_temp_c"] == 4.5

    def test_training_and_scoring_use_same_features(self, tmp_path: Path) -> None:
        """Training and RiskEngine.score produce the same feature_digest for same events."""
        artifact_dir = tmp_path / "artifact"
        decision_time = datetime(2026, 7, 1, 10, tzinfo=UTC)
        event = make_temp_event(
            event_id="evt-shared",
            device_time=datetime(2026, 7, 1, 8, tzinfo=UTC),
            received_at=datetime(2026, 7, 1, 9, tzinfo=UTC),
            value=6.5,
            source="sensor-coast",
        )
        rows = build_training_rows([event], [], [("s-1", decision_time)])
        train(rows, artifact_dir)

        engine = RiskEngine(artifact_dir, max_shipments=10)
        engine.ingest(event)
        prediction = engine.score("s-1", decision_time)
        assert prediction.feature_digest == _feature_digest(rows[0].features)


# ---------------------------------------------------------------------------
# Ingest, duplicates, revisions
# ---------------------------------------------------------------------------


class TestIngest:
    """RiskEngine.ingest: duplicates, conflicts, revisions, delivery order."""

    def test_duplicate_ingest_is_idempotent(self, artifact_dir: Path) -> None:
        """Delivering the exact same event twice returns False and counts it once."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        event = TelemetryEvent(
            event_id="evt-1",
            revision=1,
            shipment_id="s-1",
            device_time=datetime(2026, 1, 1, 8, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 8, 5, tzinfo=UTC),
            kind="temperature_c",
            value=4.0,
            source="sensor-north",
            payload={},
        )
        assert engine.ingest(event) is True
        assert engine.ingest(event) is False
        assert engine.stats()["seen_deliveries"] == 1

    def test_conflicting_duplicate_delivery_raises(self, artifact_dir: Path) -> None:
        """Same event_id+revision with different content raises ValueError."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        original = TelemetryEvent(
            event_id="evt-1",
            revision=1,
            shipment_id="s-1",
            device_time=datetime(2026, 1, 1, 8, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 8, 5, tzinfo=UTC),
            kind="temperature_c",
            value=4.0,
            source="sensor-north",
            payload={},
        )
        conflicting = TelemetryEvent(
            event_id="evt-1",
            revision=1,
            shipment_id="s-1",
            device_time=datetime(2026, 1, 1, 8, tzinfo=UTC),
            received_at=datetime(2026, 1, 1, 8, 5, tzinfo=UTC),
            kind="temperature_c",
            value=99.0,
            source="sensor-north",
            payload={},
        )
        assert engine.ingest(original) is True
        with pytest.raises(ValueError, match="conflicting redelivery"):
            engine.ingest(conflicting)

    def test_higher_revision_used_when_known_at_score_time(self, artifact_dir: Path) -> None:
        """At score time, use the highest revision already received by then."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        as_of = datetime(2026, 3, 1, 12, tzinfo=UTC)

        engine.ingest(
            make_temp_event(
                event_id="evt-temp",
                revision=1,
                device_time=datetime(2026, 3, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 3, 1, 9, tzinfo=UTC),
                value=9.0,
            )
        )
        engine.ingest(
            make_temp_event(
                event_id="evt-temp",
                revision=2,
                device_time=datetime(2026, 3, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 3, 1, 10, tzinfo=UTC),
                value=3.0,
            )
        )

        prediction = engine.score("s-1", as_of)
        assert prediction.probability < 0.5
        assert "temperature_high" not in prediction.reasons

    def test_stale_lower_revision_is_ignored(self, artifact_dir: Path) -> None:
        """A lower revision does not beat a higher revision already known at as_of."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        as_of = datetime(2026, 3, 1, 12, tzinfo=UTC)

        engine.ingest(
            make_temp_event(
                event_id="evt-temp",
                revision=2,
                device_time=datetime(2026, 3, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 3, 1, 9, tzinfo=UTC),
                value=3.0,
            )
        )
        before = engine.score("s-1", as_of).to_wire()
        # Store the lower revision too (needed for earlier as_of times), but at
        # this as_of the higher revision still wins.
        assert engine.ingest(
            make_temp_event(
                event_id="evt-temp",
                revision=1,
                device_time=datetime(2026, 3, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 3, 1, 10, tzinfo=UTC),
                value=99.0,
            )
        ) is True
        assert engine.score("s-1", as_of).to_wire() == before

    def test_correction_first_still_scores_earlier_as_of(self, artifact_dir: Path) -> None:
        """If a correction is delivered before the original, earlier as_of still uses rev1."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        early_as_of = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)
        late_as_of = datetime(2026, 3, 1, 12, tzinfo=UTC)

        engine.ingest(
            make_temp_event(
                event_id="evt-temp",
                revision=2,
                device_time=datetime(2026, 3, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 3, 1, 10, tzinfo=UTC),
                value=3.0,
            )
        )
        engine.ingest(
            make_temp_event(
                event_id="evt-temp",
                revision=1,
                device_time=datetime(2026, 3, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 3, 1, 9, tzinfo=UTC),
                value=12.0,
            )
        )

        early = engine.score("s-1", early_as_of)
        late = engine.score("s-1", late_as_of)
        assert "temperature_high" in early.reasons
        assert "temperature_high" not in late.reasons
        assert early.feature_digest != late.feature_digest

    def test_out_of_order_delivery_scoring(self, artifact_dir: Path) -> None:
        """Events can be ingested out of device_time order and still score correctly."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        as_of = datetime(2026, 4, 1, 12, tzinfo=UTC)

        engine.ingest(
            make_temp_event(
                event_id="evt-late",
                device_time=datetime(2026, 4, 1, 11, tzinfo=UTC),
                received_at=datetime(2026, 4, 1, 11, 30, tzinfo=UTC),
                value=7.0,
            )
        )
        engine.ingest(
            make_temp_event(
                event_id="evt-early",
                device_time=datetime(2026, 4, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 4, 1, 8, 30, tzinfo=UTC),
                value=4.0,
            )
        )

        prediction = engine.score("s-1", as_of)
        assert prediction.degraded is False
        assert prediction.as_of.tzinfo is not None
        assert prediction.probability >= 0.0


# ---------------------------------------------------------------------------
# Scoring & corrections
# ---------------------------------------------------------------------------


class TestScoring:
    """RiskEngine.score: time cutoff, corrections, degraded mode, timestamps."""

    def test_late_correction_does_not_change_past_score(self, artifact_dir: Path) -> None:
        """A correction that arrives later does not change scores at earlier as_of times."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        as_of = datetime(2026, 1, 1, 10, tzinfo=UTC)

        engine.ingest(
            TelemetryEvent(
                event_id="evt-1",
                revision=1,
                shipment_id="s-1",
                device_time=datetime(2026, 1, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 1, 1, 8, 5, tzinfo=UTC),
                kind="temperature_c",
                value=9.0,
                source="sensor-north",
                payload={},
            )
        )
        before = engine.score("s-1", as_of).to_wire()

        engine.ingest(
            TelemetryEvent(
                event_id="evt-1",
                revision=2,
                shipment_id="s-1",
                device_time=datetime(2026, 1, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 1, 1, 18, tzinfo=UTC),
                kind="temperature_c",
                value=3.0,
                source="sensor-north",
                payload={"correction": True},
            )
        )
        assert engine.score("s-1", as_of).to_wire() == before

    def test_score_without_events_is_degraded(self, artifact_dir: Path) -> None:
        """No events for a shipment returns degraded=True (no evidence, not definitely safe)."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        prediction = engine.score("missing-shipment", datetime(2026, 1, 1, tzinfo=UTC))
        assert prediction.degraded is True
        assert prediction.probability == 0.0
        assert prediction.reasons == ("no_events_seen",)

    def test_door_open_event_affects_features(self, artifact_dir: Path) -> None:
        """Door open events change the score (door_open_count feature)."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        as_of = datetime(2026, 9, 1, 12, tzinfo=UTC)

        engine.ingest(
            make_temp_event(
                event_id="evt-temp",
                device_time=datetime(2026, 9, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 9, 1, 8, 5, tzinfo=UTC),
                value=4.0,
            )
        )
        without_door = engine.score("s-1", as_of).to_wire()

        engine.ingest(
            TelemetryEvent(
                event_id="evt-door",
                revision=1,
                shipment_id="s-1",
                device_time=datetime(2026, 9, 1, 9, tzinfo=UTC),
                received_at=datetime(2026, 9, 1, 9, 5, tzinfo=UTC),
                kind="door_open",
                value=1.0,
                source="ops-console",
                payload={},
            )
        )
        assert engine.score("s-1", as_of).to_wire() != without_door

    def test_single_event_scoring_is_stable(self, artifact_dir: Path) -> None:
        """Scoring the same shipment at the same time twice gives identical bytes."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        event = make_temp_event(
            event_id="evt-single",
            device_time=datetime(2026, 11, 2, 8, tzinfo=UTC),
            received_at=datetime(2026, 11, 2, 8, 5, tzinfo=UTC),
            value=5.5,
        )
        as_of = datetime(2026, 11, 2, 10, tzinfo=UTC)
        engine.ingest(event)

        first = engine.score("s-1", as_of).to_wire()
        second = engine.score("s-1", as_of).to_wire()
        assert first == second

        prediction = engine.score("s-1", as_of)
        assert prediction.degraded is False
        assert prediction.feature_digest

    def test_naive_datetime_inputs_are_normalized(self, artifact_dir: Path) -> None:
        """Naive datetimes are treated as UTC on output (README requires UTC-aware)."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        engine.ingest(
            make_temp_event(
                event_id="evt-naive",
                device_time=datetime(2026, 8, 1, 8),
                received_at=datetime(2026, 8, 1, 8, 5),
                value=5.0,
            )
        )
        prediction = engine.score("s-1", datetime(2026, 8, 1, 10))
        assert prediction.as_of.tzinfo is not None

    def test_non_utc_offset_timestamps_normalized(self, artifact_dir: Path) -> None:
        """Non-UTC inputs are converted to UTC on the prediction as_of field."""
        engine = RiskEngine(artifact_dir, max_shipments=10)
        eastern = timezone(timedelta(hours=-5))
        engine.ingest(
            make_temp_event(
                event_id="evt-offset",
                device_time=datetime(2026, 11, 4, 8, 0, tzinfo=eastern),
                received_at=datetime(2026, 11, 4, 8, 5, tzinfo=eastern),
                value=4.0,
            )
        )
        prediction = engine.score("s-1", datetime(2026, 11, 4, 10, 0, tzinfo=eastern))
        assert prediction.as_of.tzinfo is not None
        assert prediction.as_of.utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Replay & determinism
# ---------------------------------------------------------------------------


class TestReplay:
    """Same event stream replayed twice must produce identical prediction bytes."""

    def test_replay_is_deterministic(self, tmp_path: Path, data_dir: Path) -> None:
        """Full ingest + score + snapshot + restore path is byte-identical across replays."""
        artifact_dir = tmp_path / "artifact"
        events = [parse_event(row) for row in load_jsonl(data_dir / "events.jsonl")]
        labels = load_jsonl(data_dir / "labels.jsonl")
        decision_times = [
            (str(row["shipment_id"]), datetime.fromisoformat(str(row["decision_time"]).replace("Z", "+00:00")))
            for row in load_jsonl(data_dir / "decision_times.jsonl")
        ]
        train(build_training_rows(events, labels, decision_times[:50]), artifact_dir)

        def run_once() -> list[bytes]:
            engine = RiskEngine(artifact_dir, max_shipments=32)
            outputs: list[bytes] = []
            for event in events[:200]:
                engine.ingest(event)
            for shipment_id, as_of in decision_times[:20]:
                outputs.append(engine.score(shipment_id, as_of).to_wire())
            snapshot = tmp_path / "snap.json"
            engine.snapshot(snapshot)
            restored = RiskEngine.restore(artifact_dir, snapshot)
            for shipment_id, as_of in decision_times[:20]:
                outputs.append(restored.score(shipment_id, as_of).to_wire())
            return outputs

        assert run_once() == run_once()

    def test_fresh_generated_stream_replay(self, tmp_path: Path, fresh_stream_dir: Path) -> None:
        """Deterministic replay on a newly generated stream (not the default data/)."""
        artifact_dir = tmp_path / "artifact"
        events = [parse_event(row) for row in load_jsonl(fresh_stream_dir / "events.jsonl")]
        labels = load_jsonl(fresh_stream_dir / "labels.jsonl")
        decision_times = [
            (str(row["shipment_id"]), datetime.fromisoformat(str(row["decision_time"]).replace("Z", "+00:00")))
            for row in load_jsonl(fresh_stream_dir / "decision_times.jsonl")[:30]
        ]
        train(build_training_rows(events, labels, decision_times), artifact_dir)

        first = replay_engine(RiskEngine(artifact_dir, max_shipments=32), events, decision_times)
        second = replay_engine(RiskEngine(artifact_dir, max_shipments=32), events, decision_times)
        assert first == second

    def test_large_replay_over_ten_thousand_events(self, tmp_path: Path, data_dir: Path) -> None:
        """Replay stays deterministic with 10k+ events (README constraint)."""
        artifact_dir = tmp_path / "artifact"
        train_full_model(artifact_dir, data_dir)
        events = [parse_event(row) for row in load_jsonl(data_dir / "events.jsonl")]
        assert len(events) > 10_000

        decision_times = [
            (str(row["shipment_id"]), datetime.fromisoformat(str(row["decision_time"]).replace("Z", "+00:00")))
            for row in load_jsonl(data_dir / "decision_times.jsonl")[:50]
        ]

        first = replay_engine(RiskEngine(artifact_dir, max_shipments=32), events, decision_times)
        second = replay_engine(RiskEngine(artifact_dir, max_shipments=32), events, decision_times)
        assert first == second

    def test_snapshot_bytes_are_deterministic(self, tmp_path: Path, data_dir: Path) -> None:
        """Snapshot JSON file bytes are identical for the same engine state."""
        artifact_dir = tmp_path / "artifact"
        events = [parse_event(row) for row in load_jsonl(data_dir / "events.jsonl")[:300]]
        decision_times = [
            (str(row["shipment_id"]), datetime.fromisoformat(str(row["decision_time"]).replace("Z", "+00:00")))
            for row in load_jsonl(data_dir / "decision_times.jsonl")[:10]
        ]
        train_minimal_model(artifact_dir, data_dir)

        def snapshot_once(path: Path) -> bytes:
            engine = RiskEngine(artifact_dir, max_shipments=32)
            replay_engine(engine, events, decision_times)
            engine.snapshot(path)
            return path.read_bytes()

        snap_a = tmp_path / "snap-a.json"
        snap_b = tmp_path / "snap-b.json"
        assert snapshot_once(snap_a) == snapshot_once(snap_b)


# ---------------------------------------------------------------------------
# Snapshots & restore
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    """snapshot() and restore() preserve engine memory across restarts."""

    def test_restore_preserves_lru_eviction_order(self, artifact_dir: Path) -> None:
        """After restore, the oldest shipment is still the first one evicted."""
        engine = RiskEngine(artifact_dir, max_shipments=2)
        for idx, shipment_id in enumerate(("s-a", "s-b")):
            engine.ingest(
                make_temp_event(
                    event_id=f"evt-{shipment_id}",
                    shipment_id=shipment_id,
                    device_time=datetime(2026, 5, 1, 8 + idx, tzinfo=UTC),
                    received_at=datetime(2026, 5, 1, 8 + idx, tzinfo=UTC),
                    value=4.0,
                )
            )
        snapshot = artifact_dir / "lru.json"
        engine.snapshot(snapshot)
        restored = RiskEngine.restore(artifact_dir, snapshot)
        restored.ingest(
            make_temp_event(
                event_id="evt-s-c",
                shipment_id="s-c",
                device_time=datetime(2026, 5, 1, 10, tzinfo=UTC),
                received_at=datetime(2026, 5, 1, 10, tzinfo=UTC),
                value=4.0,
            )
        )
        stats = restored.stats()
        assert stats["active_shipments"] == 2
        # s-a was oldest; after restore + new ingest it should be gone.
        as_of = datetime(2026, 5, 1, 12, tzinfo=UTC)
        assert restored.score("s-a", as_of).degraded is True
        assert restored.score("s-b", as_of).degraded is False
        assert restored.score("s-c", as_of).degraded is False

    def test_restore_preserves_idempotency(self, tmp_path: Path, data_dir: Path) -> None:
        """After restore, duplicate ingest of an already-seen event still returns False."""
        artifact_dir = tmp_path / "artifact"
        train_minimal_model(artifact_dir, data_dir)
        event = make_temp_event(
            event_id="evt-restore",
            device_time=datetime(2026, 5, 1, 8, tzinfo=UTC),
            received_at=datetime(2026, 5, 1, 8, 5, tzinfo=UTC),
        )

        engine = RiskEngine(artifact_dir, max_shipments=10)
        assert engine.ingest(event) is True
        snapshot = tmp_path / "snap.json"
        engine.snapshot(snapshot)

        restored = RiskEngine.restore(artifact_dir, snapshot)
        assert restored.ingest(event) is False

    def test_restored_snapshot_scores_after_model_reload(self, tmp_path: Path, data_dir: Path) -> None:
        """Restored engine can reload model and still score and dedupe events."""
        artifact_a = tmp_path / "artifact-a"
        artifact_b = tmp_path / "artifact-b"
        train_minimal_model(artifact_a, data_dir)
        shutil.copytree(artifact_a, artifact_b)

        bundle = joblib.load(artifact_b / "model.joblib")
        bundle["model_version"] = "logreg-v2"
        joblib.dump(bundle, artifact_b / "model.joblib")

        event = make_temp_event(
            event_id="evt-snap-reload",
            device_time=datetime(2026, 11, 1, 8, tzinfo=UTC),
            received_at=datetime(2026, 11, 1, 8, 5, tzinfo=UTC),
            value=6.0,
        )
        as_of = datetime(2026, 11, 1, 12, tzinfo=UTC)

        engine = RiskEngine(artifact_a, max_shipments=10)
        engine.ingest(event)
        snapshot = tmp_path / "snap.json"
        engine.snapshot(snapshot)

        restored = RiskEngine.restore(artifact_a, snapshot)
        assert restored.stats()["model_version"] != "logreg-v2"
        assert restored.reload_model(artifact_b) is True
        assert restored.stats()["model_version"] == "logreg-v2"

        prediction = restored.score("s-1", as_of)
        assert prediction.degraded is False
        assert prediction.probability >= 0.0
        assert restored.ingest(event) is False


# ---------------------------------------------------------------------------
# Model reload
# ---------------------------------------------------------------------------


class TestModelReload:
    """reload_model(): swap weights on success, keep old model on failure."""

    def test_reload_failure_keeps_old_model(self, tmp_path: Path) -> None:
        """Bad artifact dir returns False and leaves the previous model serving."""
        artifact_dir = tmp_path / "artifact"
        train_minimal_model(artifact_dir)
        engine = RiskEngine(artifact_dir, max_shipments=10)
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        original_version = engine.stats()["model_version"]
        assert engine.reload_model(bad_dir) is False
        assert engine.stats()["model_version"] == original_version

    def test_reload_success_swaps_model(self, tmp_path: Path, data_dir: Path) -> None:
        """Valid new model file updates model_version and increments reload_count."""
        artifact_a = tmp_path / "artifact-a"
        artifact_b = tmp_path / "artifact-b"
        train_minimal_model(artifact_a, data_dir)
        shutil.copytree(artifact_a, artifact_b)

        bundle = joblib.load(artifact_b / "model.joblib")
        bundle["model_version"] = "logreg-v2"
        joblib.dump(bundle, artifact_b / "model.joblib")

        engine = RiskEngine(artifact_a, max_shipments=10)
        original_version = engine.stats()["model_version"]
        assert engine.reload_model(artifact_b) is True
        assert engine.stats()["model_version"] == "logreg-v2"
        assert engine.stats()["model_version"] != original_version
        assert engine.stats()["reload_count"] == 1


# ---------------------------------------------------------------------------
# Memory & eviction
# ---------------------------------------------------------------------------


class TestMemoryEviction:
    """max_shipments LRU cap: drop oldest shipments, bound duplicate tracking."""

    def test_max_shipments_eviction(self, artifact_dir: Path) -> None:
        """When max_shipments=2, ingesting a third shipment evicts the oldest."""
        engine = RiskEngine(artifact_dir, max_shipments=2)

        for idx in range(3):
            engine.ingest(
                TelemetryEvent(
                    event_id=f"evt-{idx}",
                    revision=1,
                    shipment_id=f"s-{idx}",
                    device_time=datetime(2026, 1, 1, idx, tzinfo=UTC),
                    received_at=datetime(2026, 1, 1, idx, 5, tzinfo=UTC),
                    kind="temperature_c",
                    value=4.0,
                    source="sensor-north",
                    payload={},
                )
            )

        assert engine.stats()["active_shipments"] == 2

    def test_max_shipments_caps_active_state_under_load(self, artifact_dir: Path) -> None:
        """Under heavy ingest, active_shipments and seen_deliveries stay bounded."""
        max_shipments = 5
        engine = RiskEngine(artifact_dir, max_shipments=max_shipments)
        deliveries = 0

        for shipment_idx in range(20):
            for event_idx in range(4):
                assert engine.ingest(
                    make_temp_event(
                        event_id=f"evt-{shipment_idx}-{event_idx}",
                        shipment_id=f"s-{shipment_idx}",
                        device_time=datetime(2026, 11, 6, event_idx % 12, tzinfo=UTC),
                        received_at=datetime(2026, 11, 6, event_idx % 12, 5, tzinfo=UTC),
                        value=4.0 + event_idx * 0.1,
                    )
                )
                deliveries += 1

        stats = engine.stats()
        assert stats["active_shipments"] == max_shipments
        assert stats["seen_deliveries"] <= max_shipments * 4
        assert stats["ingest_count"] == deliveries

    def test_evicted_shipment_can_be_reloaded_with_new_events(self, artifact_dir: Path) -> None:
        """Evicted shipment starts fresh; new events for it can still be scored."""
        engine = RiskEngine(artifact_dir, max_shipments=1)

        engine.ingest(
            make_temp_event(
                event_id="evt-old",
                shipment_id="s-old",
                device_time=datetime(2026, 10, 1, 8, tzinfo=UTC),
                received_at=datetime(2026, 10, 1, 8, 5, tzinfo=UTC),
            )
        )
        engine.ingest(
            make_temp_event(
                event_id="evt-new",
                shipment_id="s-new",
                device_time=datetime(2026, 10, 1, 9, tzinfo=UTC),
                received_at=datetime(2026, 10, 1, 9, 5, tzinfo=UTC),
            )
        )
        assert engine.stats()["active_shipments"] == 1

        engine.ingest(
            make_temp_event(
                event_id="evt-old-return",
                shipment_id="s-old",
                device_time=datetime(2026, 10, 1, 10, tzinfo=UTC),
                received_at=datetime(2026, 10, 1, 10, 5, tzinfo=UTC),
            )
        )
        assert engine.stats()["active_shipments"] == 1
        prediction = engine.score("s-old", datetime(2026, 10, 1, 12, tzinfo=UTC))
        assert prediction.degraded is False


# ---------------------------------------------------------------------------
# Concurrency, scale, artifacts
# ---------------------------------------------------------------------------


class TestOps:
    """Production ops: thread safety and train() artifact completeness."""

    def test_concurrent_ingest_and_score(self, artifact_dir: Path) -> None:
        """Eight threads ingesting and scoring concurrently do not raise errors."""
        engine = RiskEngine(artifact_dir, max_shipments=32)
        errors: list[Exception] = []

        def worker(offset: int) -> None:
            try:
                for idx in range(40):
                    event = make_temp_event(
                        event_id=f"evt-{offset}-{idx}",
                        shipment_id=f"s-{offset}",
                        device_time=datetime(2026, 6, 1, idx % 12, tzinfo=UTC),
                        received_at=datetime(2026, 6, 1, idx % 12, 5, tzinfo=UTC),
                        value=4.0 + idx * 0.01,
                    )
                    engine.ingest(event)
                    engine.score(f"s-{offset}", datetime(2026, 6, 1, 12, tzinfo=UTC))
            except Exception as exc:  # pragma: no cover - surfaced via errors list
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(offset,)) for offset in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        assert engine.stats()["active_shipments"] <= 32

    def test_artifact_is_self_contained(self, tmp_path: Path, data_dir: Path) -> None:
        """train() writes model.joblib and metrics.json with all README-required fields."""
        artifact_dir = tmp_path / "artifact"
        train_minimal_model(artifact_dir, data_dir)
        assert (artifact_dir / "model.joblib").exists()
        assert (artifact_dir / "metrics.json").exists()
        metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        assert "pr_auc" in metrics
        assert "brier_score" in metrics
        assert "baseline_pr_auc" in metrics
        assert "model_comparison" in metrics
        assert "selected_model" in metrics
        assert "slices" in metrics
        assert len(metrics["slices"]) >= 2
        assert all(slice_row["rows"] > 0 for slice_row in metrics["slices"])
        assert all("pr_auc" in slice_row for slice_row in metrics["slices"])
