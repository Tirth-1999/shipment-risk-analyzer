"""Train a risk model and score live shipment events.

Training flow:
    build_training_rows(events, labels, decision_times) -> list of TrainingRow
    train(rows, artifact_dir) -> saves model.joblib and metrics.json

Live scoring flow:
    RiskEngine(artifact_dir) -> ingest events -> score(shipment_id, as_of)

Rules:
    Use received_at (not device_time) to decide what data was known when.
    Skip training rows when the 6 hour outcome is not known yet.
    Same event history must always produce the same score bytes.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import Prediction, TelemetryEvent, TrainingRow

# Prediction / label window: incident in (decision_time, decision_time + HORIZON].
HORIZON = timedelta(hours=6)
MODEL_VERSION_SUFFIX = "v1"
MODEL_CANDIDATES = ("logreg", "hgb")  # linear baseline vs non-linear tabular model
FEATURE_NAMES = (  # shared by training rows and RiskEngine.score()
    "latest_temp_c",
    "mean_temp_c",
    "max_temp_c",
    "temp_slope_c_per_h",
    "temp_reading_count",
    "hours_since_first_reading",
    "door_open_count",
)


def _utc(value: datetime | str) -> datetime:
    """Convert a datetime or ISO string to timezone-aware UTC.

    Args:
        value: Datetime or ISO-8601 string (Z suffix allowed).

    Returns:
        UTC datetime with tzinfo set.
    """
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _events_known_at(events: Iterable[TelemetryEvent], as_of: datetime) -> list[TelemetryEvent]:
    """Return events the system had received by the score time.

    Keeps the highest revision per event_id. Sorts by device_time for features.

    Args:
        events: All events for one shipment.
        as_of: Score time; events with received_at after this are dropped.

    Returns:
        List of known events, sorted by device_time then event_id.
    """
    as_of = _utc(as_of)
    best: dict[str, TelemetryEvent] = {}
    for event in events:
        if _utc(event.received_at) > as_of:
            continue
        current = best.get(event.event_id)
        if current is None or event.revision > current.revision:
            best[event.event_id] = event
    return sorted(best.values(), key=lambda item: (_utc(item.device_time), item.event_id))


def _build_features(events: Sequence[TelemetryEvent]) -> dict[str, float | int | str | None]:
    """Build the seven model features from known events.

    Same function is used in training and live scoring.

    Args:
        events: Events already filtered to what was known at decision time.

    Returns:
        Dict with latest_temp_c, mean_temp_c, max_temp_c, temp_slope_c_per_h,
        temp_reading_count, hours_since_first_reading, and door_open_count.
    """
    temps = [
        float(event.value)
        for event in events
        if event.kind == "temperature_c" and isinstance(event.value, (int, float))
    ]
    door_opens = sum(
        1
        for event in events
        if event.kind == "door_open" and event.value not in (0, 0.0, "0", None, False)
    )
    first_time = _utc(events[0].device_time) if events else None
    last_time = _utc(events[-1].device_time) if events else None

    latest_temp = temps[-1] if temps else 0.0
    mean_temp = float(np.mean(temps)) if temps else 0.0
    max_temp = float(max(temps)) if temps else 0.0
    if len(temps) >= 2 and first_time and last_time:
        hours = max((_utc(last_time) - _utc(first_time)).total_seconds() / 3600.0, 0.25)
        slope = (temps[-1] - temps[0]) / hours
    else:
        slope = 0.0

    hours_since_first = 0.0
    if first_time is not None:
        hours_since_first = max((_utc(last_time) - _utc(first_time)).total_seconds() / 3600.0, 0.0)

    return {
        "latest_temp_c": latest_temp,
        "mean_temp_c": mean_temp,
        "max_temp_c": max_temp,
        "temp_slope_c_per_h": slope,
        "temp_reading_count": len(temps),
        "hours_since_first_reading": hours_since_first,
        "door_open_count": door_opens,
    }


def _feature_vector(features: Mapping[str, float | int | str | None]) -> np.ndarray:
    """Turn a feature dict into a single row for sklearn.

    Args:
        features: Feature dict from _build_features.

    Returns:
        1 x 7 float numpy array in FEATURE_NAMES order.
    """
    return np.array([[float(features[name]) for name in FEATURE_NAMES]], dtype=float)


def _feature_digest(features: Mapping[str, float | int | str | None]) -> str:
    """Hash feature values for replay and audit checks.

    Args:
        features: Feature dict from _build_features.

    Returns:
        First 16 characters of a SHA-256 hex digest.
    """
    payload = {name: float(features[name]) for name in FEATURE_NAMES}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _label_available_at(label: Mapping[str, object]) -> datetime:
    """Return when an incident label became available in the data.

    Args:
        label: One label record from labels.jsonl.

    Returns:
        UTC datetime when the label is treated as published.
        Uses label_available_at if present, else incident_at.
    """
    if "label_available_at" in label:
        return _utc(label["label_available_at"])  # type: ignore[arg-type]
    return _utc(label["incident_at"])  # type: ignore[arg-type]


def _observation_cutoff(labels: Iterable[Mapping[str, object]], decision_times: Iterable[tuple[str, datetime]], explicit: datetime | None) -> datetime:
    """Find the latest time where training labels can be trusted.

    Args:
        labels: All incident labels.
        decision_times: All (shipment_id, decision_time) pairs for training.
        explicit: Fixed cutoff if provided; otherwise inferred from data.

    Returns:
        UTC datetime after which all 6 hour windows and labels are complete.
    """
    if explicit is not None:
        return _utc(explicit)
    candidates: list[datetime] = []
    for label in labels:
        candidates.append(_label_available_at(label))
    for _, decision_time in decision_times:
        candidates.append(_utc(decision_time) + HORIZON)
    if candidates:
        return max(candidates)
    return datetime.max.replace(tzinfo=timezone.utc)


def _incident_label(shipment_id: str, decision_time: datetime, labels_by_shipment: Mapping[str, list[Mapping[str, object]]], observation_cutoff: datetime) -> int | None:
    """Label whether an incident happens in the next 6 hours.

    Args:
        shipment_id: Shipment to label.
        decision_time: Time we are scoring at.
        labels_by_shipment: Incidents grouped by shipment id.
        observation_cutoff: Last time labels are considered complete.

    Returns:
        1 if incident in (decision_time, decision_time + 6h],
        0 if window is complete with no incident,
        None if outcome is still unknown (row should be skipped).
    """
    decision_time = _utc(decision_time)
    observation_cutoff = _utc(observation_cutoff)
    if decision_time + HORIZON > observation_cutoff:
        return None
    window_end = decision_time + HORIZON
    for label in labels_by_shipment.get(shipment_id, []):
        incident_at = _utc(label["incident_at"])  # type: ignore[arg-type]
        if decision_time < incident_at <= window_end:
            if _label_available_at(label) <= observation_cutoff:
                return 1
            return None
    return 0


def _prediction_reasons(features: Mapping[str, float | int | str | None], probability: float) -> tuple[str, ...]:
    """Build short reason tags for a prediction.

    Args:
        features: Feature dict from _build_features.
        probability: Model score from 0 to 1.

    Returns:
        Tuple of tags like temperature_high or door_open_seen.
    """
    reasons: list[str] = []
    if float(features["latest_temp_c"]) >= 8.0:
        reasons.append("temperature_high")
    if float(features["temp_slope_c_per_h"]) >= 0.5:
        reasons.append("warming_trend")
    if int(features["door_open_count"]) > 0:
        reasons.append("door_open_seen")
    if probability >= 0.5:
        reasons.append("model_high_risk")
    return tuple(reasons)


def build_training_rows(events: Iterable[TelemetryEvent], labels: Iterable[Mapping[str, object]], decision_times: Iterable[tuple[str, datetime]], *, observation_cutoff: datetime | None = None) -> list[TrainingRow]:
    """Build training examples at each decision time.

    Uses only events received on or before decision_time.
    Skips rows where the 6 hour outcome is not known yet.

    Args:
        events: Telemetry events (any delivery order).
        labels: Incident records with incident_at and label_available_at.
        decision_times: List of (shipment_id, decision_time) to build rows for.
        observation_cutoff: Optional label cutoff; inferred from data if omitted.

    Returns:
        Sorted list of TrainingRow. Unknown-outcome rows are omitted.
    """
    label_list = list(labels)
    decision_list = list(decision_times)
    cutoff = _observation_cutoff(label_list, decision_list, observation_cutoff)

    events_by_shipment: dict[str, list[TelemetryEvent]] = {}
    for event in events:
        events_by_shipment.setdefault(event.shipment_id, []).append(event)

    labels_by_shipment: dict[str, list[Mapping[str, object]]] = {}
    for label in label_list:
        labels_by_shipment.setdefault(str(label["shipment_id"]), []).append(label)

    rows: list[TrainingRow] = []
    for shipment_id, decision_time in decision_list:
        decision_time = _utc(decision_time)
        label = _incident_label(shipment_id, decision_time, labels_by_shipment, cutoff)
        if label is None:
            continue
        known_events = _events_known_at(events_by_shipment.get(shipment_id, []), decision_time)
        features = _build_features(known_events)
        rows.append(
            TrainingRow(
                shipment_id=shipment_id,
                decision_time=decision_time,
                features=features,
                label=label,
                metadata={"known_event_count": len(known_events)},
            )
        )
    rows.sort(key=lambda row: (row.shipment_id, row.decision_time))
    return rows


def _split_by_shipment(rows: Sequence[TrainingRow]) -> tuple[list[TrainingRow], list[TrainingRow]]:
    """Split 80% train / 20% test by shipment start time.

    Earlier shipments go to train, later ones to test.
    All rows from one shipment stay on the same side.

    Args:
        rows: All training rows.

    Returns:
        (train_rows, test_rows) tuple.
    """
    first_decision: dict[str, datetime] = {}
    for row in rows:
        current = first_decision.get(row.shipment_id)
        if current is None or row.decision_time < current:
            first_decision[row.shipment_id] = row.decision_time
    shipment_ids = sorted(first_decision, key=lambda shipment_id: first_decision[shipment_id])
    if len(shipment_ids) <= 1:
        return list(rows), []
    split_at = max(1, int(len(shipment_ids) * 0.8))
    train_ids = set(shipment_ids[:split_at])
    train_rows = [row for row in rows if row.shipment_id in train_ids]
    test_rows = [row for row in rows if row.shipment_id not in train_ids]
    return train_rows, test_rows


def _rows_to_xy(rows: Sequence[TrainingRow]) -> tuple[np.ndarray, np.ndarray]:
    """Convert training rows to feature matrix and label array.

    Args:
        rows: Training rows with features and labels.

    Returns:
        (x, y) tuple for sklearn fit/predict.
    """
    x = np.vstack([_feature_vector(row.features) for row in rows])
    y = np.array([row.label for row in rows], dtype=int)
    return x, y


def _positive_class_probabilities(model: Pipeline, features: np.ndarray) -> list[float]:
    """Get incident probability from a fitted model.

    Args:
        model: Fitted sklearn pipeline.
        features: Feature matrix (N rows x 7 columns).

    Returns:
        List of probabilities. All zeros if training had no positive labels.
    """
    probabilities = model.predict_proba(features)
    classifier = model.named_steps["clf"]
    classes = list(getattr(classifier, "classes_", [0]))
    if 1 not in classes:
        return [0.0 for _ in range(len(probabilities))]
    positive_index = classes.index(1)
    return probabilities[:, positive_index].tolist()


def _make_candidate_model(name: str, y_train: np.ndarray) -> Pipeline:
    """Create one model candidate before training.

    Args:
        name: "logreg" for logistic regression or "hgb" for gradient boosting.
        y_train: Training labels used to detect single-class edge case.

    Returns:
        Unfitted sklearn Pipeline.
    """
    if len(set(y_train.tolist())) < 2:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", DummyClassifier(strategy="prior")),
            ]
        )
    if name == "logreg":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=1729)),
            ]
        )
    return Pipeline(
        steps=[
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=150,
                    learning_rate=0.08,
                    max_depth=4,
                    random_state=1729,
                ),
            ),
        ]
    )


def _safe_pr_auc(y_true: np.ndarray, y_score: Sequence[float]) -> float:
    """Compute PR-AUC without warnings when there are no positives.

    Args:
        y_true: True binary labels.
        y_score: Predicted scores.

    Returns:
        PR-AUC float, or 0.0 when y_true has no positive labels.
    """
    labels = np.asarray(y_true, dtype=int)
    if labels.size == 0 or labels.sum() == 0:
        return 0.0
    return float(average_precision_score(labels, y_score))


def _evaluate_split(model: Pipeline, x_eval: np.ndarray, y_eval: np.ndarray, baseline_prob: float) -> dict[str, float]:
    """Score one model on a held-out set.

    Args:
        model: Fitted pipeline.
        x_eval: Feature matrix for evaluation.
        y_eval: True labels.
        baseline_prob: Constant baseline (mean train label rate).

    Returns:
        Dict with pr_auc, brier_score, and matching baseline metrics.
    """
    probabilities = _positive_class_probabilities(model, x_eval)
    baseline_probs = [baseline_prob] * len(y_eval)
    return {
        "pr_auc": _safe_pr_auc(y_eval, probabilities),
        "brier_score": float(brier_score_loss(y_eval, probabilities)),
        "baseline_pr_auc": _safe_pr_auc(y_eval, baseline_probs),
        "baseline_brier_score": float(brier_score_loss(y_eval, baseline_probs)),
    }


def _slice_metrics(model: Pipeline, rows: Sequence[TrainingRow], baseline_prob: float, *, name: str, predicate: Callable[[TrainingRow], bool]) -> dict[str, object]:
    """Metrics for one subset of held-out rows.

    Args:
        model: Fitted pipeline.
        rows: Held-out training rows.
        baseline_prob: Constant baseline rate.
        name: Slice name for the report.
        predicate: Returns True for rows in this slice.

    Returns:
        Dict with name, row count, and metrics (or just name and rows=0 if empty).
    """
    subset = [row for row in rows if predicate(row)]
    if not subset:
        return {"name": name, "rows": 0}
    x_eval, y_eval = _rows_to_xy(subset)
    metrics = _evaluate_split(model, x_eval, y_eval, baseline_prob)
    return {"name": name, "rows": len(subset), **metrics}


def _select_model(comparison: Mapping[str, Mapping[str, float]]) -> str:
    """Pick the best model from the candidate comparison.

    Args:
        comparison: Per-model metrics from _evaluate_split.

    Returns:
        Winning candidate name ("logreg" or "hgb").
        Higher PR-AUC wins; tie goes to lower Brier score.
    """
    return max(
        MODEL_CANDIDATES,
        key=lambda name: (
            comparison[name]["pr_auc"],
            -comparison[name]["brier_score"],
        ),
    )


def train(rows: Sequence[TrainingRow], artifact_dir: Path) -> Mapping[str, object]:
    """Train models, pick the best one, and save files to artifact_dir.

    Trains logistic regression and gradient boosting, compares on held-out
    shipments, writes model.joblib and metrics.json.

    Args:
        rows: Training rows from build_training_rows.
        artifact_dir: Output folder for model.joblib and metrics.json.

    Returns:
        Metrics dict (same content written to metrics.json).
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    train_rows, test_rows = _split_by_shipment(rows)
    x_train, y_train = _rows_to_xy(train_rows)
    eval_rows = test_rows if test_rows else train_rows
    x_eval, y_eval = _rows_to_xy(eval_rows)

    baseline_prob = float(np.mean(y_train))
    fitted: dict[str, Pipeline] = {}
    comparison: dict[str, dict[str, float]] = {}

    for name in MODEL_CANDIDATES:
        candidate = _make_candidate_model(name, y_train)
        candidate.fit(x_train, y_train)
        fitted[name] = candidate
        comparison[name] = _evaluate_split(candidate, x_eval, y_eval, baseline_prob)

    selected = _select_model(comparison)
    model = fitted[selected]
    selected_metrics = comparison[selected]
    model_version = f"{selected}-{MODEL_VERSION_SUFFIX}"

    eval_rows = test_rows if test_rows else train_rows
    # Early vs late in the trip: both fire on this generator's decision schedule.
    slice_metrics = [
        _slice_metrics(
            model,
            eval_rows,
            baseline_prob,
            name="early_window",
            predicate=lambda row: float(row.features["hours_since_first_reading"]) < 10.0,
        ),
        _slice_metrics(
            model,
            eval_rows,
            baseline_prob,
            name="late_window",
            predicate=lambda row: float(row.features["hours_since_first_reading"]) >= 10.0,
        ),
    ]

    metrics: dict[str, object] = {
        "evaluation_split": "80/20 by first decision time (later shipments held out)",
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "selected_model": selected,
        "model_comparison": comparison,
        "pr_auc": selected_metrics["pr_auc"],
        "brier_score": selected_metrics["brier_score"],
        "baseline_pr_auc": selected_metrics["baseline_pr_auc"],
        "baseline_brier_score": selected_metrics["baseline_brier_score"],
        "slices": slice_metrics,
    }

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "model_version": model_version,
            "selected_model": selected,
            "metrics": metrics,
        },
        artifact_dir / "model.joblib",
    )
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _load_model_bundle(artifact_dir: Path) -> dict[str, object]:
    """Load model.joblib and check required fields.

    Args:
        artifact_dir: Folder containing model.joblib from train().

    Returns:
        Dict with model, feature_names, model_version, and related fields.

    Raises:
        ValueError: If required fields are missing from the file.
    """
    path = artifact_dir / "model.joblib"
    bundle = joblib.load(path)
    required = {"model", "feature_names", "model_version"}
    if not required.issubset(bundle):
        raise ValueError("artifact missing required fields")
    return bundle


def _event_to_dict(event: TelemetryEvent) -> dict[str, object]:
    """Convert an event to a JSON-friendly dict for snapshots.

    Args:
        event: TelemetryEvent to serialize.

    Returns:
        Dict with ISO datetime strings and all event fields.
    """
    return {
        "device_time": _utc(event.device_time).isoformat(),
        "event_id": event.event_id,
        "kind": event.kind,
        "payload": dict(event.payload),
        "received_at": _utc(event.received_at).isoformat(),
        "revision": event.revision,
        "shipment_id": event.shipment_id,
        "source": event.source,
        "value": event.value,
    }


def _delivery_fingerprint(event: TelemetryEvent) -> str:
    """Hash full event content to detect conflicting duplicates.

    Args:
        event: TelemetryEvent to hash.

    Returns:
        SHA-256 hex digest of the event JSON.
    """
    raw = json.dumps(_event_to_dict(event), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _event_from_dict(raw: Mapping[str, object]) -> TelemetryEvent:
    """Rebuild a TelemetryEvent from snapshot JSON.

    Args:
        raw: Event dict from a snapshot file.

    Returns:
        TelemetryEvent instance.
    """
    return TelemetryEvent(
        event_id=str(raw["event_id"]),
        revision=int(raw["revision"]),
        shipment_id=str(raw["shipment_id"]),
        device_time=_utc(datetime.fromisoformat(str(raw["device_time"]).replace("Z", "+00:00"))),
        received_at=_utc(datetime.fromisoformat(str(raw["received_at"]).replace("Z", "+00:00"))),
        kind=str(raw["kind"]),
        value=raw["value"],  # type: ignore[assignment]
        source=str(raw["source"]),
        payload=dict(raw["payload"]),  # type: ignore[arg-type]
    )


class _ShipmentState:
    """In-memory events for one shipment, keyed by event_id and revision."""

    __slots__ = ("events",)

    def __init__(self) -> None:
        self.events: dict[str, dict[int, TelemetryEvent]] = {}


class RiskEngine:
    """Score live shipment events using a trained model.

    Call ingest() to add events, score() to get a prediction.
    Use snapshot() and restore() to save and reload state.
    Use reload_model() to swap model files without clearing event memory.

    Thread-safe. Drops oldest shipments when max_shipments is exceeded.
    Duplicate events are ignored; conflicting duplicates raise ValueError.
    Failed model reload keeps the previous model running.
    """

    def __init__(self, artifact_dir: Path, max_shipments: int = 10_000):
        """Create a scorer from a saved model artifact.

        Args:
            artifact_dir: Path to folder with model.joblib from train().
            max_shipments: Max shipments kept in memory (oldest dropped first).

        Returns:
            None. Initializes empty event memory and loads the model.
        """
        self._lock = threading.RLock()
        self._max_shipments = max_shipments
        self._seen_deliveries: set[tuple[str, int]] = set()
        self._delivery_fingerprints: dict[tuple[str, int], str] = {}
        self._shipments: OrderedDict[str, _ShipmentState] = OrderedDict()
        self._ingest_count = 0
        self._score_count = 0
        self._reload_count = 0
        self._artifact_dir = artifact_dir
        self._model_bundle = _load_model_bundle(artifact_dir)
        self._model = self._model_bundle["model"]
        self._model_version = str(self._model_bundle["model_version"])

    def ingest(self, event: TelemetryEvent) -> bool:
        """Add one event from the stream.

        Args:
            event: Delivered telemetry event.

        Returns:
            True if shipment state changed, False if duplicate or stale.

        Raises:
            ValueError: Same event_id and revision with different content.
        """
        delivery_key = (event.event_id, event.revision)
        fingerprint = _delivery_fingerprint(event)
        with self._lock:
            if delivery_key in self._seen_deliveries:
                if self._delivery_fingerprints.get(delivery_key) != fingerprint:
                    raise ValueError(
                        f"conflicting redelivery for {event.event_id} revision {event.revision}"
                    )
                return False
            self._seen_deliveries.add(delivery_key)
            self._delivery_fingerprints[delivery_key] = fingerprint
            self._ingest_count += 1

            state = self._shipments.get(event.shipment_id)
            if state is None:
                state = _ShipmentState()
                self._shipments[event.shipment_id] = state
            self._shipments.move_to_end(event.shipment_id)

            revisions = state.events.setdefault(event.event_id, {})
            if event.revision in revisions:
                return False

            # Keep every revision. score() picks the highest one known at as_of.
            revisions[event.revision] = event
            self._enforce_capacity()
            return True

    def score(self, shipment_id: str, as_of: datetime) -> Prediction:
        """Estimate incident risk in the next 6 hours at a given time.

        Only uses events with received_at on or before as_of.

        Args:
            shipment_id: Shipment to score.
            as_of: UTC time for the prediction.

        Returns:
            Prediction with probability, reasons, and degraded flag.
            Probability is 0 with degraded=True when no events are known yet.
        """
        as_of = _utc(as_of)
        with self._lock:
            self._score_count += 1
            if shipment_id in self._shipments:
                self._shipments.move_to_end(shipment_id)

            state = self._shipments.get(shipment_id)
            events: list[TelemetryEvent] = []
            if state:
                for revisions in state.events.values():
                    events.extend(revisions.values())
            known_events = _events_known_at(events, as_of)
            features = _build_features(known_events)

            degraded = False
            reasons: tuple[str, ...] = ("no_events_seen",)
            if known_events:
                vector = _feature_vector(features)
                probability = float(_positive_class_probabilities(self._model, vector)[0])
                reasons = _prediction_reasons(features, probability)
            else:
                degraded = True
                probability = 0.0
                reasons = ("no_events_seen",)

            return Prediction(
                shipment_id=shipment_id,
                as_of=as_of,
                probability=probability,
                model_version=self._model_version,
                feature_digest=_feature_digest(features),
                degraded=degraded,
                reasons=reasons,
            )

    def snapshot(self, destination: Path) -> None:
        """Save engine state to a JSON file.

        Writes to a .tmp file first, then renames for atomic save.

        Args:
            destination: Path for the snapshot JSON file.

        Returns:
            None.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_suffix(".tmp")
        with self._lock:
            payload = {
                "ingest_count": self._ingest_count,
                "max_shipments": self._max_shipments,
                "model_version": self._model_version,
                "score_count": self._score_count,
                "seen_deliveries": sorted(
                    [{"event_id": key[0], "revision": key[1]} for key in self._seen_deliveries],
                    key=lambda item: (item["event_id"], item["revision"]),
                ),
                # List preserves LRU order (oldest first) for correct restore eviction.
                "shipments": [
                    {
                        "shipment_id": shipment_id,
                        "events": sorted(
                            [
                                _event_to_dict(event)
                                for revisions in state.events.values()
                                for event in revisions.values()
                            ],
                            key=lambda item: (item["event_id"], item["revision"]),
                        ),
                    }
                    for shipment_id, state in self._shipments.items()
                ],
            }
        tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        tmp.replace(destination)

    @classmethod
    def restore(cls, artifact_dir: Path, snapshot: Path) -> RiskEngine:
        """Load engine state from a snapshot file.

        Args:
            artifact_dir: Folder with model.joblib.
            snapshot: Path to snapshot JSON from snapshot().

        Returns:
            RiskEngine restored to the saved state.
        """
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        engine = cls(artifact_dir, max_shipments=int(payload["max_shipments"]))
        with engine._lock:
            engine._ingest_count = int(payload["ingest_count"])
            engine._score_count = int(payload["score_count"])
            engine._seen_deliveries = {
                (item["event_id"], int(item["revision"])) for item in payload["seen_deliveries"]
            }
            engine._delivery_fingerprints = {}
            engine._shipments = OrderedDict()
            shipments_payload = payload["shipments"]
            # New snapshots are an LRU-ordered list; old dict format still loads.
            if isinstance(shipments_payload, dict):
                shipment_items = list(shipments_payload.items())
            else:
                shipment_items = [
                    (item["shipment_id"], item) for item in shipments_payload
                ]
            for shipment_id, shipment_payload in shipment_items:
                state = _ShipmentState()
                for raw in shipment_payload["events"]:
                    event = _event_from_dict(raw)
                    state.events.setdefault(event.event_id, {})[event.revision] = event
                    key = (event.event_id, event.revision)
                    engine._delivery_fingerprints[key] = _delivery_fingerprint(event)
                engine._shipments[shipment_id] = state
        return engine

    def reload_model(self, artifact_dir: Path) -> bool:
        """Load a new model file without clearing event memory.

        Args:
            artifact_dir: Folder with the new model.joblib to try.

        Returns:
            True if swap succeeded, False if file is bad (old model kept).
        """
        try:
            candidate = _load_model_bundle(artifact_dir)
        except (OSError, ValueError, KeyError):
            return False
        with self._lock:
            self._model_bundle = candidate
            self._model = candidate["model"]
            self._model_version = str(candidate["model_version"])
            self._artifact_dir = artifact_dir
            self._reload_count += 1
        return True

    def stats(self) -> Mapping[str, int | float | str]:
        """Return counters for debugging and monitoring.

        Returns:
            Dict with ingest_count, score_count, active_shipments,
            model_version, max_shipments, reload_count, and seen_deliveries.
        """
        with self._lock:
            return {
                "active_shipments": len(self._shipments),
                "ingest_count": self._ingest_count,
                "max_shipments": self._max_shipments,
                "model_version": self._model_version,
                "reload_count": self._reload_count,
                "score_count": self._score_count,
                "seen_deliveries": len(self._seen_deliveries),
            }

    def _drop_seen_deliveries(self, state: _ShipmentState) -> None:
        """Remove duplicate-tracking keys for an evicted shipment.

        Args:
            state: Shipment state being removed from memory.

        Returns:
            None.
        """
        for event_id, revisions in state.events.items():
            for revision in revisions:
                key = (event_id, revision)
                self._seen_deliveries.discard(key)
                self._delivery_fingerprints.pop(key, None)

    def _enforce_capacity(self) -> None:
        """Drop oldest shipments until count is within max_shipments.

        Returns:
            None.
        """
        while len(self._shipments) > self._max_shipments:
            _, evicted = self._shipments.popitem(last=False)
            self._drop_seen_deliveries(evicted)
