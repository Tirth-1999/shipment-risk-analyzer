"""Data types shared by training, scoring, and tests.

TelemetryEvent: one message from a shipment sensor.
TrainingRow: one training example at a decision time.
Prediction: one risk score returned by RiskEngine.

Important times on TelemetryEvent:
    device_time: when the sensor says the reading happened.
    received_at: when our system received it (used for scoring cutoff).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """One telemetry message for a shipment.

    The same event_id can have multiple revisions when data is corrected.

    Attributes:
        event_id: Stable id for this logical event.
        revision: Version number for corrections to event_id.
        shipment_id: Which shipment this belongs to.
        device_time: Sensor clock time (used to order readings).
        received_at: When the platform received this revision.
        kind: Event type, e.g. temperature_c or door_open.
        value: Reading value for this kind.
        source: Sensor or system name.
        payload: Extra metadata (not used as model input).
    """

    event_id: str
    revision: int
    shipment_id: str
    device_time: datetime
    received_at: datetime
    kind: str
    value: float | str | None
    source: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingRow:
    """One labeled example used to train the model.

    Features use only events received on or before decision_time.
    Incident labels are never copied into features.

    Attributes:
        shipment_id: Shipment identifier.
        decision_time: UTC time we pretend to score at.
        features: Seven numeric features (same as live scoring).
        label: 1 if incident in next 6 hours, else 0.
        metadata: Extra info for debugging (e.g. known_event_count).
    """

    shipment_id: str
    decision_time: datetime
    features: Mapping[str, float | int | str | None]
    label: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Prediction:
    """Risk score for one shipment at one point in time.

    Attributes:
        shipment_id: Shipment that was scored.
        as_of: UTC time used for the score.
        probability: Risk from 0 to 1 (0 when degraded with no events).
        model_version: Which model file was used, e.g. logreg-v1.
        feature_digest: Short hash of features for replay checks.
        degraded: True when no events were known (not a model load error).
        reasons: Short tags explaining the score, e.g. temperature_high.
    """

    shipment_id: str
    as_of: datetime
    probability: float
    model_version: str
    feature_digest: str
    degraded: bool
    reasons: tuple[str, ...]

    def to_wire(self) -> bytes:
        """Serialize this prediction to stable JSON bytes.

        Returns:
            UTF-8 bytes with sorted keys (same input always gives same bytes).
        """
        import json

        value = {
            "as_of": self.as_of.isoformat(),
            "degraded": self.degraded,
            "feature_digest": self.feature_digest,
            "model_version": self.model_version,
            "probability": self.probability,
            "reasons": list(self.reasons),
            "shipment_id": self.shipment_id,
        }
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
