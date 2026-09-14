"""Refrigerated shipment risk scoring package.

Public functions:
    build_training_rows: build training examples from events and labels
    train: fit a model and write artifact files
    RiskEngine: ingest live events and score shipments

See contracts.py for data types and DECISIONS.md for design notes.
"""

from .contracts import Prediction, TelemetryEvent, TrainingRow
from .solution import RiskEngine, build_training_rows, train

__all__ = [
    "Prediction",
    "RiskEngine",
    "TelemetryEvent",
    "TrainingRow",
    "build_training_rows",
    "train",
]
