"""Optional Streamlit demo (not graded): train, metrics, and five walkthrough shipments.

Tabs: Training metrics | Inference | Event timeline | Late correction.
Uses the same ``build_training_rows``, ``train``, and ``RiskEngine`` as production code.
Run from ``optional/``: ``streamlit run app.py``
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay

OPTIONAL_DIR = Path(__file__).resolve().parent
ROOT = OPTIONAL_DIR.parent
sys.path[:0] = [str(ROOT / "src"), str(OPTIONAL_DIR)]

from dispatch_risk.solution import RiskEngine, build_training_rows, train  # noqa: E402
from demo_helpers import (  # noqa: E402
    WALKTHROUGH_DIR,
    events_for_shipment,
    holdout_evaluation,
    incident_time_for_shipment,
    label_in_next_6h,
    load_decision_times,
    load_events,
    load_labels,
    walkthrough_shipment_options,
    write_walkthrough_dataset,
)

ARTIFACT_DIR = ROOT / "artifact"
DATA_DIR = ROOT / "data"

SHIPMENT_COLORS = {
    "s-demo-stable": "#2563eb",
    "s-demo-warming": "#dc2626",
    "s-demo-correction": "#9333ea",
    "s-demo-door": "#ea580c",
    "s-demo-delayed": "#0891b2",
}
TIMELINE_LEGEND = [
    ("Temperature reading", "o", "#64748b"),
    ("Duplicate delivery", "o", "#64748b"),
    ("Late correction", "D", "#9333ea"),
    ("Door open", "v", "#ea580c"),
]

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
    }
)

APP_CSS = """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2.5rem; max-width: 1100px; }
    .hero {
        background: linear-gradient(135deg, #eef4fb 0%, #f8fbff 100%);
        border: 1px solid #dbeafe; border-radius: 12px;
        padding: 1rem 1.25rem; margin-bottom: 1rem;
    }
    .hero h1 { font-size: 1.4rem; font-weight: 700; color: #1c2430; margin: 0 0 0.3rem 0; }
    .hero p { font-size: 0.95rem; color: #4a5568; margin: 0; line-height: 1.45; }
    .banner-ok {
        background: #ecfdf5; border: 1px solid #6ee7b7; border-radius: 10px;
        padding: 0.85rem 1rem; margin: 0; color: #065f46; font-size: 0.95rem;
    }
    div[data-testid="stVerticalBlock"]:has(.banner-ok) + div[data-testid="stVerticalBlock"] {
        margin-top: 0.35rem;
    }
    .use-case-box {
        background: #fffbeb; border: 1px solid #fcd34d; border-radius: 10px;
        padding: 0.85rem 1rem; margin-bottom: 1rem; color: #78350f; font-size: 0.92rem;
    }

    /* Sidebar: card layout, no scroll */
    [data-testid="stSidebar"] {
        background: #f8fafc; border-right: 1px solid #e2e8f0;
        overflow: hidden !important;
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    [data-testid="stSidebar"] ::-webkit-scrollbar { display: none; width: 0; height: 0; }
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {
        overflow: hidden !important;
        overflow-y: hidden !important;
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem; padding-bottom: 1rem;
        overflow: visible;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
        padding: 1rem 1rem 1.1rem 1rem !important;
        margin-bottom: 0.85rem !important;
        background: #ffffff;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px;
        box-sizing: border-box;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stAlert"] {
        margin: 0.5rem 0 0 0 !important;
        padding: 0.6rem 0.75rem !important;
        font-size: 0.84rem;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stAlert"] p {
        font-size: 0.84rem; line-height: 1.45; margin: 0;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]:last-child {
        margin-bottom: 0 !important;
    }
    [data-testid="stSidebar"] .sidebar-section {
        font-size: 0.82rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.05em; color: #334155;
        margin: 0 0 0.35rem 0; padding-bottom: 0.5rem;
        border-bottom: 1px solid #eef2f6;
        width: 100%; max-width: 100%; box-sizing: border-box;
    }
    [data-testid="stSidebar"] .sidebar-desc {
        font-size: 0.82rem; color: #64748b; line-height: 1.4;
        margin: 0 0 0.65rem 0;
        width: 100%; max-width: 100%; box-sizing: border-box;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
        gap: 0.75rem !important;
    }
    [data-testid="stSidebar"] .stButton button {
        min-height: 2.5rem; font-size: 0.92rem;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] label,
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        font-size: 0.88rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        margin-top: 0.15rem; font-size: 0.82rem; line-height: 1.4;
    }
    [data-testid="stSidebar"] [data-testid="stForm"] { margin: 0; }
    [data-testid="stSidebar"] [data-testid="stForm"] [data-testid="stVerticalBlock"] {
        gap: 0.55rem !important;
    }

    /* Tabs: segmented control */
    [data-testid="stTabs"] { margin-top: 0.75rem; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.3rem;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.3rem;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] button {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        padding: 0.65rem 1.15rem !important;
        min-height: 42px !important;
        background: transparent !important;
        border-radius: 9px !important;
        border: none !important;
        border-bottom: none !important;
        transition: background 0.15s ease, color 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #1f5c99 !important;
        background: #ffffff !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
        border-bottom: none !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: 1.25rem; }

    /* Timeline filter bar */
    .filter-bar-title {
        font-size: 0.95rem; font-weight: 700; color: #1c2430; margin: 0 0 0.2rem 0;
    }
    .filter-bar-desc {
        font-size: 0.84rem; color: #64748b; line-height: 1.4; margin: 0 0 0.85rem 0;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.filter-bar-title) {
        background: #fafbfc !important;
        margin-bottom: 1rem !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.filter-bar-title) [data-testid="stRadio"] label {
        font-size: 0.88rem !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.filter-bar-title) [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.65rem;
        margin-top: 0.15rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:has(.filter-bar-title) .stButton button {
        min-height: 2.5rem;
        font-size: 0.88rem;
    }

    div[data-testid="metric-container"] {
        background: #f8fafc; border: 1px solid #e8edf2; border-radius: 10px;
        padding: 0.85rem 1rem; min-height: 86px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; color: #1f5c99; }
    div[data-testid="stMetricLabel"] { color: #64748b; font-size: 0.85rem; white-space: normal !important; }
</style>
"""


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@contextmanager
def _panel(title: str, description: str = ""):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if description:
            st.caption(description)
        yield


def _show_figure(fig: plt.Figure) -> None:
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    plt.close(fig)


def _load_shipment(shipment_id: str) -> dict:
    events = load_events(WALKTHROUGH_DIR)
    labels = load_labels(WALKTHROUGH_DIR)
    decisions = load_decision_times(WALKTHROUGH_DIR)
    return {
        "events": events_for_shipment(events, shipment_id),
        "decisions": [(sid, t) for sid, t in decisions if sid == shipment_id],
        "incident_at": incident_time_for_shipment(labels, shipment_id),
        "shipment_id": shipment_id,
    }


def _shipment_meta(shipment_id: str) -> dict | None:
    for row in walkthrough_shipment_options(WALKTHROUGH_DIR):
        if str(row.get("shipment_id")) == shipment_id:
            return row
    return None


def _generate_demo_stream(seed: int) -> dict:
    summary = write_walkthrough_dataset(WALKTHROUGH_DIR, seed)
    return summary  # type: ignore[return-value]


def _plot_confusion(matrix: list[list[int]]) -> plt.Figure:
    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix=arr, display_labels=["No incident", "Incident"]).plot(
        ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize": 13}
    )
    ax.set_title(f"Confusion matrix, {int(arr[0,0]+arr[1,1])}/{int(arr.sum())} correct", pad=12)
    fig.subplots_adjust(bottom=0.14, top=0.88)
    return fig


def _plot_model_comparison(comparison: dict[str, dict[str, float]], selected: str) -> plt.Figure:
    short = {"logreg": "Logistic reg.", "hgb": "Grad. boosting"}
    keys = list(comparison.keys())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, key, vals, ylab in zip(
        axes,
        ["pr_auc", "brier_score"],
        [[comparison[k]["pr_auc"] for k in keys], [comparison[k]["brier_score"] for k in keys]],
        ["PR-AUC ↑", "Brier ↓"],
        strict=True,
    ):
        colors = ["#1f5c99" if k == selected else "#cbd5e1" for k in keys]
        ax.bar([short.get(k, k) for k in keys], vals, color=colors, width=0.55)
        ax.set_ylabel(ylab)
        ax.grid(axis="y", alpha=0.4)
    fig.suptitle(f"Selected: {short.get(selected, selected)}", fontsize=11)
    fig.subplots_adjust(wspace=0.32, top=0.82, bottom=0.18)
    return fig


def _plot_pr_curve(pr_curve: dict[str, list[float]], pr_auc: float, baseline: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(pr_curve["recall"], pr_curve["precision"], linewidth=2.2, color="#1f5c99", label=f"Model {pr_auc:.3f}")
    ax.axhline(y=max(baseline, 0.01), color="#94a3b8", linestyle="--", label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.4)
    fig.subplots_adjust(bottom=0.14, top=0.88)
    return fig


def _plot_risk_scores(pred_df: pd.DataFrame, shipment_id: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = pred_df["as_of"].str[-5:].tolist()
    values = pred_df["probability"].tolist()
    colors = ["#22a06b" if v < 0.3 else "#e06c00" if v < 0.7 else "#c9372c" for v in values]
    ax.bar(labels, values, color=colors, width=0.5)
    ax.axhline(0.5, color="#64748b", linestyle="--")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Risk scores for {shipment_id}")
    ax.grid(axis="y", alpha=0.4)
    fig.subplots_adjust(bottom=0.14, top=0.88)
    return fig


def _shipment_color(shipment_id: str) -> str:
    return SHIPMENT_COLORS.get(shipment_id, "#64748b")


def _shipment_short_label(shipment_id: str) -> str:
    meta = _shipment_meta(shipment_id)
    if meta and meta.get("use_case"):
        return str(meta["use_case"])
    return shipment_id.replace("s-demo-", "").replace("s-", "")


def _events_to_timeline_df(events: list, shipment_id: str) -> pd.DataFrame:
    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, object]] = []
    for idx, event in enumerate(events):
        key = (event.event_id, event.revision)
        duplicate = key in seen
        if not duplicate:
            seen.add(key)
        delay_min = (_utc(event.received_at) - _utc(event.device_time)).total_seconds() / 60.0
        flags: list[str] = []
        if duplicate:
            flags.append("duplicate")
        if event.revision > 1:
            flags.append("correction")
        if event.kind == "door_open":
            flags.append("door")
        if delay_min >= 60:
            flags.append("delayed")
        rows.append(
            {
                "shipment_id": shipment_id,
                "delivery_idx": idx,
                "event_id": event.event_id,
                "kind": event.kind,
                "value": event.value,
                "revision": event.revision,
                "device_time": _utc(event.device_time),
                "received_at": _utc(event.received_at),
                "flags": ", ".join(flags) if flags else "normal",
                "delay_min": round(delay_min, 1),
            }
        )
    return pd.DataFrame(rows)


def _format_timeline_axis(ax: plt.Axes, label: str) -> None:
    ax.set_xlabel(label, labelpad=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")


def _plot_timeline_chart(
    timeline_df: pd.DataFrame,
    *,
    time_col: str,
    title: str,
) -> plt.Figure:
    shipment_ids = list(dict.fromkeys(timeline_df["shipment_id"].tolist()))
    fig = plt.figure(figsize=(11, 8.0))
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[3.2, 1.2],
        hspace=0.68,
        top=0.94,
        bottom=0.07,
        left=0.09,
        right=0.97,
    )
    ax = fig.add_subplot(gs[0, 0])
    leg_ax = fig.add_subplot(gs[1, 0])
    leg_ax.axis("off")

    max_temp = 8.5
    for shipment_id in shipment_ids:
        ship_df = timeline_df[timeline_df["shipment_id"] == shipment_id].copy()
        color = _shipment_color(shipment_id)
        temps = ship_df[ship_df["kind"] == "temperature_c"].sort_values(time_col)

        if not temps.empty:
            values = temps["value"].astype(float)
            max_temp = max(max_temp, float(values.max()))
            ax.plot(
                temps[time_col],
                values,
                color=color,
                linewidth=1.8,
                alpha=0.35,
                zorder=1,
            )
            normal = temps[~temps["flags"].str.contains("duplicate|correction", na=False, regex=True)]
            if not normal.empty:
                ax.scatter(
                    normal[time_col],
                    normal["value"].astype(float),
                    color=color,
                    s=38,
                    zorder=2,
                )
            duplicates = temps[temps["flags"].str.contains("duplicate", na=False)]
            if not duplicates.empty:
                ax.scatter(
                    duplicates[time_col],
                    duplicates["value"].astype(float),
                    marker="o",
                    s=68,
                    facecolors="none",
                    edgecolors="#1c2430",
                    linewidths=1.8,
                    zorder=4,
                )
            corrections = temps[temps["flags"].str.contains("correction", na=False)]
            if not corrections.empty:
                ax.scatter(
                    corrections[time_col],
                    corrections["value"].astype(float),
                    marker="D",
                    s=88,
                    color=color,
                    edgecolors="#1c2430",
                    linewidths=0.8,
                    zorder=4,
                )

        doors = ship_df[ship_df["kind"] == "door_open"]
        if not doors.empty:
            door_y = max_temp + 0.6
            ax.scatter(
                doors[time_col],
                [door_y] * len(doors),
                marker="v",
                s=100,
                color=color,
                edgecolors="#1c2430",
                linewidths=0.8,
                zorder=5,
            )

    ax.axhline(8.0, color="#94a3b8", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Temperature (°C)", labelpad=10)
    ax.set_title(title, pad=14, fontsize=11)
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0, max_temp + 1.4)
    ax.margins(x=0.02)
    _format_timeline_axis(ax, "Device time" if time_col == "device_time" else "Received at (platform time)")

    shipment_handles = [
        Line2D(
            [0],
            [0],
            color=_shipment_color(sid),
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=_shipment_short_label(sid),
        )
        for sid in shipment_ids
    ]
    shipment_handles.append(
        Line2D([0], [0], color="#94a3b8", linestyle="--", linewidth=1.5, label="8 °C guide")
    )
    marker_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor=c, markeredgecolor=c, markersize=8, label=n)
        for n, m, c in TIMELINE_LEGEND
    ]

    ship_ncol = min(3, max(len(shipment_handles), 1))
    ship_legend = leg_ax.legend(
        handles=shipment_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ship_ncol,
        fontsize=9,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    leg_ax.add_artist(ship_legend)
    leg_ax.legend(
        handles=marker_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=9,
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    return fig


def _score_stream(engine: RiskEngine, shipment: dict) -> pd.DataFrame:
    rows = []
    for _, as_of in shipment["decisions"]:
        pred = engine.score(shipment["shipment_id"], as_of)
        rows.append(
            {
                "as_of": as_of.strftime("%Y-%m-%d %H:%M"),
                "probability": round(pred.probability, 3),
                "probability_pct": f"{pred.probability:.1%}",
                "reasons": ", ".join(pred.reasons) or "(none)",
                "true_label_next_6h": label_in_next_6h(as_of, shipment["incident_at"]),
            }
        )
    return pd.DataFrame(rows)


def _render_training_tab(metrics: dict, holdout: dict) -> None:
    selected = str(metrics.get("selected_model", "logreg"))
    with _panel("Summary", "80/20 split by shipment. Logistic regression vs gradient boosting."):
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected model", selected.upper())
        c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        c3.metric("Brier", f"{metrics['brier_score']:.4f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Train rows", metrics["train_rows"])
        c5.metric("Test rows", metrics["test_rows"])
        c6.metric("Accuracy @0.5", f"{holdout['accuracy']:.3f}")

    if "model_comparison" in metrics:
        with _panel("Model comparison"):
            _show_figure(_plot_model_comparison(metrics["model_comparison"], selected))

    with _panel("Evaluation charts"):
        left, right = st.columns(2, gap="large")
        with left:
            _show_figure(_plot_confusion(holdout["confusion_matrix"]))
        with right:
            rate = sum(holdout["y_true"]) / max(len(holdout["y_true"]), 1)
            _show_figure(_plot_pr_curve(holdout["pr_curve"], holdout["pr_auc"], rate))


def _render_gen_notice() -> None:
    manifest = st.session_state.get("gen_notice")
    if not manifest:
        return

    with st.container(border=True):
        msg_col, close_col = st.columns([11, 1])
        with msg_col:
            seed_note = f", seed <code>{manifest.get('seed')}</code>" if manifest.get("seed") is not None else ""
            st.markdown(
                f'<div class="banner-ok">✓ Generated <b>{manifest.get("shipment_count", 5)} demo shipments</b> '
                f'({manifest.get("events")} events){seed_note}. Saved to <code>optional/data/walkthrough/</code>.</div>',
                unsafe_allow_html=True,
            )
        with close_col:
            if st.button("✕", key="dismiss_gen_notice", help="Dismiss acknowledgement"):
                st.session_state.gen_notice = None
                st.rerun()
        st.dataframe(
            pd.DataFrame(manifest["shipments"])[["shipment_id", "use_case", "talk_about", "has_incident", "events"]].rename(
                columns={"talk_about": "notes"}
            ),
            hide_index=True,
            use_container_width=True,
        )


def _shipment_has_late_correction(shipment_id: str) -> bool:
    """True when this demo stream includes a late-arriving revision."""
    shipment = _load_shipment(shipment_id)
    return any(event.revision > 1 for event in shipment["events"])


def _sync_demo_shipment_selection(shipment_id: str) -> None:
    """Keep Phase B tabs aligned when the demo shipment changes."""
    if st.session_state.get("_last_phase_b_shipment") != shipment_id:
        st.session_state._last_phase_b_shipment = shipment_id
        st.session_state.timeline_compare_ids = [shipment_id]
        st.session_state["timeline_multiselect"] = [shipment_id]


@contextmanager
def _sidebar_block(title: str, description: str = ""):
    with st.container(border=True):
        st.markdown(f'<div class="sidebar-section">{title}</div>', unsafe_allow_html=True)
        if description:
            st.markdown(f'<div class="sidebar-desc">{description}</div>', unsafe_allow_html=True)
        yield


def _render_phase_b_sidebar() -> str | None:
    """Shared demo shipment picker for inference, timeline, and correction tabs."""
    options = walkthrough_shipment_options(WALKTHROUGH_DIR)
    if not options:
        return None

    ids = [str(r["shipment_id"]) for r in options]
    label_for = {str(r["shipment_id"]): str(r["use_case"]) for r in options}

    preselect = st.session_state.get("phase_b_shipment")
    if preselect not in ids:
        preselect = st.session_state.shipment_id if st.session_state.shipment_id in ids else ids[0]
    st.session_state.shipment_id = preselect

    pick = st.selectbox(
        "Demo shipment",
        ids,
        index=ids.index(preselect),
        format_func=lambda sid: label_for.get(sid, sid),
        key="phase_b_shipment",
    )
    st.session_state.shipment_id = pick
    _sync_demo_shipment_selection(pick)

    meta = _shipment_meta(pick)
    if meta:
        st.warning(f"**{meta.get('use_case', '')}**: {meta.get('talk_about', '')}")
        if _shipment_has_late_correction(pick):
            st.caption("This shipment has a late correction. Open the **Late correction** tab.")
    return pick


def _render_inference_tab(shipment_id: str) -> None:
    shipment = _load_shipment(shipment_id)
    engine = RiskEngine(ARTIFACT_DIR, max_shipments=10)
    for event in shipment["events"]:
        engine.ingest(event)
    with _panel(f"Scores for {shipment_id}"):
        pred_df = _score_stream(engine, shipment)
        _show_figure(_plot_risk_scores(pred_df, shipment_id))
        st.dataframe(
            pred_df.rename(columns={
                "as_of": "Score as of", "probability_pct": "Risk",
                "reasons": "Reasons",                 "true_label_next_6h": "Actual label",
            })[["Score as of", "Risk", "Reasons", "Actual label"]],
            hide_index=True,
            use_container_width=True,
        )


def _render_timeline_tab(shipment_id: str) -> None:
    options = walkthrough_shipment_options(WALKTHROUGH_DIR)
    option_ids = [str(r["shipment_id"]) for r in options]
    option_labels = {sid: f"{_shipment_short_label(sid)} ({sid})" for sid in option_ids}

    compare_default = st.session_state.get("timeline_compare_ids") or [shipment_id]
    compare_default = [sid for sid in compare_default if sid in option_ids] or [shipment_id]

    if st.session_state.get("timeline_pick_all"):
        picked = [sid for sid in st.session_state.timeline_pick_all if sid in option_ids]
        st.session_state.timeline_multiselect = picked or option_ids
        st.session_state.timeline_compare_ids = list(st.session_state.timeline_multiselect)
        del st.session_state.timeline_pick_all
    elif st.session_state.get("timeline_pick_one"):
        pick_id = str(st.session_state.timeline_pick_one)
        if pick_id in option_ids:
            st.session_state.timeline_multiselect = [pick_id]
            st.session_state.timeline_compare_ids = [pick_id]
        del st.session_state.timeline_pick_one
    elif "timeline_multiselect" not in st.session_state:
        st.session_state.timeline_multiselect = compare_default

    with st.container(border=True):
        st.markdown('<div class="filter-bar-title">Chart filters</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="filter-bar-desc">Compare demo streams on one chart. '
            "Pick ingest time or sensor time for the x-axis.</div>",
            unsafe_allow_html=True,
        )

        ship_col, action_col, axis_col = st.columns([3.2, 1, 1.3], vertical_alignment="bottom")
        with ship_col:
            selected_ids = st.multiselect(
                "Shipments to compare",
                option_ids,
                format_func=lambda sid: option_labels.get(sid, sid),
                key="timeline_multiselect",
                placeholder="Pick one or more shipments",
            )
            st.session_state.timeline_compare_ids = selected_ids
        with action_col:
            pick_all = st.button("Select all 5", use_container_width=True, type="secondary")
            pick_one = st.button("This shipment", use_container_width=True)
        with axis_col:
            time_axis = st.radio(
                "X-axis",
                ["received_at", "device_time"],
                format_func=lambda k: "Received at" if k == "received_at" else "Device time",
                horizontal=True,
                help="Received at is when the platform got the event. Device time is the sensor clock.",
            )

        if pick_all:
            st.session_state.timeline_pick_all = option_ids
            st.rerun()
        if pick_one:
            st.session_state.timeline_pick_one = shipment_id
            st.rerun()

    if not selected_ids:
        st.warning("Pick at least one shipment to plot.")
        return

    timeline_parts = []
    for sid in selected_ids:
        shipment = _load_shipment(sid)
        timeline_parts.append(_events_to_timeline_df(shipment["events"], sid))
    timeline_df = pd.concat(timeline_parts, ignore_index=True)

    axis_label = "received_at (platform)" if time_axis == "received_at" else "device_time (sensor)"
    with _panel(
        "Temperature timeline",
        "Color is shipment. Hollow circle is duplicate. Diamond is correction. Triangle is door open.",
    ):
        _show_figure(
            _plot_timeline_chart(
                timeline_df,
                time_col=time_axis,
                title=f"Temperature by {axis_label}, {len(selected_ids)} shipment(s)",
            )
        )

    with _panel("Event log", "Sorted by delivery order. Flags mark duplicates, corrections, and delays."):
        display = timeline_df.copy()
        display["device_time"] = display["device_time"].dt.strftime("%Y-%m-%d %H:%M")
        display["received_at"] = display["received_at"].dt.strftime("%Y-%m-%d %H:%M")
        display = display.rename(
            columns={
                "shipment_id": "Shipment",
                "delivery_idx": "#",
                "event_id": "Event ID",
                "kind": "Kind",
                "value": "Value",
                "revision": "Rev",
                "device_time": "Device time",
                "received_at": "Received at",
                "flags": "Flags",
                "delay_min": "Delay (min)",
            }
        )

        def _row_style(row: pd.Series) -> list[str]:
            flag = str(row["Flags"])
            if "correction" in flag:
                return ["background-color: #f3e8ff"] * len(row)
            if "duplicate" in flag:
                return ["background-color: #fef3c7"] * len(row)
            if "door" in flag:
                return ["background-color: #ffedd5"] * len(row)
            if "delayed" in flag:
                return ["background-color: #e0f2fe"] * len(row)
            return [""] * len(row)

        st.dataframe(
            display.style.apply(_row_style, axis=1),
            hide_index=True,
            use_container_width=True,
            height=min(420, 38 + 35 * len(display)),
        )

    focus_id = shipment_id if shipment_id in selected_ids else selected_ids[0]
    shipment = _load_shipment(focus_id)
    engine = RiskEngine(ARTIFACT_DIR, max_shipments=10)
    for event in shipment["events"]:
        engine.ingest(event)
    with _panel("Score at a decision time", f"Demo shipment: {option_labels.get(shipment_id, shipment_id)}"):
        times = [t for _, t in shipment["decisions"]]
        if times:
            as_of = st.select_slider("Decision time", times, format_func=lambda t: t.strftime("%H:%M UTC"))
            pred = engine.score(shipment_id, as_of)
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk", f"{pred.probability:.1%}")
            m2.metric("Events known", sum(1 for e in shipment["events"] if _utc(e.received_at) <= _utc(as_of)))
            m3.metric("Reasons", ", ".join(pred.reasons) or "none")


def _render_correction_tab(shipment_id: str) -> None:
    shipment = _load_shipment(shipment_id)
    correction = next((e for e in shipment["events"] if e.revision > 1), None)
    if correction is None:
        st.info("No late correction in this shipment.")
        return
    as_of_past = next(t for _, t in shipment["decisions"] if t.hour == _utc(correction.device_time).hour)
    demo = RiskEngine(ARTIFACT_DIR, max_shipments=10)
    for event in shipment["events"]:
        if event is correction:
            break
        demo.ingest(event)
    before = demo.score(shipment_id, as_of_past)
    demo.ingest(correction)
    after = demo.score(shipment_id, as_of_past)
    with _panel("Late correction"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Before", f"{before.probability:.1%}")
        c2.metric("After", f"{after.probability:.1%}")
        c3.metric("Unchanged?", "Yes" if before.to_wire() == after.to_wire() else "No")


def main() -> None:
    st.set_page_config(page_title="Shipment Risk Analyser", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="hero"><h1>Shipment Risk Analyser</h1>'
        "<p>Train on historical data, pick a demo shipment, and score without retraining.</p></div>",
        unsafe_allow_html=True,
    )

    for key, default in (
        ("metrics", None),
        ("holdout", None),
        ("gen_notice", None),
        ("shipment_id", None),
        ("demo_seed", 4242),
        ("timeline_compare_ids", None),
        ("_last_phase_b_shipment", None),
    ):
        if key not in st.session_state:
            st.session_state[key] = default

    if (ARTIFACT_DIR / "model.joblib").exists() and st.session_state.metrics is None:
        st.session_state.metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
        rows = build_training_rows(load_events(DATA_DIR), load_labels(DATA_DIR), load_decision_times(DATA_DIR))
        st.session_state.holdout = holdout_evaluation(rows, ARTIFACT_DIR)

    model_ready = (ARTIFACT_DIR / "model.joblib").exists()
    walkthrough_ready = bool(walkthrough_shipment_options(WALKTHROUGH_DIR))
    phase_b_ready = model_ready and walkthrough_ready
    shipment_id: str | None = None

    # --- Sidebar: training, demo data, Phase B shipment ---
    with st.sidebar:
        with _sidebar_block("Training", "Train on files in <code>data/</code>. Demo shipments are not used here."):
            if st.button("Train model", type="primary", use_container_width=True):
                with st.spinner("Training..."):
                    rows = build_training_rows(
                        load_events(DATA_DIR), load_labels(DATA_DIR), load_decision_times(DATA_DIR)
                    )
                    st.session_state.metrics = train(rows, ARTIFACT_DIR)
                    st.session_state.holdout = holdout_evaluation(rows, ARTIFACT_DIR)
                st.toast("Model saved", icon="✅")

        with _sidebar_block("Demo data"):
            with st.form("demo_form", border=False):
                seed_input = st.number_input(
                    "Seed",
                    min_value=0,
                    max_value=999_999,
                    value=int(st.session_state.demo_seed),
                    step=1,
                    help="Try 7777 for a different stream.",
                )
                generate = st.form_submit_button("Generate 5 shipments", use_container_width=True)
                if generate:
                    st.session_state.demo_seed = int(seed_input)
                    manifest = _generate_demo_stream(st.session_state.demo_seed)
                    st.session_state.gen_notice = manifest
                    first_id = str(manifest["shipments"][0]["shipment_id"])
                    st.session_state.shipment_id = first_id
                    st.session_state._last_phase_b_shipment = first_id
                    st.session_state.timeline_compare_ids = [first_id]
                    st.session_state["timeline_multiselect"] = [first_id]
                    st.rerun()

            if walkthrough_ready:
                count = len(walkthrough_shipment_options(WALKTHROUGH_DIR))
                st.caption(f"Seed **{st.session_state.demo_seed}**, {count} shipments loaded.")
            elif not phase_b_ready:
                st.caption("Generate demo shipments and train the model to use the scoring tabs.")

        if phase_b_ready:
            with _sidebar_block("Demo shipment", "Used by the inference, timeline, and correction tabs."):
                shipment_id = _render_phase_b_sidebar()
        elif walkthrough_ready and not model_ready:
            with _sidebar_block("Demo shipment", "Train the model first to score demo shipments."):
                st.caption("Demo files are ready. Click **Train model** above.")

    _render_gen_notice()

    if not walkthrough_ready and st.session_state.gen_notice is None:
        st.info("In the sidebar, set a seed and click **Generate 5 demo shipments**.")

    tab_labels = ["Training metrics", "Inference", "Event timeline"]
    show_correction_tab = bool(
        phase_b_ready and shipment_id and _shipment_has_late_correction(shipment_id)
    )
    if show_correction_tab:
        tab_labels.append("Late correction")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        if st.session_state.metrics is None:
            st.info("Click **Train model** in the sidebar. Training uses `data/` only, not demo shipments.")
        else:
            _render_training_tab(st.session_state.metrics, st.session_state.holdout)

    with tabs[1]:
        if not phase_b_ready:
            st.info("Train model and generate demo shipments first.")
        elif shipment_id:
            _render_inference_tab(shipment_id)

    with tabs[2]:
        if not phase_b_ready:
            st.info("Train model and generate demo shipments first.")
        elif shipment_id:
            _render_timeline_tab(shipment_id)

    if show_correction_tab and shipment_id:
        with tabs[3]:
            _render_correction_tab(shipment_id)


if __name__ == "__main__":
    main()
