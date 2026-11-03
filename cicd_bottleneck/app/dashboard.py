"""PipelineIQ — developer-aware CI/CD bottleneck intelligence."""

from __future__ import annotations

import warnings

import streamlit as st

from app.components import brand, html
from app.data import active_filter_count, alerts, apply_filters, metrics, predictions, report_html
from app.pages import alerts as alerts_page
from app.pages import architecture, developers, model_lab, overview, simulator, stages
from app.styles import inject_styles
from config import PLATFORMS

warnings.filterwarnings("ignore", message=r"The `probability` parameter was deprecated.*", category=FutureWarning)

PAGES = {
    "Executive Overview": overview.render,
    "Developer Behavior": developers.render,
    "Stage Performance": stages.render,
    "Alert Center": alerts_page.render,
    "What-If Simulator": simulator.render,
    "Model Lab": model_lab.render,
    "Architecture": architecture.render,
}

_NAV = (
    ("Executive Overview", ":material/speed:"),
    ("Developer Behavior", ":material/groups:"),
    ("Stage Performance", ":material/stacked_bar_chart:"),
    ("Alert Center", ":material/notifications:"),
    ("What-If Simulator", ":material/tune:"),
    ("Model Lab", ":material/science:"),
    ("Architecture", ":material/account_tree:"),
)


def _choose_page() -> str:
    if "page" not in st.session_state:
        st.session_state.page = "Executive Overview"

    def _go(name: str) -> None:
        st.session_state.page = name

    with st.container(key="nav", gap="xxsmall"):
        for name, icon in _NAV:
            st.button(
                name,
                icon=icon,
                key=f"nav_{name}",
                width="stretch",
                type="primary" if st.session_state.page == name else "secondary",
                on_click=_go,
                args=(name,),
            )
    return st.session_state.page


def _filters(frame) -> dict:
    projects = sorted(frame["project_id"].unique())
    names = dict(zip(frame["project_id"], frame["project_name"]))
    developers_ids = sorted(frame["developer_id"].unique())
    bands = ["green", "amber", "red"]
    min_day = frame["timestamp"].min().date()
    max_day = frame["timestamp"].max().date()
    if "ui_v2" not in st.session_state:
        st.session_state.ui_v2 = True
        st.session_state.default_start = min_day
        st.session_state.default_end = max_day
        st.session_state.f_projects = []
        st.session_state.f_developers = []
        st.session_state.f_platforms = []
        st.session_state.f_bands = []
        st.session_state.f_dates = (min_day, max_day)
    with st.expander("Filters", expanded=True, icon=":material/filter_alt:"):
        html('<p class="muted">Leave a filter blank to include everything.</p>')
        chosen_projects = st.multiselect(
            "Projects",
            projects,
            key="f_projects",
            placeholder="All projects",
            format_func=lambda key: f"{names.get(key, key)}",
        )
        dates = st.date_input("Date range", key="f_dates", min_value=min_day, max_value=max_day)
        if isinstance(dates, tuple) and len(dates) == 2:
            start, end = dates
        else:
            start = end = dates if not isinstance(dates, tuple) else dates[0]
        chosen_developers = st.multiselect("Developers", developers_ids, key="f_developers", placeholder="All developers")
        chosen_platforms = st.multiselect("Platform", list(PLATFORMS), key="f_platforms", placeholder="All platforms")
        chosen_bands = st.multiselect("Risk band", bands, key="f_bands", format_func=str.title, placeholder="All risk bands")
        filters = {
            "projects": chosen_projects,
            "developers": chosen_developers,
            "platforms": chosen_platforms,
            "risk_bands": chosen_bands,
            "start": start,
            "end": end,
        }
        defaults = {
            "projects": [],
            "developers": [],
            "platforms": [],
            "risk_bands": [],
            "start": st.session_state.default_start,
            "end": st.session_state.default_end,
        }
        count = active_filter_count(filters, defaults)
        html(f'<div class="chip">{"No filters" if count == 0 else f"{count} filter" + ("s" if count != 1 else "") + " active"}</div>')

        def _reset() -> None:
            st.session_state.f_projects = []
            st.session_state.f_developers = []
            st.session_state.f_platforms = []
            st.session_state.f_bands = []
            st.session_state.f_dates = (st.session_state.default_start, st.session_state.default_end)

        st.button("Reset filters", on_click=_reset, width="stretch")
    st.toggle("Presentation mode", key="presentation")
    view = apply_filters(frame, filters)
    alert_view = apply_filters(alerts(), filters)
    report = report_html(view if view is not None else frame.iloc[0:0], metrics(), 0 if alert_view is None else len(alert_view))
    st.download_button("Export report", report, file_name="pipelineiq_report.html", mime="text/html", width="stretch")
    return filters


def _hero(payload: dict | None) -> None:
    if payload:
        name = payload.get("deployed_model_display", payload["best_model_display"])
        f1 = payload.get("deployed_f1", payload["best_f1"])
        chip = f"Model: {name} • F1 {f1 * 100:.1f}%"
    else:
        chip = "Model: waiting on reports/metrics.json"
    html(
        f"""
        <div class="hero">
          <div class="kicker">CI/CD INTELLIGENCE</div>
          <h1>PipelineIQ</h1>
          <p class="tagline">Predict bottlenecks from developer behavior — before the pipeline runs</p>
          <span class="chip chip-teal">{chip}</span>
        </div>
        """
    )


def main() -> None:
    st.set_page_config(page_title="PipelineIQ", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
    inject_styles(bool(st.session_state.get("presentation")))
    frame = predictions()
    with st.sidebar:
        brand()
        page = _choose_page()
        filters = _filters(frame) if frame is not None else {"projects": [], "developers": [], "platforms": [], "risk_bands": [], "start": None, "end": None}
    inject_styles(bool(st.session_state.get("presentation")))
    if page == "Executive Overview":
        _hero(metrics())
    with st.spinner("Loading pipeline evidence..."):
        PAGES[page](filters)


if __name__ == "__main__":
    main()
