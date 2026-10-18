"""Reusable PipelineIQ interface pieces."""

from __future__ import annotations

import streamlit as st

from config import FEATURE_LABELS


def html(fragment: str) -> None:
    """Render markup directly so card grids are not broken up by Markdown."""
    st.html(fragment)


def plotly_theme() -> dict:
    """Shared Plotly layout: transparent surfaces, muted grid, Inter type."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "color": "#E5E7EB", "size": 13},
        "margin": {"l": 48, "r": 20, "t": 56, "b": 40},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#E5E7EB"}},
        "colorway": ["#2DD4BF", "#6366F1", "#F59E0B", "#EF4444", "#22C55E", "#38BDF8"],
    }


def brand() -> None:
    html(
        """
        <div class="brand">
          <div class="brand-mark">IQ</div>
          <div>
            <div class="brand-name">PipelineIQ</div>
            <div class="brand-sub">Developer-aware CI/CD</div>
          </div>
        </div>
        """
    )


def kpi_row(cards: list[dict]) -> None:
    """Five (or fewer) animated KPI cards. Each card dict: label, value, text, note."""
    parts = []
    for card in cards:
        value = card["value"]
        decimals = card.get("decimals", 0)
        prefix = card.get("prefix", "")
        suffix = card.get("suffix", "")
        parts.append(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{card["label"]}</div>
              <div class="kpi-value" data-count="{value}" data-decimals="{decimals}"
                   data-prefix="{prefix}" data-suffix="{suffix}">{card["text"]}</div>
              <div class="kpi-delta">{card.get("note", "")}</div>
            </div>
            """
        )
    html(f'<div class="kpi-grid">{"".join(parts)}</div>')


def section_header(title: str, subtitle: str = "") -> None:
    html(f'<div class="section"><h2>{title}</h2><p>{subtitle}</p></div>')


def risk_badge(level: str) -> str:
    label = {"green": "LOW", "amber": "MEDIUM", "red": "HIGH", "low": "LOW", "medium": "MEDIUM", "high": "HIGH"}.get(
        str(level).lower(), str(level).upper()
    )
    css = {"LOW": "badge-low", "MEDIUM": "badge-medium", "HIGH": "badge-high"}.get(label, "")
    live = " dot-live" if label == "HIGH" else ""
    return f'<span class="badge {css}"><span class="dot{live}"></span> {label}</span>'


def alert_card(alert: dict, when: str) -> None:
    features = []
    for piece in str(alert.get("top_3_contributing_features", "")).split(";"):
        if not piece or ":" not in piece:
            continue
        name, raw = piece.split(":", 1)
        features.append(f'<span class="feature-chip">{FEATURE_LABELS.get(name, name)} {raw}</span>')
    prob = float(alert.get("ml_probability", 0))
    html(
        f"""
        <div class="alert-card">
          <div class="alert-top">
            <div>{risk_badge(alert.get("severity_label", "low"))}</div>
            <div class="muted">{when}</div>
          </div>
          <div class="alert-title">{alert.get("stage", "").replace("_", " ").title()} · {alert.get("project_name", alert.get("project_id"))}</div>
          <div class="muted">z-score <span class="mono">{float(alert.get("z_score", 0)):.2f}</span> · window {int(alert.get("anomaly_window_runs", 0))} runs · {alert.get("platform", "")}</div>
          <div class="prob-track"><div class="prob-fill" style="width:{prob * 100:.0f}%"></div></div>
          <div class="muted">Model probability <span class="mono">{prob:.0%}</span></div>
          <div class="chips">{"".join(features)}</div>
          <p>{alert.get("plain_language_reason", "")}</p>
        </div>
        """
    )


def insight_callout(kicker: str, text: str) -> str:
    return f'<div class="callout"><div class="kicker">{kicker}</div><p>{text}</p></div>'


def missing_file(path: str, command: str) -> None:
    html(
        f"""
        <div class="callout">
          <div class="kicker">WAITING ON DATA</div>
          <p><strong>{path}</strong> has not been generated yet.</p>
          <p class="mono">{command}</p>
        </div>
        """
    )


def styled_table(headers: list[str], rows: list[list[str]]) -> None:
    head = "".join(f"<th>{cell}</th>" for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    html(f'<div class="table-wrap"><table class="piq"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
