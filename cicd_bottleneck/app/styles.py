"""Dark mission-control CSS for PipelineIQ."""

from __future__ import annotations

import streamlit as st

_BASE = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: Inter, sans-serif; }
[data-testid="stAppViewContainer"] { background: #0B1020; color: #E5E7EB; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: #0E152B; border-right: 1px solid #1F2A4D; }
[data-testid="stSidebarNav"] { display: none; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1680px; }
[data-testid="stSidebar"] { width: 320px; min-width: 320px; }
[data-testid="stSidebar"] [data-testid="stCustomComponentV1"] { background: #0E152B; }
[data-testid="stPlotlyChart"] { background: #121A33; border: 1px solid #1F2A4D; border-radius: 14px; padding: 6px 8px 10px; margin-bottom: 8px; }
.table-wrap { overflow-x: auto; margin-bottom: 16px; }
div[data-testid="stExpander"] details { background: #121A33; border: 1px solid #1F2A4D; border-radius: 12px; }
.block-container [data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }
.block-container [data-testid="stRadio"] label { background: #121A33; border: 1px solid #1F2A4D; border-radius: 999px; padding: 4px 12px; }
[data-testid="stSidebar"] label, [data-testid="stWidgetLabel"] p { color: #94A3B8; }
.stButton > button, .stDownloadButton > button { background: #121A33; color: #E5E7EB; border: 1px solid #2DD4BF; border-radius: 10px; font-weight: 600; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #5EEAD4; color: #5EEAD4; }
h1, h2, h3, p, label, span { font-family: Inter, sans-serif; }
.hero { background: linear-gradient(135deg, #121A33 0%, #10182E 55%, #0E1A2C 100%); border: 1px solid #1F2A4D; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; box-shadow: 0 0 40px rgba(45,212,191,0.08); }
.kicker { color: #2DD4BF; letter-spacing: 0.14em; font-size: 12px; font-weight: 700; }
.hero h1 { margin: 6px 0 4px; font-size: 42px; line-height: 1.05; color: #F8FAFC; }
.tagline { color: #94A3B8; font-size: 16px; margin: 0 0 14px; }
.chip, .badge { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 6px 12px; font-size: 13px; font-weight: 600; border: 1px solid #1F2A4D; background: #0B1020; color: #E5E7EB; }
.chip-teal { border-color: #2DD4BF; color: #2DD4BF; }
.brand { display: flex; gap: 12px; align-items: center; margin-bottom: 18px; }
.brand-mark { width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; font-family: "JetBrains Mono", monospace; font-weight: 700; color: #06221E; background: #2DD4BF; box-shadow: 0 0 18px rgba(45,212,191,0.45); }
.brand-name { font-weight: 700; font-size: 18px; color: #F8FAFC; }
.brand-sub { color: #94A3B8; font-size: 12px; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
.kpi-card, .callout, .alert-card, .layer-card, .panel { background: #121A33; border: 1px solid #1F2A4D; border-radius: 14px; box-shadow: 0 8px 28px rgba(2,6,23,0.35); }
.kpi-card { padding: 16px 18px; border-left: 3px solid #2DD4BF; transition: transform .18s ease, border-color .18s ease; }
.kpi-card:hover, .alert-card:hover { transform: translateY(-3px); border-color: #2DD4BF; }
.kpi-label { color: #94A3B8; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.kpi-value { font-family: "JetBrains Mono", monospace; font-size: clamp(20px, 1.5vw, 30px); color: #F8FAFC; margin: 6px 0 2px; }
.kpi-delta { color: #94A3B8; font-size: 12px; }
.callout { padding: 16px 18px; margin-bottom: 12px; }
.callout .kicker { margin-bottom: 6px; }
.callout p { margin: 0; color: #E5E7EB; line-height: 1.45; }
.section { margin: 8px 0 16px; }
.section h2 { margin: 0; font-size: 22px; color: #F8FAFC; }
.section p { margin: 4px 0 0; color: #94A3B8; }
.badge-low { color: #22C55E; border-color: #22C55E; }
.badge-medium { color: #F59E0B; border-color: #F59E0B; }
.badge-high { color: #EF4444; border-color: #EF4444; }
.dot { width: 8px; height: 8px; border-radius: 99px; background: currentColor; display: inline-block; }
.dot-live { animation: pulse 1.2s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: .35; } 100% { opacity: 1; } }
.alert-card { padding: 16px 18px; margin-bottom: 12px; }
.alert-top { display: flex; justify-content: space-between; gap: 12px; align-items: center; }
.alert-title { font-weight: 650; color: #F8FAFC; }
.muted { color: #94A3B8; font-size: 13px; }
.mono { font-family: "JetBrains Mono", monospace; }
.prob-track { height: 8px; background: #0B1020; border-radius: 99px; margin: 8px 0; overflow: hidden; }
.prob-fill { height: 8px; background: linear-gradient(90deg, #F59E0B, #EF4444); border-radius: 99px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.feature-chip { background: #0B1020; border: 1px solid #1F2A4D; border-radius: 999px; padding: 4px 10px; font-size: 12px; color: #E5E7EB; }
.story { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-bottom: 24px; }
.layers { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
.layer-card { padding: 14px; min-height: 140px; }
.layer-card strong { display: block; color: #2DD4BF; margin-bottom: 6px; }
.arrow-row { display: flex; justify-content: space-between; color: #6366F1; font-size: 12px; margin: -8px 8px 16px; }
table.piq { width: 100%; border-collapse: collapse; background: #121A33; border: 1px solid #1F2A4D; border-radius: 14px; overflow: hidden; }
table.piq th { text-align: left; color: #94A3B8; font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; padding: 10px 12px; border-bottom: 1px solid #1F2A4D; }
table.piq td { padding: 10px 12px; border-bottom: 1px solid #1F2A4D; color: #E5E7EB; vertical-align: middle; }
table.piq tr:last-child td { border-bottom: 0; }
.rule { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 14px 16px; }
.rule span { background: #0B1020; border: 1px solid #1F2A4D; border-radius: 10px; padding: 8px 10px; }
@media (max-width: 1200px) {
  .kpi-grid, .story, .layers { grid-template-columns: 1fr 1fr; }
  .hero h1 { font-size: 32px; }
}
"""

_PRESENT = """
.block-container { max-width: 1860px; }
.hero h1 { font-size: 52px; }
.kpi-value { font-size: 38px; }
.section h2 { font-size: 28px; }
"""


def inject_styles(presentation: bool = False) -> None:
    """Inject the global theme. Presentation mode enlarges type for a projector."""
    css = _BASE + (_PRESENT if presentation else "")
    st.html(f"<style>{css}</style>")
