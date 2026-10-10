"""
NeuroScale — AI Powered Autonomous Docker Resource Manager
Production-grade multi-page monitoring dashboard using Streamlit's official st.navigation() and st.Page().
Guarantees that direct URL navigation (e.g. /p1_live_monitoring, /p2_ai_analytics, etc.)
always renders the full page view rather than a blank screen.
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules import reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.styles.theme import DARK_THEME_CSS
from dashboard.components.data_provider import BackendDataProvider
from dashboard.components.header import render_header, render_status_bar, render_footer

# Page view renderers
from dashboard.views.p1_live_monitoring import render_live_monitoring_page
from dashboard.views.p2_ai_analytics import render_ai_analytics_page
from dashboard.views.p3_dataset_explorer import render_dataset_explorer_page
from dashboard.views.p4_model_performance import render_model_performance_page
from dashboard.views.p5_system_architecture import render_system_architecture_page


# Global data provider instance
provider = BackendDataProvider(project_root=PROJECT_ROOT)


def _render_common_chrome():
    """Render CSS theme, header, and live status bar across all pages."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    status = provider.get_system_status()
    render_header()
    render_status_bar(status)
    return status


def _render_sidebar(status):
    """Render the sidebar telemetry details, mode toggles, and run selector."""
    st.sidebar.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <span style="font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px;">
            NEUROSCALE CONSOLE
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 1. Mode Controls (Live vs Replay & Autonomous Mode)
    if "telemetry_mode" not in st.session_state:
        st.session_state["telemetry_mode"] = "Replay Mode"

    if "autonomous_recommendations" not in st.session_state:
        st.session_state["autonomous_recommendations"] = True

    st.sidebar.markdown("<h4 style='font-size: 12px; margin: 0 0 6px 0; color: #94a3b8; text-transform: uppercase; font-weight: 700;'>Telemetry Stream Mode</h4>", unsafe_allow_html=True)
    telemetry_mode = st.sidebar.radio(
        "Telemetry Stream Mode",
        options=["Replay Mode", "Live Mode"],
        index=0 if st.session_state["telemetry_mode"] == "Replay Mode" else 1,
        label_visibility="collapsed",
        key="telemetry_mode_radio",
    )
    st.session_state["telemetry_mode"] = telemetry_mode

    auto_rec = st.sidebar.toggle(
        "⚡ Autonomous Recommendations",
        value=st.session_state["autonomous_recommendations"],
        help="Enables real-time Transformer forecasting, Autoencoder anomaly scoring, and Factorized DQN actuation recommendations",
        key="auto_rec_toggle",
    )
    st.session_state["autonomous_recommendations"] = auto_rec

    st.sidebar.markdown("---")

    # 2. Run Selector (primarily for Replay Mode)
    available_runs = provider.get_available_runs()
    run_options = ["latest"] + available_runs

    if "selected_run" not in st.session_state:
        st.session_state["selected_run"] = "latest"

    selected_run = st.sidebar.selectbox(
        "Active Replay Run",
        options=run_options,
        index=run_options.index(st.session_state["selected_run"]) if st.session_state["selected_run"] in run_options else 0,
        disabled=(telemetry_mode == "Live Mode"),
        key="run_selector",
    )
    st.session_state["selected_run"] = selected_run

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='font-size: 13px; margin: 0 0 6px 0; color: #94a3b8;'>Environment Telemetry</h4>", unsafe_allow_html=True)
    
    active_containers_live = len(provider.live_engine.get_active_containers())
    fleet_text = f"{active_containers_live} Containers (Live)" if telemetry_mode == "Live Mode" else f"{status.get('container_count', 0)} Containers (Replay)"

    st.sidebar.info(
        f"**Active Mode:** `{telemetry_mode}`\n\n"
        f"**Active Run:** `{selected_run if telemetry_mode == 'Replay Mode' else 'live-stream'}`\n\n"
        f"**Autonomous Guard:** `{'Active' if auto_rec else 'Standby'}`\n\n"
        f"**Total Samples:** `{status.get('dataset_samples', 0):,}` rows\n\n"
        f"**Docker Engine:** `{status.get('docker_status', 'Unknown')}`\n\n"
        f"**Fleet:** `{fleet_text}`\n\n"
        f"**Checkpoints:** `{status.get('ai_status', 'Unknown')}`"
    )

    if st.sidebar.button("🔄 Refresh Telemetry Stream", use_container_width=True):
        st.rerun()

    return selected_run


# ---------------------------------------------------------------------------
# Individual Page Entrypoints (Callable Functions)
# ---------------------------------------------------------------------------

def page_live_monitoring():
    status = _render_common_chrome()
    selected_run = _render_sidebar(status)
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    render_live_monitoring_page(provider, selected_run=selected_run)
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer(status)


def page_ai_analytics():
    status = _render_common_chrome()
    _render_sidebar(status)
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    render_ai_analytics_page(provider)
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer(status)


def page_dataset_explorer():
    status = _render_common_chrome()
    selected_run = _render_sidebar(status)
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    render_dataset_explorer_page(provider, selected_run=selected_run)
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer(status)


def page_model_performance():
    status = _render_common_chrome()
    _render_sidebar(status)
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    render_model_performance_page(provider)
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer(status)


def page_system_architecture():
    status = _render_common_chrome()
    _render_sidebar(status)
    st.markdown('<div class="page-container">', unsafe_allow_html=True)
    render_system_architecture_page(provider)
    st.markdown('</div>', unsafe_allow_html=True)
    render_footer(status)


# ---------------------------------------------------------------------------
# Streamlit Official Multipage Definition (st.Page & st.navigation)
# ---------------------------------------------------------------------------

p1 = st.Page(
    page_live_monitoring,
    title="Live Monitoring",
    icon="⚡",
    url_path="p1_live_monitoring",
    default=True,
)

p2 = st.Page(
    page_ai_analytics,
    title="AI Analytics",
    icon="🧠",
    url_path="p2_ai_analytics",
)

p3 = st.Page(
    page_dataset_explorer,
    title="Dataset Explorer",
    icon="📊",
    url_path="p3_dataset_explorer",
)

p4 = st.Page(
    page_model_performance,
    title="Model Performance",
    icon="📈",
    url_path="p4_model_performance",
)

p5 = st.Page(
    page_system_architecture,
    title="System Architecture",
    icon="🏛️",
    url_path="p5_system_architecture",
)

# Register official navigation routes
pg = st.navigation(
    {
        "Operations": [p1],
        "Intelligence": [p2, p4],
        "Platform & Data": [p3, p5],
    },
    position="sidebar",
)


def main():
    st.set_page_config(
        page_title="NeuroScale | Autonomous Cloud Resource Manager",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    pg.run()


if __name__ == "__main__":
    main()
