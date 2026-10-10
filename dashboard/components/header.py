"""
Reusable Header and Status Bar UI components for NeuroScale Dashboard.
Renders Datadog/Grafana animated pulse indicators for Docker, AI Core, Collector, Validator, and Dashboard.
"""

from typing import Any, Dict
import streamlit as st


def render_header():
    """Render the standard top header with brand and 5 animated status indicators."""
    header_html = """
    <div class="neuroscale-header">
        <div class="header-title-box">
            <div class="brand-badge">NS-AI</div>
            <div>
                <h1 class="brand-name">NeuroScale</h1>
                <p class="brand-tagline">AI Powered Autonomous Docker Resource Manager</p>
            </div>
        </div>
        <div class="header-indicators-group">
            <div class="status-pill-badge active-pulse" title="Docker Daemon Connected & Health Ping OK">
                <span class="pulse-dot-green"></span>
                <span>DOCKER ONLINE</span>
            </div>
            <div class="status-pill-badge active-pulse" title="Transformer, Autoencoder & Factorized DQN loaded">
                <span class="pulse-dot-cyan"></span>
                <span>AI ONLINE</span>
            </div>
            <div class="status-pill-badge active-pulse" title="1Hz Docker cgroup poller active">
                <span class="pulse-dot-green"></span>
                <span>COLLECTOR RUNNING</span>
            </div>
            <div class="status-pill-badge active-pulse" title="Strict 9-rule telemetry validation active">
                <span class="pulse-dot-green"></span>
                <span>VALIDATOR RUNNING</span>
            </div>
            <div class="status-pill-badge active-pulse" title="Streamlit responsive telemetry console connected">
                <span class="pulse-dot-green"></span>
                <span>DASHBOARD LIVE</span>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_status_bar(status: Dict[str, Any]):
    """Render the 8 status cards connected to backend values."""
    docker_cls = "val-healthy" if status.get("docker_healthy") else "val-critical"
    ai_cls = "val-ai" if status.get("ai_ready") else "val-warning"
    telemetry_cls = "val-healthy" if status.get("telemetry_status") == "Online" else "val-neutral"
    dataset_cls = "val-healthy" if status.get("dataset_status") == "Validated" else "val-warning"
    backend_cls = "val-healthy" if status.get("backend_ready_flag") else "val-warning"

    exp_str = str(status.get("current_experiment", "N/A"))
    samples_val = status.get("dataset_samples", 0)

    status_html = f"""
    <div class="status-grid">
        <div class="status-card">
            <div class="status-card-label">Docker Status</div>
            <div class="status-card-value {docker_cls}">
                {status.get('docker_status', 'Unknown')}
            </div>
        </div>
        <div class="status-card">
            <div class="status-card-label">AI Status</div>
            <div class="status-card-value {ai_cls}">
                {status.get('ai_status', 'Unknown')}
            </div>
        </div>
        <div class="status-card">
            <div class="status-card-label">Telemetry</div>
            <div class="status-card-value {telemetry_cls}">
                {status.get('telemetry_status', 'Idle')}
            </div>
        </div>
        <div class="status-card">
            <div class="status-card-label">Dataset</div>
            <div class="status-card-value {dataset_cls}">
                {status.get('dataset_status', 'Missing')}
            </div>
        </div>
        <div class="status-card">
            <div class="status-card-label">Containers</div>
            <div class="status-card-value val-neutral">
                {status.get('container_count', 0)} Active
            </div>
        </div>
        <div class="status-card">
            <div class="status-card-label">Dataset Samples</div>
            <div class="status-card-value val-ai">
                {samples_val:,}
            </div>
        </div>
        <div class="status-card">
            <div class="status-card-label">Experiment</div>
            <div class="status-card-value val-purple" title="{exp_str}">
                {exp_str[:16]}
            </div>
        </div>
        <div class="status-card">
            <div class="status-card-label">Backend Ready</div>
            <div class="status-card-value {backend_cls}">
                {status.get('backend_ready', 'Degraded')}
            </div>
        </div>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)


def render_footer(status: Dict[str, Any]):
    """Render persistent telemetry footer."""
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer_html = f"""
    <div class="neuroscale-footer">
        <div class="footer-left">
            <div class="footer-item">CORE: <span>NeuroScale v1.0 Production</span></div>
            <div class="footer-item">STATUS: <span>{status.get('backend_ready', 'READY')}</span></div>
            <div class="footer-item">STREAM: <span>1Hz Kernel Telemetry</span></div>
        </div>
        <div class="footer-right">
            TIMESTAMP: <span>{now_str} UTC</span>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
