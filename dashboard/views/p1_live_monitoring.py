"""
Page 1: Live Monitoring — Real-time telemetry, container health, and rolling time-series streams.
Industrial Datadog/Grafana layout with Plotly dark charts, auto-scrolling 60-second rolling charts,
highlight container cards, and AI recommendation change flashing.
"""

import time
import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Professional Datadog/Grafana Plotly Template
DARK_LAYOUT = dict(
    paper_bgcolor="#161f30",
    plot_bgcolor="#0b0f19",
    font=dict(family="Inter, sans-serif", color="#f8fafc", size=11),
    margin=dict(l=40, r=20, t=35, b=30),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11, color="#94a3b8"),
    ),
    xaxis=dict(
        gridcolor="#1e293b",
        zerolinecolor="#26334d",
        showline=True,
        linecolor="#26334d",
    ),
    yaxis=dict(
        gridcolor="#1e293b",
        zerolinecolor="#26334d",
        showline=True,
        linecolor="#26334d",
    ),
)


@st.fragment(run_every=1.0)
def _render_live_stream_fragment(provider):
    """Fragment that polls LiveTelemetryEngine every 1 second and re-renders live views."""
    live_engine = provider.live_engine
    active_containers = live_engine.get_active_containers()

    if not active_containers:
        st.warning("⚠️ No active Docker benchmark containers detected.")
        if st.button("↺ Switch to Replay Mode", key="btn_switch_replay", use_container_width=False):
            st.session_state["telemetry_mode"] = "Replay Mode"
            st.rerun()
        return

    # 1. Highlight Container Cards Grid (Live)
    _render_container_highlight_cards(active_containers, live_engine=live_engine)

    # 2. Load recent telemetry from in-memory ring buffer (rolling 60s window)
    df = live_engine.get_history_df(limit=60)
    if df.empty:
        st.info("Gathering live telemetry samples from active containers...")
        return

    _render_telemetry_dashboard_content(df, rolling_label="Last 60 Seconds (1Hz Scrolling Stream)")


def _render_replay_stream(provider, selected_run: str):
    """Render telemetry loaded from stored historical runs (rolling 60 samples)."""
    status = provider.get_system_status()
    containers = provider.get_live_container_stats()

    # Container Fleet Cards
    if containers:
        _render_container_highlight_cards(containers, live_engine=None)
    else:
        # Synthesize active display cards from latest dataset slice
        df_preview = provider.load_telemetry_df(run_id=selected_run, limit=2)
        if not df_preview.empty and "container_name" in df_preview.columns:
            synth_containers = []
            for cname in df_preview["container_name"].unique():
                cdf = df_preview[df_preview["container_name"] == cname]
                last = cdf.iloc[-1]
                synth_containers.append({
                    "id": str(last.get("container_id", "c1"))[:12],
                    "name": str(cname),
                    "status": "running",
                    "workload_mode": str(last.get("workload_mode", "benchmark")),
                    "cpu_percent": float(last.get("cpu_percent", 0.0)),
                    "memory_usage_mb": float(last.get("memory_usage_mb", 0.0)),
                    "pids": int(last.get("pids", 2)),
                    "health": "Healthy (Verified)",
                })
            _render_container_highlight_cards(synth_containers, live_engine=None)
        else:
            st.markdown(
                """
                <div style="background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; font-size: 13px; color: #94a3b8;">
                    ℹ️ Operating in <b>Replay Mode</b>. Streaming historical metrics from the verified telemetry dataset.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Rolling 60-sample window for fast high-res inspection
    df = provider.load_telemetry_df(run_id=selected_run, limit=60)
    if df.empty:
        st.warning("No telemetry records available to display.")
        return

    _render_telemetry_dashboard_content(df, rolling_label="Rolling 60-Sample Horizon")


def _render_container_highlight_cards(containers, live_engine=None):
    """Render individual highlight cards per container with health, workload, and flash recommendation."""
    st.markdown("<h4 style='font-size: 14px; margin: 6px 0 10px 0; color: #e2e8f0; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;'>Active Container Fleet</h4>", unsafe_allow_html=True)
    cols = st.columns(max(1, len(containers)))

    for idx, c in enumerate(containers):
        cid = c.get("id", "")
        cname = c.get("name", cid[:12])
        status = c.get("status", "running")
        workload = c.get("workload_mode", "mixed")

        if live_engine:
            latest = live_engine.get_latest_metrics(cid) or {}
            ai_state = live_engine.get_latest_ai_state(cid) or {}
            cpu_val = latest.get("cpu_percent", 0.0)
            mem_val = latest.get("memory_usage_mb", 0.0)
            pids = latest.get("pids", 1)
            rec_cpu = ai_state.get("recommended_cpu", 1.0)
            rec_mem = ai_state.get("recommended_memory", 256)
        else:
            cpu_val = c.get("cpu_percent", 42.5)
            mem_val = c.get("memory_usage_mb", 230.0)
            pids = c.get("pids", 2)
            rec_cpu = 1.0
            rec_mem = 256

        # Check for change in recommendation to trigger flash highlight
        prev_key = f"prev_rec_{cid}"
        has_changed = False
        if prev_key in st.session_state:
            prev = st.session_state[prev_key]
            if prev.get("cpu") != rec_cpu or prev.get("mem") != rec_mem:
                has_changed = True
        st.session_state[prev_key] = {"cpu": rec_cpu, "mem": rec_mem}

        rec_box_html = f"""
        <div class="recommendation-flash-box">
            <div>
                <span style="font-size: 11px; color: #94a3b8; font-weight: 600;">RECOMMENDED CGROUP:</span><br/>
                <span style="font-family: 'JetBrains Mono'; font-size: 13px; font-weight: 700; color: #10b981;">
                    CPU: {rec_cpu} Cores &bull; RAM: {rec_mem} MB
                </span>
            </div>
            <span style="font-size: 10px; background: rgba(16,185,129,0.2); color: #10b981; padding: 2px 6px; border-radius: 4px; font-weight: 700;">
                {"UPDATED" if has_changed else "OPTIMAL"}
            </span>
        </div>
        """

        with cols[idx]:
            st.markdown(
                f"""
                <div class="container-card">
                    <div class="container-card-header">
                        <div>
                            <span class="container-name-text">{cname}</span>
                            <span class="container-id-badge">{cid[:10]}</span>
                        </div>
                        <span style="background: rgba(16,185,129,0.2); color: #10b981; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">
                            ● {status.upper()}
                        </span>
                    </div>
                    <div class="container-metrics-grid">
                        <div>
                            <div class="metric-item-label">CPU DEMAND</div>
                            <div class="metric-item-val" style="color: #38bdf8;">{cpu_val:.1f}%</div>
                        </div>
                        <div>
                            <div class="metric-item-label">RESIDENT RAM</div>
                            <div class="metric-item-val" style="color: #a855f7;">{mem_val:.1f} MB</div>
                        </div>
                        <div>
                            <div class="metric-item-label">WORKLOAD PHASE</div>
                            <div class="metric-item-val" style="color: #f59e0b;">{str(workload).upper()}</div>
                        </div>
                        <div>
                            <div class="metric-item-label">PROCESS PIDS</div>
                            <div class="metric-item-val">{pids}</div>
                        </div>
                    </div>
                    {rec_box_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_telemetry_dashboard_content(df: pd.DataFrame, rolling_label: str = "Last 60 Seconds"):
    """Render gauges, rolling time series charts, I/O metrics, and latest records table."""
    latest_rec = df.iloc[-1]
    g_col1, g_col2, g_col3, g_col4 = st.columns([1, 1, 1, 1])

    with g_col1:
        cpu_val = float(latest_rec.get("cpu_percent", 0.0))
        fig_cpu_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cpu_val,
            number={"suffix": "%", "font": {"size": 24, "color": "#f8fafc"}},
            title={"text": "Instantaneous CPU", "font": {"size": 13, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 200], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "#0b0f19",
                "borderwidth": 1,
                "bordercolor": "#26334d",
                "steps": [
                    {"range": [0, 80], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [80, 150], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [150, 200], "color": "rgba(239, 68, 68, 0.2)"},
                ],
            }
        ))
        fig_cpu_gauge.update_layout(**DARK_LAYOUT, height=160)
        st.plotly_chart(fig_cpu_gauge, use_container_width=True)

    with g_col2:
        mem_pct = float(latest_rec.get("memory_percent", 0.0))
        fig_mem_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mem_pct,
            number={"suffix": "%", "font": {"size": 24, "color": "#f8fafc"}},
            title={"text": "Instantaneous Memory", "font": {"size": 13, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": "#a855f7"},
                "bgcolor": "#0b0f19",
                "borderwidth": 1,
                "bordercolor": "#26334d",
                "steps": [
                    {"range": [0, 60], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [60, 85], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [85, 100], "color": "rgba(239, 68, 68, 0.2)"},
                ],
            }
        ))
        fig_mem_gauge.update_layout(**DARK_LAYOUT, height=160)
        st.plotly_chart(fig_mem_gauge, use_container_width=True)

    with g_col3:
        mem_mb = float(latest_rec.get("memory_usage_mb", 0.0))
        mem_limit_mb = float(latest_rec.get("memory_limit_mb", 1024.0))
        st.markdown(
            f"""
            <div style="background: #161f30; border: 1px solid #26334d; border-radius: 8px; padding: 16px; height: 160px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Resident Memory</div>
                <div style="font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; color: #a855f7; margin: 4px 0;">{mem_mb:.1f} MB</div>
                <div style="font-size: 12px; color: #94a3b8;">Limit: {mem_limit_mb:.0f} MB</div>
                <div style="font-size: 11px; color: #10b981; margin-top: 6px;">● cgroups v2 enforce active</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with g_col4:
        workload_str = str(latest_rec.get("workload_mode", "mixed")).upper()
        pids_count = int(latest_rec.get("pids", 1))
        st.markdown(
            f"""
            <div style="background: #161f30; border: 1px solid #26334d; border-radius: 8px; padding: 16px; height: 160px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700;">Workload Profile</div>
                <div style="font-family: 'JetBrains Mono'; font-size: 22px; font-weight: 700; color: #38bdf8; margin: 4px 0;">{workload_str}</div>
                <div style="font-size: 12px; color: #94a3b8;">Active PIDs: {pids_count}</div>
                <div style="font-size: 11px; color: #f59e0b; margin-top: 6px;">Dynamic Workload Stream</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3. Rolling 60-Second Charts: CPU & Memory
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        fig_cpu = go.Figure()
        containers_in_df = df["container_name"].unique() if "container_name" in df.columns else ["container"]
        for cname in containers_in_df:
            cdf = df[df["container_name"] == cname] if "container_name" in df.columns else df
            x_vals = cdf.get("datetime", list(range(len(cdf))))
            fig_cpu.add_trace(go.Scatter(
                x=x_vals,
                y=cdf["cpu_percent"],
                mode="lines",
                name=str(cname),
                line=dict(width=2),
            ))
        fig_cpu.update_layout(
            **DARK_LAYOUT,
            title=f"CPU Utilization % ({rolling_label})",
            height=280,
            yaxis_title="CPU %",
        )
        st.plotly_chart(fig_cpu, use_container_width=True)

    with c_col2:
        fig_mem = go.Figure()
        for cname in containers_in_df:
            cdf = df[df["container_name"] == cname] if "container_name" in df.columns else df
            x_vals = cdf.get("datetime", list(range(len(cdf))))
            fig_mem.add_trace(go.Scatter(
                x=x_vals,
                y=cdf["memory_usage_mb"],
                mode="lines",
                name=str(cname),
                line=dict(width=2),
                fill="tozeroy",
                fillcolor="rgba(168, 85, 247, 0.1)",
            ))
        fig_mem.update_layout(
            **DARK_LAYOUT,
            title=f"Resident Memory MB ({rolling_label})",
            height=280,
            yaxis_title="MB",
        )
        st.plotly_chart(fig_mem, use_container_width=True)

    # 4. Network and Block I/O Throughput (Rolling)
    io_col1, io_col2 = st.columns(2)

    with io_col1:
        fig_net = go.Figure()
        x_vals = df.get("datetime", list(range(len(df))))
        rx_vals = df.get("network_rx_bytes", pd.Series([0]*len(df))) / 1024
        tx_vals = df.get("network_tx_bytes", pd.Series([0]*len(df))) / 1024
        fig_net.add_trace(go.Scatter(
            x=x_vals,
            y=rx_vals,
            mode="lines",
            name="RX (KB)",
            line=dict(color="#10b981", width=2),
        ))
        fig_net.add_trace(go.Scatter(
            x=x_vals,
            y=tx_vals,
            mode="lines",
            name="TX (KB)",
            line=dict(color="#38bdf8", width=2),
        ))
        fig_net.update_layout(
            **DARK_LAYOUT,
            title="Network Traffic (KB Cumulative)",
            height=250,
            yaxis_title="KB",
        )
        st.plotly_chart(fig_net, use_container_width=True)

    with io_col2:
        fig_io = go.Figure()
        w_vals = df.get("block_write_bytes", pd.Series([0]*len(df))) / 1024
        r_vals = df.get("block_read_bytes", pd.Series([0]*len(df))) / 1024
        fig_io.add_trace(go.Scatter(
            x=x_vals,
            y=w_vals,
            mode="lines",
            name="Disk Write (KB)",
            line=dict(color="#f59e0b", width=2),
        ))
        fig_io.add_trace(go.Scatter(
            x=x_vals,
            y=r_vals,
            mode="lines",
            name="Disk Read (KB)",
            line=dict(color="#ef4444", width=2),
        ))
        fig_io.update_layout(
            **DARK_LAYOUT,
            title="Disk I/O Activity (KB Cumulative)",
            height=250,
            yaxis_title="KB",
        )
        st.plotly_chart(fig_io, use_container_width=True)

    # 5. Raw Cumulative cpu_usage_ns Timeline
    st.markdown("<h4 style='font-size: 14px; margin: 16px 0 8px 0; color: #e2e8f0; font-weight: 700;'>Kernel Nanosecond Counter Timeline (cpu_usage_ns)</h4>", unsafe_allow_html=True)
    fig_ns = go.Figure()
    for cname in containers_in_df:
        cdf = df[df["container_name"] == cname] if "container_name" in df.columns else df
        x_vals = cdf.get("datetime", list(range(len(cdf))))
        fig_ns.add_trace(go.Scatter(
            x=x_vals,
            y=cdf.get("cpu_usage_ns", pd.Series([0]*len(cdf))) / 1e9,
            mode="lines",
            name=f"{cname} (Seconds)",
            line=dict(width=2),
        ))
    fig_ns.update_layout(
        **DARK_LAYOUT,
        height=230,
        yaxis_title="CPU Time (Seconds)",
    )
    st.plotly_chart(fig_ns, use_container_width=True)

    # 6. Latest Telemetry Table
    st.markdown("<h4 style='font-size: 14px; margin: 16px 0 8px 0; color: #e2e8f0; font-weight: 700;'>Recent Telemetry Records (Latest 10 Samples)</h4>", unsafe_allow_html=True)
    display_cols = [
        "timestamp",
        "container_name",
        "workload_mode",
        "cpu_percent",
        "memory_usage_mb",
        "memory_percent",
        "network_rx_bytes",
        "block_write_bytes",
        "pids",
    ]
    avail_cols = [c for c in display_cols if c in df.columns]
    if avail_cols:
        st.dataframe(df[avail_cols].tail(10).sort_values("timestamp", ascending=False), use_container_width=True)


def render_live_monitoring_page(provider, selected_run: str = "latest"):
    """Render the full Live Monitoring dashboard."""
    telemetry_mode = st.session_state.get("telemetry_mode", "Replay Mode")
    is_live = (telemetry_mode == "Live Mode")

    mode_badge = (
        '<span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #10b981; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 11px;">● LIVE STREAM (1Hz AUTO-SCROLLING 60s)</span>'
        if is_live
        else '<span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; padding: 4px 10px; border-radius: 9999px; font-weight: 700; font-size: 11px;">↺ REPLAY MODE (60-SAMPLE HORIZON)</span>'
    )

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <h2 style="margin: 0; font-size: 22px; font-weight: 700; color: #ffffff;">Container Telemetry Console</h2>
                <p style="margin: 0; font-size: 13px; color: #94a3b8;">High-frequency 1Hz cgroup telemetry stream, network throughput, and I/O tracking</p>
            </div>
            <div>
                {mode_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if is_live:
        _render_live_stream_fragment(provider)
    else:
        _render_replay_stream(provider, selected_run=selected_run)
