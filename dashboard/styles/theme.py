"""
NeuroScale Theme & CSS Styling System.
Provides an authentic Datadog/Grafana-inspired dark theme for the frontend,
complete with glowing radar pulses, container cards, animated pipeline badges, and responsive grids.
"""

DARK_THEME_CSS = """
<style>
/* -------------------------------------------------------------
   NeuroScale Design System: Datadog / Grafana Inspired Dark Theme
------------------------------------------------------------- */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0b0f19;
    --bg-secondary: #111827;
    --bg-card: #161f30;
    --bg-card-hover: #1c283f;
    --border-color: #26334d;
    --border-subtle: #1e293b;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    
    /* Semantic Colors */
    --color-healthy: #10b981;    /* Emerald Green */
    --color-ai: #38bdf8;         /* Cyan/Blue */
    --color-warning: #f59e0b;    /* Amber/Yellow */
    --color-critical: #ef4444;   /* Crimson Red */
    --color-purple: #a855f7;     /* Purple */
}

/* Global Typography & Background */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: var(--text-primary);
}

.main {
    background-color: var(--bg-primary);
}

/* Top Navigation / App Header */
.neuroscale-header {
    background: linear-gradient(180deg, #162032 0%, #0f172a 100%);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.header-title-box {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-badge {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
    color: white;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 13px;
    padding: 6px 10px;
    border-radius: 6px;
    letter-spacing: 0.5px;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
}

.brand-name {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}

.brand-tagline {
    font-size: 12px;
    font-weight: 400;
    color: var(--text-secondary);
    margin: 0;
}

/* Multi-Indicator Status Pill Row */
.header-indicators-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.status-pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #10b981;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    padding: 4px 10px;
    border-radius: 9999px;
    transition: all 0.2s ease;
}

.status-pill-badge.active-pulse {
    border-color: rgba(16, 185, 129, 0.5);
    background: rgba(16, 185, 129, 0.12);
}

.pulse-dot-green {
    width: 7px;
    height: 7px;
    background-color: var(--color-healthy);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--color-healthy);
    animation: pulse-green 1.8s infinite;
}

.pulse-dot-cyan {
    width: 7px;
    height: 7px;
    background-color: var(--color-ai);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--color-ai);
    animation: pulse-cyan 1.8s infinite;
}

@keyframes pulse-green {
    0% { transform: scale(0.9); opacity: 0.7; box-shadow: 0 0 2px #10b981; }
    50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 10px #10b981; }
    100% { transform: scale(0.9); opacity: 0.7; box-shadow: 0 0 2px #10b981; }
}

@keyframes pulse-cyan {
    0% { transform: scale(0.9); opacity: 0.7; box-shadow: 0 0 2px #38bdf8; }
    50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 10px #38bdf8; }
    100% { transform: scale(0.9); opacity: 0.7; box-shadow: 0 0 2px #38bdf8; }
}

/* Status Cards Grid */
.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}

.status-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 12px 14px;
    transition: all 0.2s ease-in-out;
}

.status-card:hover {
    background-color: var(--bg-card-hover);
    border-color: #3b82f6;
    transform: translateY(-2px);
}

.status-card-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 4px;
}

.status-card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 700;
    line-height: 1.2;
    display: flex;
    align-items: center;
    gap: 6px;
}

.val-healthy { color: var(--color-healthy); }
.val-ai { color: var(--color-ai); }
.val-warning { color: var(--color-warning); }
.val-critical { color: var(--color-critical); }
.val-purple { color: var(--color-purple); }
.val-neutral { color: var(--text-primary); }

/* Main Content Page Container */
.page-container {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 22px;
    margin-top: 8px;
    min-height: 480px;
}

/* Container Highlight Cards */
.container-card {
    background: #161f30;
    border: 1px solid #26334d;
    border-radius: 10px;
    padding: 16px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    margin-bottom: 12px;
}

.container-card:hover {
    border-color: #38bdf8;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(56, 189, 248, 0.15);
}

.container-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.container-name-text {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 14px;
    color: #38bdf8;
}

.container-id-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #64748b;
    background: #0f172a;
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid #1e293b;
}

.container-metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
    font-size: 12px;
    margin-top: 10px;
    background: #0f172a;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #1e293b;
}

.metric-item-label {
    color: #94a3b8;
    font-size: 11px;
}

.metric-item-val {
    color: #f8fafc;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* Flashing Recommendation Banner */
.recommendation-flash-box {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(56, 189, 248, 0.15) 100%);
    border: 1px solid #10b981;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    animation: flash-pulse 2s infinite ease-in-out;
}

@keyframes flash-pulse {
    0% { border-color: rgba(16, 185, 129, 0.3); box-shadow: 0 0 0 rgba(16, 185, 129, 0); }
    50% { border-color: rgba(16, 185, 129, 1.0); box-shadow: 0 0 12px rgba(16, 185, 129, 0.35); }
    100% { border-color: rgba(16, 185, 129, 0.3); box-shadow: 0 0 0 rgba(16, 185, 129, 0); }
}

/* Footer Container */
.neuroscale-footer {
    border-top: 1px solid var(--border-subtle);
    padding: 16px 8px;
    margin-top: 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
}
</style>
"""
