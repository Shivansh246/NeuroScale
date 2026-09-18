"""Generate publication-ready SVG plots from existing data/real_docker_demo.jsonl

Using Python standard library (no third-party graphics dependencies).
"""

import json
import os
import sys

DATA_FILE = "data/real_docker_demo.jsonl"
OUT_DIR = "docs/figures"


def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.", file=sys.stderr)
        sys.exit(1)
    with open(DATA_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def create_svg(width, height, content):
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
{content}
</svg>"""


def plot_cpu_trajectory(records, out_path):
    W, H = 800, 420
    pad_left, pad_right, pad_top, pad_bot = 60, 40, 50, 60
    plot_w = W - pad_left - pad_right
    plot_h = H - pad_top - pad_bot

    n = len(records)
    max_cpu = 120.0  # max y

    def get_x(i):
        return pad_left + (i / (n - 1)) * plot_w

    def get_y(val):
        v = max(0.0, min(val, max_cpu))
        return pad_top + plot_h - (v / max_cpu) * plot_h

    # Phase background bands
    # Phases: idle (0..13), cpu_spike (14..21), mem_spike (22..29), recovery (30..41)
    phases = [
        ("Warmup & Normal", 0, 13, "#f8f9fa", "#6c757d"),
        ("CPU Spike", 14, 21, "#fee2e2", "#dc2626"),
        ("Memory Spike", 22, 29, "#fef3c7", "#d97706"),
        ("Recovery", 30, 41, "#e0f2fe", "#0284c7"),
    ]

    elements = []

    # Title
    elements.append(
        f'<text x="{W/2}" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">Real Docker Demonstration: CPU Tracking &amp; Quota Allocation</text>'
    )

    # Phase bands
    for name, start_i, end_i, bg, border in phases:
        x1 = get_x(start_i)
        x2 = get_x(end_i)
        elements.append(
            f'<rect x="{x1}" y="{pad_top}" width="{x2-x1}" height="{plot_h}" fill="{bg}" opacity="0.8"/>'
        )
        elements.append(
            f'<text x="{(x1+x2)/2}" y="{pad_top+18}" font-size="11" font-weight="600" text-anchor="middle" fill="{border}">{name}</text>'
        )

    # Gridlines
    for y_val in [0, 25, 50, 75, 100]:
        y = get_y(y_val)
        elements.append(
            f'<line x1="{pad_left}" y1="{y}" x2="{W-pad_right}" y2="{y}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4"/>'
        )
        elements.append(
            f'<text x="{pad_left-8}" y="{y+4}" font-size="11" text-anchor="end" fill="#64748b">{y_val}%</text>'
        )

    # Secondary axis for allocated cores: 1.0C = 50%, 2.0C = 100%
    # Actual CPU line
    pts_actual = []
    pts_pred = []
    pts_alloc = []
    for i, r in enumerate(records):
        x = get_x(i)
        cpu_act = r.get("current_cpu", 0.0)
        pts_actual.append(f"{x:.1f},{get_y(cpu_act):.1f}")

        cpu_pred = r.get("predicted_cpu")
        if cpu_pred is not None:
            pts_pred.append(f"{x:.1f},{get_y(cpu_pred):.1f}")

        # Map 1.0 Core -> 50% relative, 2.0 Cores -> 100%
        cores = r.get("current_cpu_alloc", 1.0)
        core_pct = cores * 50.0
        pts_alloc.append(f"{x:.1f},{get_y(core_pct):.1f}")

    # Step line for allocation
    alloc_path_parts = []
    for i in range(len(records)):
        x1 = get_x(i)
        x2 = get_x(i + 1) if i + 1 < len(records) else x1
        cores = records[i].get("current_cpu_alloc", 1.0) * 50.0
        y = get_y(cores)
        if i == 0:
            alloc_path_parts.append(f"M {x1:.1f} {y:.1f}")
        alloc_path_parts.append(f"L {x2:.1f} {y:.1f}")
    elements.append(
        f'<path d="{" ".join(alloc_path_parts)}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="6,3"/>'
    )

    # Actual line
    elements.append(
        f'<path d="M {" L ".join(pts_actual)}" fill="none" stroke="#dc2626" stroke-width="2.5"/>'
    )

    # Pred line
    if pts_pred:
        elements.append(
            f'<path d="M {" L ".join(pts_pred)}" fill="none" stroke="#7c3aed" stroke-width="2" stroke-dasharray="3,3"/>'
        )

    # X-axis
    elements.append(
        f'<line x1="{pad_left}" y1="{pad_top+plot_h}" x2="{W-pad_right}" y2="{pad_top+plot_h}" stroke="#475569" stroke-width="1.5"/>'
    )
    elements.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+plot_h}" stroke="#475569" stroke-width="1.5"/>'
    )

    for i in range(0, n, 5):
        x = get_x(i)
        elements.append(
            f'<line x1="{x}" y1="{pad_top+plot_h}" x2="{x}" y2="{pad_top+plot_h+5}" stroke="#475569" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x}" y="{pad_top+plot_h+18}" font-size="11" text-anchor="middle" fill="#64748b">C{records[i]["cycle_number"]}</text>'
        )

    # X label
    elements.append(
        f'<text x="{pad_left + plot_w/2}" y="{H-15}" font-size="12" font-weight="600" text-anchor="middle" fill="#334155">Control Cycle Number</text>'
    )
    # Y label
    elements.append(
        f'<text x="18" y="{pad_top + plot_h/2}" font-size="12" font-weight="600" text-anchor="middle" fill="#334155" transform="rotate(-90 18 {pad_top + plot_h/2})">CPU Utilization (%) / Quota</text>'
    )

    # Legend
    leg_y = H - 38
    elements.append(
        f'<line x1="{pad_left+20}" y1="{leg_y}" x2="{pad_left+50}" y2="{leg_y}" stroke="#dc2626" stroke-width="2.5"/>'
    )
    elements.append(
        f'<text x="{pad_left+55}" y="{leg_y+4}" font-size="11" fill="#1e293b">Actual CPU %</text>'
    )

    elements.append(
        f'<line x1="{pad_left+160}" y1="{leg_y}" x2="{pad_left+190}" y2="{leg_y}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="3,3"/>'
    )
    elements.append(
        f'<text x="{pad_left+195}" y="{leg_y+4}" font-size="11" fill="#1e293b">Transformer Predicted CPU %</text>'
    )

    elements.append(
        f'<line x1="{pad_left+400}" y1="{leg_y}" x2="{pad_left+430}" y2="{leg_y}" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="6,3"/>'
    )
    elements.append(
        f'<text x="{pad_left+435}" y="{leg_y+4}" font-size="11" fill="#1e293b">Allocated Quota (1.0C = 50%, 2.0C = 100%)</text>'
    )

    with open(out_path, "w") as f:
        f.write(create_svg(W, H, "\n".join(elements)))
    print(f"Generated: {out_path}")


def plot_memory_trajectory(records, out_path):
    W, H = 800, 420
    pad_left, pad_right, pad_top, pad_bot = 65, 40, 50, 60
    plot_w = W - pad_left - pad_right
    plot_h = H - pad_top - pad_bot

    n = len(records)
    max_mem = 1100.0  # max y MB

    def get_x(i):
        return pad_left + (i / (n - 1)) * plot_w

    def get_y(val):
        v = max(0.0, min(val, max_mem))
        return pad_top + plot_h - (v / max_mem) * plot_h

    phases = [
        ("Warmup & Normal", 0, 13, "#f8f9fa", "#6c757d"),
        ("CPU Spike", 14, 21, "#fee2e2", "#dc2626"),
        ("Memory Spike", 22, 29, "#fef3c7", "#d97706"),
        ("Recovery", 30, 41, "#e0f2fe", "#0284c7"),
    ]

    elements = []

    # Title
    elements.append(
        f'<text x="{W/2}" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">Real Docker Demonstration: Memory Tracking &amp; Limit Allocation</text>'
    )

    # Phase bands
    for name, start_i, end_i, bg, border in phases:
        x1 = get_x(start_i)
        x2 = get_x(end_i)
        elements.append(
            f'<rect x="{x1}" y="{pad_top}" width="{x2-x1}" height="{plot_h}" fill="{bg}" opacity="0.8"/>'
        )
        elements.append(
            f'<text x="{(x1+x2)/2}" y="{pad_top+18}" font-size="11" font-weight="600" text-anchor="middle" fill="{border}">{name}</text>'
        )

    # Gridlines
    for y_val in [0, 256, 512, 768, 1024]:
        y = get_y(y_val)
        elements.append(
            f'<line x1="{pad_left}" y1="{y}" x2="{W-pad_right}" y2="{y}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4"/>'
        )
        elements.append(
            f'<text x="{pad_left-8}" y="{y+4}" font-size="11" text-anchor="end" fill="#64748b">{y_val} MB</text>'
        )

    pts_actual = []
    pts_pred = []
    alloc_path_parts = []

    for i, r in enumerate(records):
        x = get_x(i)
        mem_act = r.get("current_memory", 0.0)
        pts_actual.append(f"{x:.1f},{get_y(mem_act):.1f}")

        mem_pred = r.get("predicted_memory")
        if mem_pred is not None:
            pts_pred.append(f"{x:.1f},{get_y(mem_pred):.1f}")

        x2 = get_x(i + 1) if i + 1 < len(records) else x
        alloc_mem = r.get("current_mem_alloc", 256.0)
        y = get_y(alloc_mem)
        if i == 0:
            alloc_path_parts.append(f"M {x:.1f} {y:.1f}")
        alloc_path_parts.append(f"L {x2:.1f} {y:.1f}")

    elements.append(
        f'<path d="{" ".join(alloc_path_parts)}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="6,3"/>'
    )
    elements.append(
        f'<path d="M {" L ".join(pts_actual)}" fill="none" stroke="#059669" stroke-width="2.5"/>'
    )

    if pts_pred:
        elements.append(
            f'<path d="M {" L ".join(pts_pred)}" fill="none" stroke="#7c3aed" stroke-width="2" stroke-dasharray="3,3"/>'
        )

    # Axes
    elements.append(
        f'<line x1="{pad_left}" y1="{pad_top+plot_h}" x2="{W-pad_right}" y2="{pad_top+plot_h}" stroke="#475569" stroke-width="1.5"/>'
    )
    elements.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+plot_h}" stroke="#475569" stroke-width="1.5"/>'
    )

    for i in range(0, n, 5):
        x = get_x(i)
        elements.append(
            f'<line x1="{x}" y1="{pad_top+plot_h}" x2="{x}" y2="{pad_top+plot_h+5}" stroke="#475569" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x}" y="{pad_top+plot_h+18}" font-size="11" text-anchor="middle" fill="#64748b">C{records[i]["cycle_number"]}</text>'
        )

    elements.append(
        f'<text x="{pad_left + plot_w/2}" y="{H-15}" font-size="12" font-weight="600" text-anchor="middle" fill="#334155">Control Cycle Number</text>'
    )
    elements.append(
        f'<text x="20" y="{pad_top + plot_h/2}" font-size="12" font-weight="600" text-anchor="middle" fill="#334155" transform="rotate(-90 20 {pad_top + plot_h/2})">Memory (MB)</text>'
    )

    # Legend
    leg_y = H - 38
    elements.append(
        f'<line x1="{pad_left+20}" y1="{leg_y}" x2="{pad_left+50}" y2="{leg_y}" stroke="#059669" stroke-width="2.5"/>'
    )
    elements.append(
        f'<text x="{pad_left+55}" y="{leg_y+4}" font-size="11" fill="#1e293b">Actual RSS Memory</text>'
    )

    elements.append(
        f'<line x1="{pad_left+190}" y1="{leg_y}" x2="{pad_left+220}" y2="{leg_y}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="3,3"/>'
    )
    elements.append(
        f'<text x="{pad_left+225}" y="{leg_y+4}" font-size="11" fill="#1e293b">Transformer Predicted Memory</text>'
    )

    elements.append(
        f'<line x1="{pad_left+420}" y1="{leg_y}" x2="{pad_left+450}" y2="{leg_y}" stroke="#2563eb" stroke-width="2.5" stroke-dasharray="6,3"/>'
    )
    elements.append(
        f'<text x="{pad_left+455}" y="{leg_y+4}" font-size="11" fill="#1e293b">Allocated Docker Limit (MB)</text>'
    )

    with open(out_path, "w") as f:
        f.write(create_svg(W, H, "\n".join(elements)))
    print(f"Generated: {out_path}")


def plot_anomaly_trajectory(records, out_path):
    W, H = 800, 360
    pad_left, pad_right, pad_top, pad_bot = 65, 40, 50, 60
    plot_w = W - pad_left - pad_right
    plot_h = H - pad_top - pad_bot

    n = len(records)
    # Log scale or clamped linear: scores go from 0 to 440
    max_score = 450.0

    def get_x(i):
        return pad_left + (i / (n - 1)) * plot_w

    def get_y(val):
        v = max(0.0, min(val, max_score))
        return pad_top + plot_h - (v / max_score) * plot_h

    phases = [
        ("Warmup & Normal", 0, 13, "#f8f9fa", "#6c757d"),
        ("CPU Spike", 14, 21, "#fee2e2", "#dc2626"),
        ("Memory Spike", 22, 29, "#fef3c7", "#d97706"),
        ("Recovery", 30, 41, "#e0f2fe", "#0284c7"),
    ]

    elements = []

    # Title
    elements.append(
        f'<text x="{W/2}" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#1e293b">Real Docker Demonstration: Autoencoder Anomaly Score</text>'
    )

    # Phase bands
    for name, start_i, end_i, bg, border in phases:
        x1 = get_x(start_i)
        x2 = get_x(end_i)
        elements.append(
            f'<rect x="{x1}" y="{pad_top}" width="{x2-x1}" height="{plot_h}" fill="{bg}" opacity="0.8"/>'
        )
        elements.append(
            f'<text x="{(x1+x2)/2}" y="{pad_top+18}" font-size="11" font-weight="600" text-anchor="middle" fill="{border}">{name}</text>'
        )

    # Gridlines
    for y_val in [0, 100, 200, 300, 400]:
        y = get_y(y_val)
        elements.append(
            f'<line x1="{pad_left}" y1="{y}" x2="{W-pad_right}" y2="{y}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4"/>'
        )
        elements.append(
            f'<text x="{pad_left-8}" y="{y+4}" font-size="11" text-anchor="end" fill="#64748b">{y_val}</text>'
        )

    # Threshold line (0.003814)
    y_thresh = get_y(0.003814)
    elements.append(
        f'<line x1="{pad_left}" y1="{y_thresh}" x2="{W-pad_right}" y2="{y_thresh}" stroke="#e11d48" stroke-width="1.5" stroke-dasharray="2,2"/>'
    )

    pts_score = []
    for i, r in enumerate(records):
        x = get_x(i)
        score = r.get("anomaly_score", 0.0) or 0.0
        pts_score.append(f"{x:.1f},{get_y(score):.1f}")

    elements.append(
        f'<path d="M {" L ".join(pts_score)}" fill="none" stroke="#d97706" stroke-width="2.5"/>'
    )

    # Add markers for active anomaly cycles
    for i, r in enumerate(records):
        if r.get("is_anomaly"):
            x = get_x(i)
            score = r.get("anomaly_score", 0.0) or 0.0
            y = get_y(score)
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#dc2626"/>'
            )

    # Axes
    elements.append(
        f'<line x1="{pad_left}" y1="{pad_top+plot_h}" x2="{W-pad_right}" y2="{pad_top+plot_h}" stroke="#475569" stroke-width="1.5"/>'
    )
    elements.append(
        f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+plot_h}" stroke="#475569" stroke-width="1.5"/>'
    )

    for i in range(0, n, 5):
        x = get_x(i)
        elements.append(
            f'<line x1="{x}" y1="{pad_top+plot_h}" x2="{x}" y2="{pad_top+plot_h+5}" stroke="#475569" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x}" y="{pad_top+plot_h+18}" font-size="11" text-anchor="middle" fill="#64748b">C{records[i]["cycle_number"]}</text>'
        )

    elements.append(
        f'<text x="{pad_left + plot_w/2}" y="{H-15}" font-size="12" font-weight="600" text-anchor="middle" fill="#334155">Control Cycle Number</text>'
    )
    elements.append(
        f'<text x="20" y="{pad_top + plot_h/2}" font-size="12" font-weight="600" text-anchor="middle" fill="#334155" transform="rotate(-90 20 {pad_top + plot_h/2})">Reconstruction MSE</text>'
    )

    # Legend
    leg_y = H - 38
    elements.append(
        f'<line x1="{pad_left+20}" y1="{leg_y}" x2="{pad_left+50}" y2="{leg_y}" stroke="#d97706" stroke-width="2.5"/>'
    )
    elements.append(
        f'<text x="{pad_left+55}" y="{leg_y+4}" font-size="11" fill="#1e293b">Autoencoder Reconstruction MSE</text>'
    )

    elements.append(
        f'<circle cx="{pad_left+270}" cy="{leg_y}" r="3" fill="#dc2626"/>'
    )
    elements.append(
        f'<text x="{pad_left+280}" y="{leg_y+4}" font-size="11" fill="#1e293b">Flagged Anomaly (`is_anomaly=True`)</text>'
    )

    elements.append(
        f'<line x1="{pad_left+490}" y1="{leg_y}" x2="{pad_left+520}" y2="{leg_y}" stroke="#e11d48" stroke-width="1.5" stroke-dasharray="2,2"/>'
    )
    elements.append(
        f'<text x="{pad_left+525}" y="{leg_y+4}" font-size="11" fill="#1e293b">Threshold (0.003814)</text>'
    )

    with open(out_path, "w") as f:
        f.write(create_svg(W, H, "\n".join(elements)))
    print(f"Generated: {out_path}")


def main():
    records = load_data()
    print(f"Loaded {len(records)} records from {DATA_FILE}")
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_cpu_trajectory(records, os.path.join(OUT_DIR, "cpu_trajectory.svg"))
    plot_memory_trajectory(
        records, os.path.join(OUT_DIR, "memory_trajectory.svg")
    )
    plot_anomaly_trajectory(
        records, os.path.join(OUT_DIR, "anomaly_trajectory.svg")
    )


if __name__ == "__main__":
    main()
