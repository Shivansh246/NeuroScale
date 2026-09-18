import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="NeuroScale Dashboard", layout="wide")

st.title("NeuroScale Autonomous Controller Dashboard")

def load_jsonl(path):
    data = []
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return pd.DataFrame(data)

available_sources = [
    p for p in ["data/real_docker_demo.jsonl", "data/control_loop.jsonl"] if os.path.exists(p)
]
if not available_sources:
    available_sources = ["data/control_loop.jsonl"]

control_path = st.sidebar.selectbox("Control Loop Data File", available_sources)
eval_path = "data/evaluation_results.jsonl"

st.sidebar.header("Data Sources")
if st.sidebar.button("Reload Data"):
    st.rerun()

st.header("A. Live / Control Loop")
df_control = load_jsonl(control_path)

if not df_control.empty:
    st.dataframe(df_control.tail(10))
    
    st.header("B. Prediction Performance")
    if 'current_cpu' in df_control.columns and 'predicted_cpu' in df_control.columns:
        st.line_chart(df_control[['current_cpu', 'predicted_cpu']])
        st.line_chart(df_control[['current_memory', 'predicted_memory']])
        
    st.header("C. Anomaly Detection")
    if 'anomaly_score' in df_control.columns:
        st.line_chart(df_control['anomaly_score'])
        if 'is_anomaly' in df_control.columns:
            st.write(f"Total Anomalies Detected: {df_control['is_anomaly'].sum()}")
        
    st.header("D. Resource Control & DQN Decisions")
    alloc_cpu_col = 'selected_cpu_allocation' if 'selected_cpu_allocation' in df_control.columns else 'current_cpu_alloc'
    alloc_mem_col = 'selected_memory_allocation' if 'selected_memory_allocation' in df_control.columns else 'current_mem_alloc'

    if alloc_cpu_col in df_control.columns and 'current_cpu' in df_control.columns:
        st.line_chart(df_control[[alloc_cpu_col, 'current_cpu']])
    if alloc_mem_col in df_control.columns and 'current_memory' in df_control.columns:
        st.line_chart(df_control[[alloc_mem_col, 'current_memory']])

    st.header("E. SLA Performance")
    if 'sla_latency' in df_control.columns:
        st.line_chart(df_control['sla_latency'])
        if 'sla_violation' in df_control.columns:
            st.write(f"Total SLA Violations: {df_control['sla_violation'].sum()}")
else:
    st.warning("No control loop data found.")

st.header("E. Strategy Comparison")
df_eval = load_jsonl(eval_path)

if not df_eval.empty:
    # aggregate by strategy
    summary = df_eval.groupby('strategy')[['cpu_wastage', 'sla_violations', 'avg_cpu_allocation', 'reward']].mean()
    st.dataframe(summary)
    
    st.bar_chart(summary['reward'])
    st.bar_chart(summary['sla_violations'])
    
    st.header("F. Experiment Summary")
    st.write(f"Total evaluation records: {len(df_eval)}")
    st.dataframe(df_eval)
else:
    st.warning("No evaluation data found.")
