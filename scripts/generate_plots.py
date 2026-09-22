import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    lines = open("experiments/closed_loop_evaluation/results/real_docker_results.jsonl", "r").readlines()
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except:
            pass
            
    active_records = [r for r in records if r.get("status") == "active"]
    os.makedirs("experiments/closed_loop_evaluation/results/plots", exist_ok=True)
    
    time_arr = np.arange(len(active_records))
    
    # 1. CPU prediction
    c_cpu = [r["current_cpu"] for r in active_records]
    p_cpu = [r["predicted_cpu"] for r in active_records]
    plt.figure()
    plt.plot(time_arr, c_cpu, label="Actual")
    plt.plot(time_arr, p_cpu, label="Predicted")
    plt.legend()
    plt.title("CPU Utilization vs Predicted")
    plt.savefig("experiments/closed_loop_evaluation/results/plots/1_cpu_prediction.png")
    
    # 2. Memory prediction
    c_mem = [r["current_memory"] for r in active_records]
    p_mem = [r["predicted_memory"] for r in active_records]
    plt.figure()
    plt.plot(time_arr, c_mem, label="Actual")
    plt.plot(time_arr, p_mem, label="Predicted")
    plt.legend()
    plt.title("Memory Utilization vs Predicted")
    plt.savefig("experiments/closed_loop_evaluation/results/plots/2_memory_prediction.png")
    
    # 3. CPU allocation
    a_cpu = [r["current_cpu_alloc"] for r in active_records]
    plt.figure()
    plt.plot(time_arr, c_cpu, label="Actual Util")
    plt.plot(time_arr, np.array(a_cpu)*100, label="Allocated (x100%)", linestyle="--")
    plt.legend()
    plt.title("CPU Allocation")
    plt.savefig("experiments/closed_loop_evaluation/results/plots/3_cpu_allocation.png")
    
    # 4. Memory allocation
    a_mem = [r["current_mem_alloc"] for r in active_records]
    plt.figure()
    plt.plot(time_arr, c_mem, label="Actual Util")
    plt.plot(time_arr, a_mem, label="Allocated", linestyle="--")
    plt.legend()
    plt.title("Memory Allocation")
    plt.savefig("experiments/closed_loop_evaluation/results/plots/4_memory_allocation.png")
    
    # 5. Anomaly score
    ano = [r["anomaly_score"] for r in active_records]
    plt.figure()
    plt.plot(time_arr, ano, color="red")
    plt.title("Anomaly Score")
    plt.yscale("log")
    plt.savefig("experiments/closed_loop_evaluation/results/plots/5_anomaly_score.png")
    
    # 6. Prediction confidence
    conf = [r["prediction_confidence"] for r in active_records]
    plt.figure()
    plt.plot(time_arr, conf, color="orange")
    plt.title("Prediction Confidence")
    plt.ylim(0, 1.1)
    plt.savefig("experiments/closed_loop_evaluation/results/plots/6_prediction_confidence.png")
    
    # 7. SLA latency
    sla = [r.get("sla_latency", 0.0) for r in active_records]
    plt.figure()
    plt.plot(time_arr, sla, color="purple")
    plt.axhline(50.0, color='r', linestyle='--', label='Threshold')
    plt.legend()
    plt.title("SLA Latency")
    plt.savefig("experiments/closed_loop_evaluation/results/plots/7_sla_latency.png")
    
    # 8. Reward
    rwd = [r["reward"] for r in active_records]
    plt.figure()
    plt.plot(time_arr, rwd, color="green")
    plt.title("RL Reward")
    plt.savefig("experiments/closed_loop_evaluation/results/plots/8_reward.png")

if __name__ == "__main__":
    main()
