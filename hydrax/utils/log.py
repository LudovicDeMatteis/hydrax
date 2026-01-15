from __future__ import annotations
from datetime import datetime
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

import matplotlib.pyplot as plt
import numpy as np
import os


def setup_log(experiment_name: str, base_dir: str = "logs") -> Path:
    """
    Creates a hierarchical log directory: base_dir / experiment_name / timestamp
    
    Args:
        experiment_name: Name of the current experiment.
        base_dir: The root directory for all logs (default: "logs").
        
    Returns:
        Path: The path to the newly created specific run directory.
    """
    # 1. Get current timestamp (Year-Month-Day_Hour-Minute-Second)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 2. Construct the full path
    log_path = Path(base_dir) / experiment_name / timestamp
    
    # 3. Create the directory
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 4. Return the absolute path
    return log_path.resolve()


def plot_solver_metrics(metrics: dict, log_dir: Path | str):
    log_dir = Path(log_dir)
    
    # --- 1. DATA EXTRACTION ---
    times = np.array(metrics["time"])
    
    # Extract components
    cost_keys = sorted([k for k in metrics.keys() if k.startswith("costs/") and k.endswith("/value")])
    
    cost_components = np.vstack([metrics[k] for k in cost_keys]) # Shape (N_costs, N_times)
    cost_labels = [f"{k.split('/')[1]}" for k in cost_keys]
    total_cost = np.sum(cost_components, axis=0)
   
    if "cem_convergence" in metrics:
        convergence = np.array(metrics["cem_convergence"])
    else:
        convergence = np.zeros_like(times)

    # --- 2. PLOTTING ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # We do NOT sharey here, because components are usually much smaller than the Total
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- PANEL 1: TOTAL COST (Sum) ---
    color_cost = 'tab:red'
    ax1.plot(times, total_cost, color=color_cost, label='Total Cost (Sum)', linewidth=2)
    ax1.set_ylabel('Total Cost', color=color_cost, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_cost)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Right Axis: Convergence
    ax2 = ax1.twinx()
    color_conv = 'tab:blue'
    ax2.plot(times, convergence, color=color_conv, linestyle='--', label='Sigma', linewidth=1.5)
    ax2.set_ylabel('Convergence (Sigma)', color=color_conv, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_conv)
    ax2.grid(False)
    
    # Legend Panel 1
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title("Total Cost & Convergence", fontsize=14, fontweight='bold')

    # --- PANEL 2: INDIVIDUAL COST LINES ---
    colors = plt.cm.tab10(np.linspace(0, 1, len(cost_labels)))
    
    # Loop through components and plot separate lines
    for i, label in enumerate(cost_labels):
        ax3.plot(times, cost_components[i], label=label, color=colors[i], linewidth=2, alpha=0.8)
    
    ax3.set_ylabel('Individual Cost Value', fontweight='bold')
    ax3.legend(loc='upper right', frameon=True, title="Cost Terms")
    ax3.set_title("Breakdown by Component (Non-Stacked)", fontsize=12)
    
    # Prevent scientific notation offset (e.g. +1e5)
    ax3.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

    ax3.set_xlabel('Simulation Time (s)', fontsize=12, fontweight='bold')
    ax3.set_xlim(times[0], times[-1])

    plt.tight_layout()
    
    # --- 3. SAVING ---
    plot_path = log_dir / "solver_metrics.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Graph saved to: {plot_path}")