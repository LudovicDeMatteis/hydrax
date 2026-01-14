from __future__ import annotations
from datetime import datetime
from pathlib import Path

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
    """
    Generates a 2-panel plot inside the specific log directory.
    
    Args:
        metrics: Dictionary containing time, cost, convergence, and breakdown keys.
        log_dir: The timestamped directory path returned by setup_log().
    """
    # Ensure log_dir is a Path object
    log_dir = Path(log_dir)
    
    # --- 1. DATA EXTRACTION ---
    times = np.array(metrics["time"])
    
    # Robustly get Total Cost
    if "cost" in metrics:
        total_cost = np.array(metrics["cost"])
    else:
        total_cost = np.array(metrics.get("best_cost", np.zeros_like(times)))

    # Robustly get Convergence
    if "cem_convergence" in metrics:
        convergence = np.array(metrics["cem_convergence"])
    else:
        convergence = np.zeros_like(times)

    # Extract Individual Cost Components (looking for keys like "costs/0/value")
    cost_keys = sorted([k for k in metrics.keys() if k.startswith("costs/") and k.endswith("/value")])
    
    if cost_keys:
        cost_components = np.vstack([metrics[k] for k in cost_keys])
        cost_labels = [f"Cost Type {k.split('/')[1]}" for k in cost_keys]
    else:
        cost_components = None

    # --- 2. PLOTTING ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # --- PANEL 1: CONVERGENCE (Dual Axis) ---
    # Left Axis: Total Cost
    color_cost = 'tab:red'
    ax1.set_ylabel('Total Best Cost', color=color_cost, fontweight='bold')
    ln1 = ax1.plot(times, total_cost, color=color_cost, label='Total Cost', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_cost)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Right Axis: Convergence (Sigma/Covariance)
    ax2 = ax1.twinx()
    color_conv = 'tab:blue'
    ax2.set_ylabel('Convergence (Sigma)', color=color_conv, fontweight='bold')
    ln2 = ax2.plot(times, convergence, color=color_conv, linestyle='--', label='Sigma / Cov', linewidth=1.5)
    ax2.tick_params(axis='y', labelcolor=color_conv)
    ax2.grid(False) 

    # Combined Legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', frameon=True)
    ax1.set_title("Algorithm Convergence & Performance", fontsize=14, fontweight='bold')

    # --- PANEL 2: COST BREAKDOWN (Stacked Area) ---
    if cost_components is not None and cost_components.shape[0] > 0:
        # Generate distinct colors
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cost_labels)))
        
        ax3.stackplot(times, cost_components, labels=cost_labels, colors=colors, alpha=0.85)
        
        ax3.set_ylabel('Cost Composition', fontweight='bold')
        ax3.legend(loc='upper right', frameon=True, title="Components")
        ax3.set_title("Cumulative Cost Breakdown", fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No individual cost components found", 
                 ha='center', va='center', transform=ax3.transAxes)

    ax3.set_xlabel('Simulation Time (s)', fontsize=12, fontweight='bold')
    ax3.set_xlim(times[0], times[-1])

    plt.tight_layout()
    
    # --- 3. SAVING ---
    # Save directly into the timestamped folder
    plot_filename = "solver_metrics.png"
    plot_path = log_dir / plot_filename
    
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    
    print(f"Graph saved to: {plot_path}")