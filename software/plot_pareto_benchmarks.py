import os
import matplotlib.pyplot as plt
import numpy as np

def plot_pareto_and_benchmarks(output_dir="docs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Performance & Energy Efficiency Comparison (Table I of Paper)
    platforms = ['CPU\n(Xeon 5218)', 'Edge-GPU\n(Xavier NX)', 'ASIC 40nm\n(GCNAX)', 'FPGA (DSP)\n(LW-GCN)', 'FP8GCN (DSP-Free)\n(Our Design)']
    
    # Cora Speedup relative to CPU
    speedup_cora = [1.0, 1.01, 9.0, 46.0, 31.0]
    speedup_citeseer = [1.0, 2.0, 16.0, 60.0, 44.0]
    speedup_pubmed = [1.0, 6.0, 7.0, 22.0, 20.0]
    
    # Energy Efficiency (graphs / kJ)
    energy_eff_cora = [3.57e4 / 3.57e4, 3.57e4 / 3.57e4 * 8, 265, 704, 712]
    energy_eff_citeseer = [1.0, 2.0, 433, 913, 1010]
    energy_eff_pubmed = [1.0, 5.0, 194, 334, 450]
    
    x = np.arange(len(platforms))
    width = 0.25
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Chart 1: Speedup
    rects1 = ax1.bar(x - width, speedup_cora, width, label='Cora', color='#2E75B6')
    rects2 = ax1.bar(x, speedup_citeseer, width, label='Citeseer', color='#ED7D31')
    rects3 = ax1.bar(x + width, speedup_pubmed, width, label='Pubmed', color='#70AD47')
    
    ax1.set_ylabel('Speedup relative to CPU (x)', fontsize=11, fontweight='bold')
    ax1.set_title('Inference Speedup across Platforms', fontsize=12, fontweight='bold', color='#1F4E78')
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms, fontsize=9)
    ax1.legend(frameon=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Chart 2: Energy Efficiency
    r1 = ax2.bar(x - width, energy_eff_cora, width, label='Cora', color='#2E75B6')
    r2 = ax2.bar(x, energy_eff_citeseer, width, label='Citeseer', color='#ED7D31')
    r3 = ax2.bar(x + width, energy_eff_pubmed, width, label='Pubmed', color='#70AD47')
    
    ax2.set_ylabel('Energy Efficiency Improvement (x CPU)', fontsize=11, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_title('Energy Efficiency Improvement (Log Scale)', fontsize=12, fontweight='bold', color='#1F4E78')
    ax2.set_xticks(x)
    ax2.set_xticklabels(platforms, fontsize=9)
    ax2.legend(frameon=True)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    comp_path = os.path.join(output_dir, 'performance_energy_comparison.png')
    plt.savefig(comp_path)
    plt.close()
    print(f"  [+] Saved benchmark chart: {comp_path}")
    
    # 2. Pareto Frontier: Accuracy Loss vs PPA (PDPres)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Multiplier configurations: (Accuracy Loss %, Relative PDPres cost)
    configs = {
        'FP8GCN (L-Mul E4M3) [Ours]': (0.8, 4.8, '#1F4E78', '*', 180),
        'FP8 Exact': (0.2, 5.8, '#2E75B6', 'o', 100),
        'INT8 DRUM-4': (2.0, 7.5, '#ED7D31', 's', 90),
        'INT8 Mitchell': (3.6, 7.2, '#70AD47', '^', 90),
        'mul8s_1L12': (9.5, 4.6, '#C00000', 'd', 90),
        'DyRecMul': (8.2, 7.8, '#7030A0', 'v', 90),
        'Ullah et al.': (6.5, 7.6, '#00B0F0', 'p', 90)
    }
    
    for name, (acc_loss, pdp, color, marker, size) in configs.items():
        ax.scatter(acc_loss, pdp, color=color, marker=marker, s=size, label=name, edgecolors='black', linewidth=1.2, zorder=5)
        ax.annotate(name.split(' ')[0], (acc_loss + 0.2, pdp + 0.1), fontsize=9)
        
    # Draw Pareto boundary line
    pareto_x = [0.2, 0.8, 9.5]
    pareto_y = [5.8, 4.8, 4.6]
    ax.plot(pareto_x, pareto_y, linestyle='--', color='#1F4E78', alpha=0.8, linewidth=2, label='Pareto Frontier')
    
    ax.set_xlabel('Accuracy Loss relative to FP32 (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Resource-Aware Cost Metric (PDPres = Power x CPD x Area)', fontsize=11, fontweight='bold')
    ax.set_title('Pareto Frontier: Accuracy Loss vs. Hardware Cost', fontsize=12, fontweight='bold', color='#1F4E78')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    
    plt.tight_layout()
    pareto_path = os.path.join(output_dir, 'pareto_frontier.png')
    plt.savefig(pareto_path)
    plt.close()
    print(f"  [+] Saved Pareto chart: {pareto_path}")

if __name__ == "__main__":
    plot_pareto_and_benchmarks()
