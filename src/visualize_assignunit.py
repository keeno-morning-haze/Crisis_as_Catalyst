import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib import font_manager

# Set up academic style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# Create directory for saving visualizations
visualization_dir = 'results/assignunit/visualizations_simpcot_new'
if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)

# Load data
raw_data = pd.read_csv('data/assignunit/raw_responses_assignunit_simpcot.csv')

# Get all unique model names
models = raw_data['model'].unique()

# Create visualizations for each model
for model in models:
    # Get data for current model
    model_data = raw_data[raw_data['model'] == model].copy()
    
    # Get day columns
    day_columns = [col for col in model_data.columns if col.startswith('Day')]
    
    # Calculate r_Blue: sum of day indices where values are between 1-6
    def get_r_blue(row):
        # Get days with blue units (1-6)
        blue_days = [(i+1, val) for i, val in enumerate(row[day_columns]) if 1 <= val <= 6]
        # Calculate sum of these day indices
        return sum(day_idx for day_idx, _ in blue_days)
    
    r_blue = model_data.apply(get_r_blue, axis=1)
    # Normalize using formula from paper: r_Blue_norm = 1 - 2 * ((r_Blue - 21) / 36)
    model_data['rBlue_norm'] = 1 - 2 * ((r_blue - 21) / 36)
    
    # Create boxplot with customized outlier display
    plt.figure(figsize=(8, 6))
    
    # Configure outlier appearance as cross markers
    flierprops = dict(marker='x', markerfacecolor='red', markersize=8, 
                      linestyle='none', markeredgecolor='red')
    
    # Set scenario order as ABCD
    order = ['A', 'B', 'C', 'D']
    
    # Create boxplot
    ax = sns.boxplot(x='scenario', y='rBlue_norm', data=model_data, 
                    showfliers=True, flierprops=flierprops, order=order,
                    width=0.6, linewidth=1.2)
    
    # Customize boxplot appearance for academic standards
    for patch in ax.artists:
        patch.set_edgecolor('black')
        patch.set_facecolor('white')
    
    plt.title(f'Distribution of rBlue_norm by Scenario - {model}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('rBlue_norm', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.box(True)
    
    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, f'boxplot_rBlue_norm_{model}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate A12 effect size
    def compute_a12(group1, group2):
        """
        Compute A12 effect size
        group1: first group data
        group2: second group data
        returns: A12 effect size value
        """
        m = len(group1)
        n = len(group2)
        r = 0
        for i in range(m):
            for j in range(n):
                if group1.iloc[i] > group2.iloc[j]:
                    r += 1
                elif group1.iloc[i] == group2.iloc[j]:
                    r += 0.5
        return r / (m * n)
    
    # Calculate A12 effect size for all scenario pairs
    scenarios = ['A', 'B', 'C', 'D']
    effect_sizes_data = []
    for i in range(len(scenarios)):
        for j in range(i+1, len(scenarios)):
            scenario1 = scenarios[i]
            scenario2 = scenarios[j]
            scores1 = model_data[model_data['scenario'] == scenario1]['rBlue_norm']
            scores2 = model_data[model_data['scenario'] == scenario2]['rBlue_norm']
            a12 = compute_a12(scores1, scores2)
            effect_sizes_data.append({
                'Comparison': f'{scenario1} vs {scenario2}',
                'A12': a12
            })
    
    effect_sizes = pd.DataFrame(effect_sizes_data)
    
    # Create effect size bar chart with academic formatting
    plt.figure(figsize=(8, 6))
    
    # Create bar plot
    ax = sns.barplot(x='Comparison', y='A12', data=effect_sizes, 
                    edgecolor='black', linewidth=1.2)
    
    # Add value annotations above each bar
    for i, v in enumerate(effect_sizes['A12']):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.title(f'A12 Effect Size Between Scenarios - {model}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Scenario Comparison', fontsize=12)
    plt.ylabel('A12 Effect Size', fontsize=12)
    plt.ylim(0, 1.05)  # Adjust y-axis to accommodate annotations
    plt.axhline(0.5, ls='--', color='red', alpha=0.7, 
                label='No Effect (A12=0.5)')
    plt.legend(frameon=True, fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.box(True)
    
    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, f'effect_sizes_{model}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
