import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Helper Function to Plot Scree & Cumulative ---
def plot_pca_figures(dataset_name, X_scaled, n_components_limit=None, threshold=0.95):
    # Fit PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Get Variance Info
    exp_var = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)
    
    n_components = len(exp_var)
    x_labels = range(1, n_components + 1)
    
    # If there are too many components (like BC=30), maybe limit x-axis for readability
    # For the report, showing all 30 might be crowded, but let's stick to standard practice.
    # We will limit x-axis tick labels if > 15
    
    # --- Figure A: Scree Plot ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(x_labels, exp_var, color='navy', alpha=0.5, label='Individual Variance')
    plt.plot(x_labels, exp_var, color='red', marker='o', linewidth=2)
    
    # Add text labels on top of bars (only for first few to avoid clutter)
    for i in range(min(10, n_components)): 
        plt.text(i+1, exp_var[i]+0.5, f'{exp_var[i]:.1f}%', ha='center', fontweight='bold')

    plt.title(f'Scree Plot of the {dataset_name} Dataset')
    plt.xlabel('Principal Component (PC)')
    plt.ylabel('Explained Variance (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x_labels)
    if n_components > 15: # Rotate if too many
        plt.xticks(rotation=90)
    plt.show() 
    # To save: plt.savefig(f'{dataset_name}_Scree_Plot.png')

    # --- Figure B: Cumulative Variance Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, cum_var, color='black', marker='o', linewidth=2, label='Cumulative Variance')
    
    # 95% Cut-off Line
    plt.axhline(y=threshold*100, color='gray', linestyle='--', linewidth=2, label=f'{threshold*100}% Threshold')
    
    # Find where it crosses threshold
    k = np.argmax(cum_var >= threshold*100) + 1
    plt.text(1, threshold*100 + 2, f'{threshold*100}% cut-off threshold', color='green', fontsize=14)

    plt.title(f'Cumulative Explained Variance Plot of the {dataset_name} Dataset')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(x_labels)
    if n_components > 15:
        plt.xticks(rotation=90)
    plt.legend()
    plt.show()
    # To save: plt.savefig(f'{dataset_name}_Cumulative_Plot.png')

# --- 1. Load & Process Breast Cancer (BC) Dataset ---
data_bc = load_breast_cancer()
scaler = StandardScaler()
X_bc_scaled = scaler.fit_transform(data_bc.data)

print("Generating Figures for BC Dataset...")
# Note: BC has 30 components.
plot_pca_figures("Breast Cancer (BC)", X_bc_scaled)

# --- 2. Load & Process Wine Recognition (WR) Dataset ---
data_wine = load_wine()
X_wine_scaled = scaler.fit_transform(data_wine.data)

print("Generating Figures for WR Dataset...")
plot_pca_figures("Wine Recognition (WR)", X_wine_scaled)