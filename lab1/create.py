import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_wine

# --- Helper Function to Create 2D Plot ---
def create_2d_plot(dataset_name, data_object, abbr):
    # 1. Prepare Data (Step 2 logic)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_object.data)
    
    # 2. Apply PCA
    pca = PCA(n_components=2) # We only need the first 2 for visualization
    X_pca = pca.fit_transform(X_scaled)
    
    # 3. Create DataFrame (The Table)
    df_pca = pd.DataFrame(X_pca, columns=['PC 1', 'PC 2'])
    
    # 4. Generate the Plot (Step 3 logic)
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    scatter = plt.scatter(df_pca['PC 1'], 
                          df_pca['PC 2'], 
                          c=data_object.target, 
                          cmap='viridis', 
                          alpha=0.7, 
                          edgecolor='k')
    
    # Labels and Title
    var_1 = pca.explained_variance_ratio_[0] * 100
    var_2 = pca.explained_variance_ratio_[1] * 100
    plt.xlabel(f'Principal Component 1 ({var_1:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({var_2:.2f}%)')
    plt.title(f'Data Distribution in 2D ({dataset_name})')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Legend
    # specific handling for dataset class names
    try:
        class_names = data_object.target_names
    except AttributeError:
        class_names = [f"Class {i}" for i in set(data_object.target)]
        
    handles, _ = scatter.legend_elements()
    plt.legend(handles, class_names, title="Classes")
    
    # Save the figure
    filename = f"{abbr}_2D_Distribution.png"
    plt.savefig(filename)
    print(f"Success: Saved {filename}")
    plt.show()

# --- Run for Breast Cancer Dataset ---
print("Generating plot for Breast Cancer...")
data_bc = load_breast_cancer()
create_2d_plot("Breast Cancer Dataset", data_bc, "BC")

# --- Run for Wine Dataset ---
print("\nGenerating plot for Wine Dataset...")
data_wine = load_wine()
create_2d_plot("Wine Recognition Dataset", data_wine, "WR")