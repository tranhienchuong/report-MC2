import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import davies_bouldin_score
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# Import UCI Repository Library
from ucimlrepo import fetch_ucirepo

# --- GLOBAL SETTINGS FOR PROFESSIONAL PLOTS ---
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'sans-serif'
sns.set_style("whitegrid")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def plot_elbow_and_dbi(data, k_range, title_prefix, fig_num):
    """
    Plots Elbow Method and DBI Side-by-Side.
    """
    inertias = []
    dbi_scores = []
    
    print(f"[{title_prefix}] Calculating Elbow & DBI for K={k_range}...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=3, random_state=42)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        if k > 1:
            dbi_scores.append(davies_bouldin_score(data, labels))
        else:
            dbi_scores.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Elbow
    ax1.plot(k_range, inertias, 'bo-', markersize=8)
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (WCSS)')
    ax1.set_title(f'Elbow Method')
    ax1.grid(True)
    
    # Plot DBI
    ax2.plot(k_range[1:], dbi_scores[1:], 'rs-', markersize=8)
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Davies-Bouldin Index')
    ax2.set_title(f'Davies-Bouldin Index')
    ax2.grid(True)
    
    plt.suptitle(f"{fig_num}: Evaluation for {title_prefix}", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_2d_pca(X_pca, labels, title, centroids_pca=None):
    """
    Scatter plot for 2D PCA results with centroids.
    """
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = sns.color_palette("tab10", len(unique_labels))
    
    for i, lbl in enumerate(unique_labels):
        plt.scatter(X_pca[labels == lbl, 0], X_pca[labels == lbl, 1], 
                    s=40, alpha=0.6, label=f'Cluster {lbl}', color=colors[i])
        
    if centroids_pca is not None:
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    s=200, c='black', marker='X', label='Centroids', edgecolors='white')
        
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.legend()
    plt.show()

# =============================================================================
# PART 1: LAND MINE DATASET (REAL DATA FROM UCI)
# =============================================================================
print("\n" + "="*50)
print("PART 1: LAND MINE DATASET (REAL DATA)")
print("="*50)

# 1. Fetch Data
print("Fetching Land Mines dataset (ID: 763)...")
land_mines = fetch_ucirepo(id=763)
X_lm = land_mines.data.features.values

# 2. Preprocess
scaler_lm = StandardScaler()
X_lm_scaled = scaler_lm.fit_transform(X_lm)
pca_lm = PCA(n_components=2)
X_lm_pca = pca_lm.fit_transform(X_lm_scaled)

# --- Figure 1: Elbow & DBI ---
plot_elbow_and_dbi(X_lm_scaled, range(1, 11), "Land Mine Dataset", "Figure 1")

# --- Figure 2: Initialization Comparison ---
print("Generating Figure 2: Centroid Initialization...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Random
km_rnd = KMeans(n_clusters=5, init='random', n_init=1, max_iter=1, random_state=10)
km_rnd.fit(X_lm_scaled)
c_rnd = pca_lm.transform(km_rnd.cluster_centers_)
ax1.scatter(X_lm_pca[:,0], X_lm_pca[:,1], c='gray', alpha=0.3)
ax1.scatter(c_rnd[:,0], c_rnd[:,1], c='red', s=200, marker='X', label='Random ICs')
ax1.set_title("Random Initialization")
ax1.legend()

# K-Means++
km_pp = KMeans(n_clusters=5, init='k-means++', n_init=1, max_iter=1, random_state=10)
km_pp.fit(X_lm_scaled)
c_pp = pca_lm.transform(km_pp.cluster_centers_)
ax2.scatter(X_lm_pca[:,0], X_lm_pca[:,1], c='gray', alpha=0.3)
ax2.scatter(c_pp[:,0], c_pp[:,1], c='blue', s=200, marker='X', label='K-Means++ ICs')
ax2.set_title("K-Means++ Initialization")
ax2.legend()
plt.suptitle("Figure 2: Comparison of Initial Centroids")
plt.show()

# --- Figure 3 & 4: Iterations ---
for i, itr in zip([3, 4], [6, 15]):
    km = KMeans(n_clusters=5, init='k-means++', max_iter=itr, n_init=1, random_state=42)
    lbl = km.fit_predict(X_lm_scaled)
    cnt = pca_lm.transform(km.cluster_centers_)
    plot_2d_pca(X_lm_pca, lbl, f"Figure {i}: K-Means with {itr} Iterations", cnt)

# --- Figure 5: 3D Visualization ---
print("Generating Figure 5: 3D Visualization...")
pca_3d = PCA(n_components=3)
X_lm_3d = pca_3d.fit_transform(X_lm_scaled)
km_final = KMeans(n_clusters=5, random_state=42, n_init=10)
lbl_final = km_final.fit_predict(X_lm_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(X_lm_3d[:,0], X_lm_3d[:,1], X_lm_3d[:,2], c=lbl_final, cmap='viridis', s=20)
ax.set_title("Figure 5: 3D Clustering Visualization")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
plt.legend(*sc.legend_elements(), title="Clusters")
plt.show()

# --- Figure 6 & 7: Different K ---
for k, fig_n in zip([4, 6], [6, 7]):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_lm_scaled)
    cnt = pca_lm.transform(km.cluster_centers_)
    plot_2d_pca(X_lm_pca, lbl, f"Figure {fig_n}: Clustering with K={k}", cnt)


# =============================================================================
# PART 2: GAS TURBINE DATASET (REAL DATA FROM UCI)
# =============================================================================
print("\n" + "="*50)
print("PART 2: GAS TURBINE DATASET (REAL DATA)")
print("="*50)

# 1. Fetch Data
print("Fetching Gas Turbine dataset (ID: 551)...")
gas_turbine = fetch_ucirepo(id=551)
df_gt = gas_turbine.data.features

# Drop 'year' column if it exists (as per report)
if 'year' in df_gt.columns:
    print("Dropping 'year' column...")
    df_gt = df_gt.drop(columns=['year'])

X_gt = df_gt.values

# 2. Preprocess
scaler_gt = StandardScaler()
X_gt_scaled = scaler_gt.fit_transform(X_gt)
pca_gt = PCA(n_components=2)
X_gt_pca = pca_gt.fit_transform(X_gt_scaled)

# --- Figure 8: Elbow & DBI ---
plot_elbow_and_dbi(X_gt_scaled, range(2, 10), "Gas Turbine", "Figure 8")

# --- Figure 9, 10, 11: Visualizations (K=4, 2, 6) ---
scenarios = [(4, 9), (2, 10), (6, 11)]
for k, fig_n in scenarios:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_gt_scaled)
    cnt = pca_gt.transform(km.cluster_centers_)
    plot_2d_pca(X_gt_pca, lbl, f"Figure {fig_n}: Clustering with K={k}", cnt)


# =============================================================================
# PART 3: N-BaIoT DATASET (SIMULATED DATA)
# =============================================================================
print("\n" + "="*50)
print("PART 3: N-BaIoT DATASET (SIMULATED)")
print("="*50)

# 1. Simulate Data (Because real data is >7GB)
print("Simulating large dataset (N-BaIoT structure)...")
# Simulating 20k samples, 115 features, 3 centers
X_iot, _ = make_blobs(n_samples=20000, n_features=115, centers=3, random_state=123)
scaler_iot = StandardScaler()
X_iot_scaled = scaler_iot.fit_transform(X_iot)

# --- Figure 12: Scree Plot (Incremental PCA) ---
print("Running Incremental PCA...")
ipca = IncrementalPCA(n_components=10)
ipca.fit(X_iot_scaled)
exp_var = ipca.explained_variance_ratio_ * 100
cum_var = np.cumsum(exp_var)

plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), exp_var, color='silver', label='Individual Variance')
plt.plot(range(1, 11), cum_var, 'k-o', label='Cumulative Variance')
for i, v in enumerate(exp_var):
    plt.text(i+1, v+0.5, f"{v:.1f}%", ha='center', fontsize=9)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Figure 12: Scree Plot (N-BaIoT)')
plt.legend()
plt.show()

# --- Figure 13: 2D Distribution ---
X_iot_pca = ipca.transform(X_iot_scaled)[:, :2]
plt.figure(figsize=(10, 7))
plt.scatter(X_iot_pca[:, 0], X_iot_pca[:, 1], c='gray', alpha=0.5, s=10)
plt.title('Figure 13: Data Distribution (2D PCA)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()

# --- Figure 14: Choosing K (Entire Simulated Set) ---
# Taking a subset for speed in Elbow/DBI calculation
plot_elbow_and_dbi(X_iot_scaled[:5000], range(2, 7), "Entire Dataset (Subset)", "Figure 14")

# --- Figure 15: K-Means (K=3) on Entire ---
mbk = MiniBatchKMeans(n_clusters=3, batch_size=2048, random_state=42, n_init='auto')
lbl_iot = mbk.fit_predict(X_iot_scaled)
cnt_iot = ipca.transform(mbk.cluster_centers_)[:, :2]
plot_2d_pca(X_iot_pca, lbl_iot, "Figure 15: K-Means on Entire Dataset (K=3)", cnt_iot)

# --- Figure 16 & 17: 10% Subspace ---
print("Processing 10% Random Subspace...")
idx = np.random.choice(len(X_iot_scaled), int(len(X_iot_scaled)*0.1), replace=False)
X_sub = X_iot_scaled[idx]
X_sub_pca = X_iot_pca[idx]

# Fig 16: Choosing K for Subspace
plot_elbow_and_dbi(X_sub, range(2, 7), "10% Subspace", "Figure 16")

# Fig 17: Clustering (K=4) on Subspace
km_sub = KMeans(n_clusters=4, random_state=42, n_init=10)
lbl_sub = km_sub.fit_predict(X_sub)
cnt_sub = ipca.transform(km_sub.cluster_centers_)[:, :2]
plot_2d_pca(X_sub_pca, lbl_sub, "Figure 17: K-Means on 10% Subspace (K=4)", cnt_sub)

print("\nAll figures generated successfully.")