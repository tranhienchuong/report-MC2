import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# 1. Load & Prepare Data
data = load_breast_cancer()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.data)

# 2. Apply PCA
pca = PCA() # Keep all components
X_pca = pca.fit_transform(X_scaled)

# 3. Create the Table (DataFrame)
columns = [f'PC {i+1}' for i in range(X_pca.shape[1])]
df_pca_table = pd.DataFrame(X_pca, columns=columns)

# 4. Show first 5 rows (The table content)
print(df_pca_table.head())

# 5. Save to Excel/CSV if needed for report
# df_pca_table.head().to_csv("BC_PCA_Table.csv")