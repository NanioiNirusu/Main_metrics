import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#############################################
# DATA LOADING AND PREPARATION
#############################################

# Load the dataset
csv_path = "J:/Masters/Datasets/ALL_results/AGIQA-3_results/no_reference_metrics_classic.csv"
df = pd.read_csv(csv_path)

#############################################
# FEATURE DETECTION AND CATEGORIZATION
#############################################

# Detect numeric columns automatically
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
target_col = "mos_quality"

if target_col not in numeric_cols:
    raise ValueError(f"Target column '{target_col}' not found or not numeric.")

#############################################
# DATA PREPROCESSING
#############################################

# Create numeric dataframe and remove rows with NaN values
df_numeric = df[numeric_cols].dropna()

# Columns to exclude from PCA/SVD
excluded_cols = [
    "Image", "prompt", "adj1", "adj2", "style",
    "mos_quality", "std_quality", "mos_align", "std_align"
]

# Our target y
y = df_numeric[target_col]

# Our features X (excluding user-provided/evaluation cols)
X = df_numeric.drop(columns=excluded_cols, errors='ignore')
print("Columns used for PCA/SVD:", X.columns.tolist())

#############################################
# FEATURE SCALING
#############################################

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_names = X.columns
n_samples, n_features = X_scaled.shape

#############################################
# PCA ANALYSIS
#############################################

# Perform Principal Component Analysis
pca = PCA()
X_pca = pca.fit_transform(X_scaled)               # (n_samples x n_features)
variance_ratios = pca.explained_variance_ratio_   # length = n_features

# Per-sample PCA scores
# Each row => one sample; each column => PC coordinate
loadings_pca = pca.components_  # shape = (n_components, n_features)
# Each row => one principal component; each column => feature's loading

# Rename each principal component by the feature of largest absolute loading
pca_names = []
for i, pc_vector in enumerate(loadings_pca):
    idx = np.argmax(np.abs(pc_vector))
    strongest_feat = feature_names[idx]
    sign_label = "neg" if pc_vector[idx] < 0 else "pos"
    pca_name = f"PC{i+1}_{strongest_feat}_{sign_label}"
    pca_names.append(pca_name)

# Create DataFrame of PCA scores (per-sample)
X_pca_df = pd.DataFrame(X_pca, columns=pca_names)
# Optionally add mos_quality or ID columns for reference
X_pca_df["mos_quality"] = y.values

# Create DataFrame for PCA loadings (per-feature)
# Rows = PCs, Columns = original features
pca_loadings_df = pd.DataFrame(
    loadings_pca,
    index=pca_names,           # row names = PC1, PC2, ...
    columns=feature_names      # column names = the original features
)

# Save both to CSV
X_pca_df.to_csv("pca_scores.csv", index=False)
pca_loadings_df.to_csv("pca_loadings.csv", index=True)
print("Saved PCA scores to 'pca_scores.csv' and PCA loadings to 'pca_loadings.csv'.")

# Correlate each PCA component with mos_quality
corr_with_mos = []
for i, pc_name in enumerate(pca_names):
    corr_val = np.corrcoef(X_pca_df[pc_name], y)[0, 1]
    corr_with_mos.append((i, pc_name, corr_val))

corr_with_mos_sorted = sorted(corr_with_mos, key=lambda x: abs(x[2]), reverse=True)
pc_idx_1, pc_name_1, corr1 = corr_with_mos_sorted[0]
pc_idx_2, pc_name_2, corr2 = corr_with_mos_sorted[1]

#############################################
# SVD ANALYSIS
#############################################

# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
# X_scaled = U * S * Vt

X_svd = U @ np.diag(S)  # (n_samples x n_features)

variance_svd = (S**2) / (n_samples - 1)
variance_ratio_svd = variance_svd / np.sum(variance_svd)

# Per-sample SVD scores
# Each row => one sample; each column => SVD component coordinate
X_svd_df = pd.DataFrame(X_svd, columns=[f"SVD-PC{i+1}" for i in range(X_svd.shape[1])])
X_svd_df["mos_quality"] = y.values

# Per-feature SVD loadings
# Vt is (n_components x n_features).
# Rows = SVD component, columns = features.
svd_names = []
for i, svd_vector in enumerate(Vt):
    idx = np.argmax(np.abs(svd_vector))
    strongest_feat = feature_names[idx]
    sign_label = "neg" if svd_vector[idx] < 0 else "pos"
    svd_name = f"SVD_PC{i+1}_{strongest_feat}_{sign_label}"
    svd_names.append(svd_name)

# We'll rename Vt's row indices with these new names for clarity
svd_loadings_df = pd.DataFrame(
    Vt,
    index=svd_names,
    columns=feature_names
)

# Save both to CSV
X_svd_df.to_csv("svd_scores.csv", index=False)
svd_loadings_df.to_csv("svd_loadings.csv", index=True)
print("Saved SVD scores to 'svd_scores.csv' and SVD loadings to 'svd_loadings.csv'.")

# Correlate each SVD component with mos_quality
corr_svd_list = []
for i, comp_name in enumerate(X_svd_df.columns[:-1]):  # skip the last "mos_quality" col
    corr_val = np.corrcoef(X_svd_df[comp_name], y)[0, 1]
    corr_svd_list.append((i, comp_name, corr_val))

corr_svd_sorted = sorted(corr_svd_list, key=lambda x: abs(x[2]), reverse=True)
svd_idx_1, svd_name_1, svd_corr1 = corr_svd_sorted[0]
svd_idx_2, svd_name_2, svd_corr2 = corr_svd_sorted[1]

#############################################
# REGRESSION ANALYSIS
#############################################

# We'll drop the "mos_quality" column from X_pca_df
X_pca_for_reg = X_pca_df.drop(columns=["mos_quality"], errors='ignore')
X_train, X_test, y_train, y_test = train_test_split(X_pca_for_reg, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2_val = r2_score(y_test, y_pred)

#############################################
# VISUALIZATION
#############################################

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) PCA: PC1 vs PC2
if X_pca_df.shape[1] > 2:  # we have at least 2 PCs plus 'mos_quality'
    pc1_label = pca_names[0]
    pc2_label = pca_names[1]
    sc1 = axes[0,0].scatter(
        X_pca_df[pc1_label], X_pca_df[pc2_label],
        c=y, cmap='viridis', alpha=0.7
    )
    axes[0,0].set_xlabel(pc1_label)
    axes[0,0].set_ylabel(pc2_label)
    axes[0,0].set_title('PCA: PC1 vs PC2 (Highest Variance)')
    cb1 = fig.colorbar(sc1, ax=axes[0,0], fraction=0.046, pad=0.04)
    cb1.set_label(target_col)
else:
    axes[0,0].axis('off')
    axes[0,0].set_title('Not enough PCA components')

# (0,1) PCA: best-corr PCs
sc2 = axes[0,1].scatter(
    X_pca_df[pc_name_1], X_pca_df[pc_name_2],
    c=y, cmap='viridis', alpha=0.7
)
axes[0,1].set_xlabel(f"{pc_name_1} (corr={corr1:.3f})")
axes[0,1].set_ylabel(f"{pc_name_2} (corr={corr2:.3f})")
axes[0,1].set_title("PCA: Best Corr Components")
cb2 = fig.colorbar(sc2, ax=axes[0,1], fraction=0.046, pad=0.04)
cb2.set_label(target_col)

# (0,2) SVD: top-variance comps => columns 0 & 1
if X_svd_df.shape[1] > 2:
    svd1_label = X_svd_df.columns[0]
    svd2_label = X_svd_df.columns[1]
    sc3 = axes[0,2].scatter(
        X_svd_df[svd1_label], X_svd_df[svd2_label],
        c=y, cmap='viridis', alpha=0.7
    )
    axes[0,2].set_xlabel(svd1_label)
    axes[0,2].set_ylabel(svd2_label)
    axes[0,2].set_title("SVD: Comp1 vs Comp2 (Highest Variance)")
    cb3 = fig.colorbar(sc3, ax=axes[0,2], fraction=0.046, pad=0.04)
    cb3.set_label(target_col)
else:
    axes[0,2].axis('off')
    axes[0,2].set_title('Not enough SVD components')

# (1,0) SVD: best-corr comps
best_svd1_label = X_svd_df.columns[svd_idx_1]
best_svd2_label = X_svd_df.columns[svd_idx_2]
sc4 = axes[1,0].scatter(
    X_svd_df[best_svd1_label], X_svd_df[best_svd2_label],
    c=y, cmap='viridis', alpha=0.7
)
axes[1,0].set_xlabel(f"{best_svd1_label} (corr={svd_corr1:.3f})")
axes[1,0].set_ylabel(f"{best_svd2_label} (corr={svd_corr2:.3f})")
axes[1,0].set_title("SVD: Best Corr Components")
cb4 = fig.colorbar(sc4, ax=axes[1,0], fraction=0.046, pad=0.04)
cb4.set_label(target_col)

# (1,1) NIQE vs BRISQUE
if {"NIQE", "BRISQUE"}.issubset(X.columns):
    sc5 = axes[1,1].scatter(X["BRISQUE"], X["NIQE"], c=y, cmap='viridis', alpha=0.7)
    axes[1,1].set_xlabel("BRISQUE (raw)")
    axes[1,1].set_ylabel("NIQE (raw)")
    axes[1,1].set_title("NIQE vs BRISQUE (colored by mos_quality)")
    cb5 = fig.colorbar(sc5, ax=axes[1,1], fraction=0.046, pad=0.04)
    cb5.set_label(target_col)
else:
    axes[1,1].axis('off')
    axes[1,1].set_title('No BRISQUE / NIQE columns')

# (1,2) summary text
summary_text = (
    f"PCA Var Ratios (1st two): {variance_ratios[0]:.3f}, {variance_ratios[1]:.3f}\n"
    f"SVD Var Ratios (1st two): {variance_ratio_svd[0]:.3f}, {variance_ratio_svd[1]:.3f}\n\n"
    f"Highest corr (PCA): {pc_name_1} (corr={corr1:.3f})\n"
    f"2nd highest corr (PCA): {pc_name_2} (corr={corr2:.3f})\n\n"
    f"Highest corr (SVD): {svd_name_1} (corr={svd_corr1:.3f})\n"
    f"2nd highest corr (SVD): {svd_name_2} (corr={svd_corr2:.3f})\n\n"
    f"Regression on PCA components -> R^2: {r2_val:.3f}\n\n"
    "Var Ratios = fraction of total variance in X explained by each PC.\n"
    "We've saved sample-level and feature-level data to CSV."
)
axes[1,2].axis('off')
axes[1,2].text(0.01, 0.5, summary_text, fontsize=12, verticalalignment='center')

plt.suptitle("Comprehensive PCA & SVD (Sample & Feature) Logging", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#############################################
# RESULTS LOGGING
#############################################

print("\n--- PCA Expl. Variance Ratios ---")
print(variance_ratios)
print("\n--- PCA Highest Correlation with mos_quality ---")
print(f"{pc_name_1} (corr={corr1:.3f})")
print(f"{pc_name_2} (corr={corr2:.3f})")

print("\n--- SVD Expl. Variance Ratios ---")
print(variance_ratio_svd)
print("\n--- SVD Highest Correlation with mos_quality ---")
print(f"{svd_name_1} (corr={svd_corr1:.3f})")
print(f"{svd_name_2} (corr={svd_corr2:.3f})")

print(f"\nRegression on PCA components -> R^2 on test set: {r2_val:.3f}")
print("Saved these files:\n"
      "  - 'pca_scores.csv' (per-sample PCA scores)\n"
      "  - 'pca_loadings.csv' (per-feature PCA loadings)\n"
      "  - 'svd_scores.csv' (per-sample SVD scores)\n"
      "  - 'svd_loadings.csv' (per-feature SVD loadings)")
print("Done!")
