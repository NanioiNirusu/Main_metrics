import pandas as pd
from scipy.stats import ttest_rel, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

#############################################
# DATA LOADING AND PREPARATION
#############################################

# Load the Spearman correlation matrix
# Note: Commented code shows alternative loading for Pearson matrix
# pearson_matrix = pd.read_csv("J:/Masters/ALL_results/_DatasetSpecific/Groupings/Real/Real_pearson.csv", index_col=0)
spearman_matrix = pd.read_csv("J:/Masters/ALL_results/_DatasetSpecific/Groupings/Real/Real_spearman.csv", index_col=0)

#############################################
# DATA CLEANING
#############################################

# Convert all values to numeric, replacing non-numeric with NaN
# pearson_matrix = pearson_matrix.apply(pd.to_numeric, errors='coerce')
spearman_matrix = spearman_matrix.apply(pd.to_numeric, errors='coerce')

#############################################
# VISUALIZATION
#############################################

# Plot the Pearson correlation matrix (currently commented out)
# plt.figure(figsize=(14, 12))
# sns.heatmap(pearson_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
#             fmt='.2f', linewidths=0.5)
# plt.title('Pearson Correlation Matrix', fontsize=16)
# plt.tight_layout()
# plt.savefig('pearson_correlation_matrix.png')
# plt.close()

# Plot the Spearman correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(spearman_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            fmt='.2f', linewidths=0.5)
plt.title('Spearman Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('spearman_correlation_matrix.png')
plt.close()

#############################################
# RESULTS LOGGING
#############################################

print("Correlation matrix plots saved.")
print("All processing and analysis complete.")
