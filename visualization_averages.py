import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#############################################
# UTILITY FUNCTIONS
#############################################

# Function to determine if higher or lower values are better for each metric
def get_metric_direction(metric_name):
    # Metrics where higher values are better
    if metric_name in ['FR-PSNR', 'SSIM', 'NR-SNR', 'NR-PSNR', 'RankinAware', 'LIQE',
                       'Laplacian Variance', 'Tenengrad', 'Modified Laplacian',
                       'RMS Contrast', 'Michelson Contrast', 'Colorfulness',
                       'Histogram Entropy', 'Histogram Spread', 'GLCM Energy',
                       'GLCM Homogeneity', 'Resolution']:
        return 'higher'
    # Metrics where lower values are better
    elif metric_name in ['BRISQUE', 'NIQE', 'LPIPS', 'DISTS', 'RAPIQUE',
                         'Edge Spread', 'MSE', 'Edge Density', 'GLCM Contrast']:
        return 'lower'
    # For GLCM Correlation, values closer to 1 are better
    elif metric_name == 'GLCM Correlation':
        return 'closer_to_one'
    else:
        return 'unknown'

#############################################
# DATA LOADING AND PREPARATION
#############################################

# Load the dataset
data = pd.read_excel('Averages_visualization.xlsx', index_col=0)
data = data.apply(pd.to_numeric, errors='coerce')

#############################################
# DATA NORMALIZATION
#############################################

# Create a normalized version of the data for better comparison
normalized_data = pd.DataFrame(index=data.index, columns=data.columns)

for col in data.columns:
    if pd.isna(data[col]).all():
        continue  # Skip columns with all NaN values

    direction = get_metric_direction(col)
    non_na_values = data[col].dropna()

    if len(non_na_values) == 0:
        continue

    min_val = non_na_values.min()
    max_val = non_na_values.max()

    if min_val == max_val:
        normalized_data[col] = 0.5  # Set to middle value if all values are the same
        continue

    if direction == 'higher':
        # Normalize so higher values are better (0-1 scale, higher is better)
        normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
    elif direction == 'lower':
        # Invert so lower values become higher (better) on 0-1 scale
        normalized_data[col] = 1 - ((data[col] - min_val) / (max_val - min_val))
    elif direction == 'closer_to_one':
        # Normalize so values closer to 1 are better
        normalized_data[col] = 1 - abs(data[col] - 1) / max(abs(min_val - 1), abs(max_val - 1))
    else:
        # If direction unknown, use standard normalization
        normalized_data[col] = (data[col] - min_val) / (max_val - min_val)

#############################################
# VISUALIZATION
#############################################

# Create heatmap for the normalized data
plt.figure(figsize=(16, 12))
sns.heatmap(normalized_data, annot=True, cmap='coolwarm', vmin=0, vmax=1,
            fmt='.2f', linewidths=0.5)
plt.title('Normalized Metrics Values Across Datasets (Higher Values = Better Performance)', fontsize=16)
plt.tight_layout()
plt.savefig('normalized_metrics_heatmap.png')
plt.close()

#############################################
# ANALYSIS OF BEST METRICS
#############################################

# Find the best metric for each dataset
best_metrics = pd.DataFrame(index=data.index, columns=['Best_Metric', 'Original_Value', 'Normalized_Value'])

for idx in normalized_data.index:
    row_data = normalized_data.loc[idx].dropna()
    if len(row_data) > 0:
        best_col = row_data.idxmax()
        best_metrics.loc[idx, 'Best_Metric'] = best_col
        best_metrics.loc[idx, 'Original_Value'] = data.loc[idx, best_col]
        best_metrics.loc[idx, 'Normalized_Value'] = normalized_data.loc[idx, best_col]

#############################################
# RESULTS LOGGING
#############################################

# Print the best metrics for each dataset
print("Best Metrics for Each Dataset:")
print(best_metrics)

# Save the best metrics to CSV
best_metrics.to_csv('best_metrics_per_dataset.csv')

print("Normalized heatmap visualization and best metrics analysis completed.")
