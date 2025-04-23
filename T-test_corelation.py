import pandas as pd
from scipy.stats import ttest_rel, pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

#############################################
# DATA LOADING AND PREPARATION
#############################################

# Load the dataset
# df = pd.read_csv("J:/Masters/ALL_results/UHD_results/UHD_selected.csv")
# Alternative data source (commented out)
df = pd.read_excel("J:/Masters/Averages_Ttest.xlsx")

# Remove the 'Average' row from the dataset
# df = df[df['Image'] != 'Average']

# Define columns to exclude from analysis
ignore_columns = ['dataset']
# 'Image','views','favorites','downloads','comments','Evaluation Time (s)'

#############################################
# FEATURE SELECTION
#############################################

# Create a list of all features to use as reference
reference_features = [col for col in df.select_dtypes(include=['number']).columns
                      if col not in ignore_columns]

#############################################
# DATA CLEANING
#############################################

# Replace infinite values with NaN
df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

# Drop columns that are entirely NaN (if any)
df = df.dropna(axis=1, how='all')
print("DataFrame shape after dropping all-NaN columns:", df.shape)

#############################################
# INITIALIZE CORRELATION MATRICES
#############################################

# Initialize matrices for Pearson and Spearman correlations with explicit float dtype
pearson_matrix = pd.DataFrame(index=reference_features, columns=reference_features, dtype=float)
spearman_matrix = pd.DataFrame(index=reference_features, columns=reference_features, dtype=float)

# Set diagonal values to 1.0 (correlation of a feature with itself)
for feature in reference_features:
    pearson_matrix.loc[feature, feature] = 1.0
    spearman_matrix.loc[feature, feature] = 1.0

# Dictionary to store all results
all_results = {}

#############################################
# MAIN ANALYSIS LOOP
#############################################

# Loop through each reference feature
for i, reference_feature in enumerate(reference_features):
    print(f"Analyzing with reference feature ({i + 1}/{len(reference_features)}): {reference_feature}")

    # Initialize results for this reference feature
    results = {}

    #############################################
    # FEATURE COMPARISON ANALYSIS
    #############################################

    for feature in df.columns:
        # Skip ignored columns and the reference feature
        if feature in ignore_columns or feature == reference_feature:
            continue

        # Only process numeric features
        if pd.api.types.is_numeric_dtype(df[feature]):
            # Create sub-DataFrame with this feature & reference
            sub_df = df[[reference_feature, feature]].copy()

            # Replace any leftover +/- inf with NaN
            sub_df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

            # Drop rows where either the reference or the current feature is NaN
            sub_df.dropna(inplace=True)

            # If there aren't enough rows, store NaNs for stats
            if len(sub_df) < 2:
                print(f"Not enough valid data for feature '{feature}' to run statistical tests.")
                t_stat = t_p_value = pearson_corr = pearson_p = spearman_corr = spearman_p = float('nan')
            else:
                # T-test (paired)
                t_stat, t_p_value = ttest_rel(sub_df[reference_feature], sub_df[feature])
                # Pearson correlation
                pearson_corr, pearson_p = pearsonr(sub_df[reference_feature], sub_df[feature])
                # Spearman correlation
                spearman_corr, spearman_p = spearmanr(sub_df[reference_feature], sub_df[feature])

                # Store correlation values in matrices
                pearson_matrix.loc[reference_feature, feature] = float(pearson_corr)
                pearson_matrix.loc[feature, reference_feature] = float(pearson_corr)
                spearman_matrix.loc[reference_feature, feature] = float(spearman_corr)
                spearman_matrix.loc[feature, reference_feature] = float(spearman_corr)

            # Average of the feature
            avg_score = sub_df[feature].mean() if len(sub_df) > 0 else float('nan')

        else:
            # Non-numeric features -> store NaN
            t_stat = t_p_value = pearson_corr = pearson_p = spearman_corr = spearman_p = avg_score = float('nan')

        # Store results
        results[feature] = {
            'T-Stat': t_stat,
            'T-p': t_p_value,
            'Pearson': pearson_corr,
            'Pearson-p': pearson_p,
            'Spearman': spearman_corr,
            'Spearman-p': spearman_p,
            'Average': avg_score
        }

    # Store all results for this reference feature
    all_results[reference_feature] = results

    #############################################
    # SAVE INDIVIDUAL FEATURE RESULTS
    #############################################

    # Save results to a CSV file with numerical suffix
    results_df = pd.DataFrame.from_dict(results, orient='index')
    output_filename = f'feature_results_{i + 1}_{reference_feature}.csv'
    results_df.to_csv(output_filename, index=True)
    print(f"Results saved to '{output_filename}'.")

#############################################
# FINALIZE AND SAVE CORRELATION MATRICES
#############################################

# Make sure all values in the matrices are numeric
# Replace any remaining non-numeric values with NaN
pearson_matrix = pearson_matrix.apply(pd.to_numeric, errors='coerce')
spearman_matrix = spearman_matrix.apply(pd.to_numeric, errors='coerce')

# Print data types to verify
print("Pearson matrix dtypes:", pearson_matrix.dtypes.unique())
print("Spearman matrix dtypes:", spearman_matrix.dtypes.unique())

# Save the correlation matrices to CSV
pearson_matrix.to_csv('pearson_correlation_matrix.csv')
spearman_matrix.to_csv('spearman_correlation_matrix.csv')
print("Correlation matrices saved to CSV files.")

#############################################
# COLLECT AND SAVE AVERAGE SCORES
#############################################

# Collect all average scores in one DataFrame
avg_scores = {}
for ref_feature, results in all_results.items():
    avg_scores[ref_feature] = {feature: data['Average'] for feature, data in results.items()}

avg_df = pd.DataFrame.from_dict(avg_scores)
avg_df.to_csv('all_average_scores.csv', index=True)
print("All average scores saved to 'all_average_scores.csv'")

#############################################
# VISUALIZATION
#############################################

# Plot the Pearson correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(pearson_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            fmt='.2f', linewidths=0.5)
plt.title('Pearson Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('pearson_correlation_matrix.png')
plt.close()

# Plot the Spearman correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(spearman_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            fmt='.2f', linewidths=0.5)
plt.title('Spearman Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('spearman_correlation_matrix.png')
plt.close()

print("Correlation matrix plots saved.")
print("All processing and analysis complete.")
