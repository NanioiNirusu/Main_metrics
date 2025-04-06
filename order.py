import pandas as pd

#############################################
# DATA LOADING AND PREPARATION
#############################################

# Read the Excel file with numbers
df = pd.read_excel("J:/Masters/Datasets/Koniq10K/koniqe_results/raw/order.xlsx")
original_numbers = df.iloc[:, 0].astype(str).tolist()  # Convert first column to list of strings

# Read the CSV file with modifications
df_modified = pd.read_csv("J:/Masters/Datasets/Koniq10K/koniq10k_scores_and_distributions/koniq10k_scores_and_distributions.csv")

print("Available columns in modified file:", df_modified.columns.tolist())

#############################################
# DATA CLEANING AND TRANSFORMATION
#############################################

# Assuming the first column in your CSV contains the file numbers
# Rename it to 'file_number' for consistency
df_modified = df_modified.rename(columns={df_modified.columns[0]: 'file_number'})

# Convert to dictionary for faster lookup
existing_data = df_modified.set_index('file_number').to_dict('index')

#############################################
# DATA MERGING
#############################################

# Create new merged data
merged_data = []
for number in original_numbers:
    base_entry = {'file_number': number}

    # Check if number exists in modified file
    if number in existing_data:
        # Merge all columns from existing data
        base_entry.update(existing_data[number])

    merged_data.append(base_entry)

#############################################
# RESULTS SAVING
#############################################

# Convert merged data to DataFrame and save
merged_df = pd.DataFrame(merged_data)
merged_df.to_csv('merged_output.csv', index=False)
