import os
import csv

#############################################
# DOCUMENTATION AND CONSTANTS
#############################################

"""
Helper module for processing text files containing BMP filename and value pairs.
This script extracts data from text files where each line contains a BMP filename and associated value,
then consolidates this information into a single CSV file for easier analysis.
"""

#############################################
# FILE PROCESSING FUNCTIONS
#############################################

def process_txt_file(file_path):
    """
    Extract BMP filenames and their numeric values from a text file.

    Args:
        file_path (str): Path to the text file

    Returns:
        list: List of dictionaries containing filename and value
    """
    bmp_data = []
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            # Split the content into individual BMP file entries
            entries = content.split()

            for entry in entries:
                # Split each entry into filename and value
                filename, value = entry.split(',')
                bmp_data.append({
                    'Filename': filename,
                    'Value': float(value)
                })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return bmp_data

#############################################
# DIRECTORY PROCESSING FUNCTIONS
#############################################

def process_directory(directory_path, output_csv):
    """
    Process text files in a directory and write BMP data to CSV.

    Args:
        directory_path (str): Path to the directory containing text files
        output_csv (str): Path to the output CSV file
    """
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    # Collect BMP file data
    all_bmp_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            file_data = process_txt_file(file_path)
            all_bmp_data.extend(file_data)

    # Write to CSV
    if all_bmp_data:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['Filename', 'Value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for entry in all_bmp_data:
                writer.writerow(entry)

        print(f"CSV file created at {output_csv}")
    else:
        print("No data found.")

#############################################
# MAIN SCRIPT EXECUTION
#############################################

directory_path = "J:/Masters/Datasets/PIPAL/Train_Label/Train_Label/"
output_csv = 'PILAL_bmp_data.csv'
process_directory(directory_path, output_csv)
