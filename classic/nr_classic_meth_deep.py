import os
import csv
import time
import cv2
import re
import numpy as np
from PIL import Image
import torch
import pyiqa
from torchvision import transforms

#############################################
# DOCUMENTATION AND CONSTANTS
#############################################

# Detailed Legend for Metrics
LEGEND = """
Legend for Image Quality Metrics:

1. BRISQUE (Blind/Reference-less Image Spatial Quality Evaluator):
   - A no-reference metric that predicts perceptual quality based on natural scene statistics.
   - Lower values indicate better quality.
   - Range: Typically 0 to 100 (higher values indicate poorer quality).
   - How It Works: BRISQUE extracts local normalized luminance (MSCN) coefficients from the image and models their distribution using a Gaussian distribution. It then computes features based on the deviations from natural scene statistics to predict quality.

2. NIQE (Natural Image Quality Evaluator):
   - A no-reference metric that measures quality based on deviations from natural image statistics.
   - Lower values indicate better quality.
   - Range: Typically 0 to 100 (higher values indicate poorer quality).
   - How It Works: NIQE builds a statistical model of natural images using features like local luminance and contrast. It then compares the test image's statistics to this model to compute quality.

3. NR-SNR (No-Reference Signal-to-Noise Ratio):
   - Measures the level of noise relative to the signal without requiring a reference image.
   - Higher values indicate less noise.
   - Range: 0 to infinity (higher values are better).
   - How It Works: NR-SNR is computed as the ratio of the mean pixel intensity (signal) to the standard deviation of noise (noise) using local statistics.

4. NR-PSNR (No-Reference Peak Signal-to-Noise Ratio):
    - Measures the estimated quality of an image without requiring a reference.
    - Higher values indicate better quality.
    - Range: 0 to infinity (higher values are better).
    - How It Works: NR-PSNR is computed as the ratio of the maximum possible pixel value (peak signal) to the estimated noise using local image variance.

5. Evaluation Time (s):
    - Measures the time taken to compute all metrics for the image.
    - Lower values indicate faster computation.
    - Range: 0 to infinity (lower values are better).
    - How It Works: Evaluation time is computed as the elapsed time between the start and end of the metric computation process.
"""


#############################################
# UTILITY FUNCTIONS
#############################################

def save_legend(output_dir):
    """Save the detailed legend to a separate text file."""
    legend_path = os.path.join(output_dir, "legend.txt")
    with open(legend_path, "w") as file:
        file.write(LEGEND)
    print(f"Legend saved to {legend_path}")


#############################################
# IMAGE QUALITY METRICS - NOISE
#############################################

def calculate_noise(img):
    """No-Reference Noise Calculation"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Calculate local statistics for better noise estimation
    local_mean = cv2.blur(gray, (3, 3))
    local_var = cv2.blur(np.square(gray - local_mean), (3, 3))

    # NR-SNR calculation
    signal_power = np.mean(np.square(local_mean))
    noise_power = np.mean(local_var)
    nr_snr = 10 * np.log10(signal_power / noise_power) if noise_power != 0 else 0

    # NR-PSNR calculation
    mse = np.mean(local_var)
    nr_psnr = 10 * np.log10((255 ** 2) / (mse + 1e-10))

    return {"nr_snr": nr_snr, "nr_psnr": nr_psnr}


#############################################
# MAIN EVALUATION FUNCTION
#############################################

def evaluate_no_reference_metrics(image_dir, output_csv):
    """Evaluate no reference metrics for all images."""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize metrics
    try:
        # Initialize PyIQA metrics - these models analyze statistical properties of images
        # to estimate quality without needing reference images
        brisque_metric = pyiqa.create_metric('brisque', device=device)  # Analyzes MSCN coefficients
        # and their distribution
        niqe_metric = pyiqa.create_metric('niqe', device=device)  # Compares against a multivariate Gaussian model
    except Exception as init_error:
        print(f"Metric initialization error: {init_error}")
        return

    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Image", "BRISQUE", "NIQE", "NR-SNR", "NR-PSNR", "Evaluation Time (s)"
        ])

        for img_name in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img_name)
            print(img_name)
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')

                # Start timer
                start_time = time.time()

                # Compute metrics
                brisque_val = brisque_metric(transforms.ToTensor()(img).unsqueeze(0).to(device)).item()
                niqe_val = niqe_metric(transforms.ToTensor()(img).unsqueeze(0).to(device)).item()
                noise = calculate_noise(img)

                # End timer
                eval_time = time.time() - start_time

                # Write results to CSV
                writer.writerow([
                    img_name, brisque_val, niqe_val, noise["nr_snr"], noise["nr_psnr"], eval_time
                ])

            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue


#############################################
# FILENAME UTILITY FUNCTIONS
#############################################

def get_unique_filename(base_output_file):
    """
    Generate a unique output filename by incrementing the distortion number

    Args:
        base_output_file (str): Base filename to check and potentially modify

    Returns:
        str: A unique filename with incremented distortion number
    """

    # Use regex to find and increment the distortion number
    def increment_distortion_number(filename):
        # Match pattern like D2 or D3 at the start of the filename
        match = re.match(r'(D)(\d+)(_no_reference_metrics_classic\.csv)', filename)
        if match:
            prefix, number, suffix = match.groups()
            return f"{prefix}{int(number) + 1}{suffix}"
        return filename

    current_filename = base_output_file

    # Check if file exists, increment distortion number if it does
    while os.path.exists(current_filename):
        current_filename = increment_distortion_number(current_filename)

    return current_filename


#############################################
# MAIN SCRIPT EXECUTION
#############################################

if __name__ == "__main__":
    # Directory containing test images
    test_directory = "J:/Masters/Datasets/UHD-IQA-database/UHD-IQA-database/All"

    base_output_file = "D1_no_reference_metrics_classic.csv"

    # Output CSV file
    output_file = get_unique_filename(base_output_file)
    # Save legend to a separate file
    save_legend(os.path.dirname(output_file))

    # Evaluate no reference metrics
    evaluate_no_reference_metrics(test_directory, output_file)

    print(f"No reference metrics evaluation completed. Results saved to {output_file}")
