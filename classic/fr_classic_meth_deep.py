import os
import csv
import re
import time
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

#############################################
# FULL-REFERENCE LEGEND
#############################################

FULL_REFERENCE_LEGEND = """
Legend for Full-Reference Image Quality Metrics:

1. PSNR (Peak Signal-to-Noise Ratio):
   - Measures the quality of an image relative to a reference image.
   - Higher values indicate better quality.
   - Typical Range (for 8-bit images): 0 to ~50 dB.
     Values above 30 dB are generally considered acceptable to good.
   - How It Works:
     PSNR is computed as the ratio between the maximum possible value of a pixel (peak) 
     and the mean squared error (MSE) between the test and reference images:
         PSNR = 10 * log10((MAX^2) / MSE)
     where MAX is the maximum pixel intensity (e.g. 255 for 8-bit images).

2. SSIM (Structural Similarity Index Measure):
   - Measures the perceptual similarity between test and reference images, 
     taking into account luminance, contrast, and structure.
   - Higher values indicate better structural similarity.
   - Range: 0 to 1 (1 indicates perfect structural similarity).
   - How It Works:
     SSIM compares local patterns of pixel intensities normalized for luminance and contrast. 
     It is composed of three factors: luminance, contrast, and structure.

3. Evaluation Time (s):
   - Measures how long it takes to compute all metrics for one image comparison.
   - Lower values indicate faster computation.
   - Range: 0 to infinity (seconds).
   - How It Works:
     This is simply the end time minus the start time for computing the metrics.
"""

#############################################
# UTILITY FUNCTIONS
#############################################

def save_full_reference_legend(output_dir):
    """
    Save the full-reference metrics legend as a text file.
    """
    legend_filename = "legend_full_reference.txt"
    legend_path = os.path.join(output_dir, legend_filename)
    with open(legend_path, "w", encoding="utf-8") as f:
        f.write(FULL_REFERENCE_LEGEND)
    print(f"Full-reference legend saved to {legend_path}")

#############################################
# IMAGE REFERENCE PROCESSING
#############################################

def extract_reference_number(test_image_name):
    """
    Extract the numeric reference from the second underscore-delimited chunk
    of the filename.
    E.g.,
      'MJ_0012_03.jpg' -> returns 12
      'SD_0000_01.jpg' -> returns 0
      'XX_0199_10.jpg' -> returns 199
    """
    parts = test_image_name.split("_")
    if len(parts) == 3:
        try:
            return int(parts[1])  # e.g. '0000' -> 0
        except ValueError:
            return None
    return None


def match_reference_image(test_image_name, reference_dir):
    # """
    # Match the test image to its corresponding reference image.
    # Currently changed to match by taking the beginning of the test image name
    # (e.g., "A0001_00_00.jpg" -> "A0001.jpg").
    #
    # Previous numeric-matching logic is commented out.
    # """
    # # ------------------------------------------------------------
    # # Old :
    # # ------------------------------------------------------------
    # #
    # reference_number = extract_reference_number(test_image_name)
    # if reference_number is None:
    #     return None
    #
    # reference_name = f"{reference_number}.jpg"
    # reference_path = os.path.join(reference_dir, reference_name)
    # if os.path.exists(reference_path):
    #     return reference_path
    # else:
    #     return None

    # ------------------------------------------------------------
    # New :
    # ------------------------------------------------------------
    # reference_prefix = test_image_name.split("_")[0]  # e.g. "A0001"
    # reference_name = f"{reference_prefix}.bmp"
    # reference_path = os.path.join(reference_dir, reference_name)
    # if os.path.exists(reference_path):
    #     return reference_path
    # else:
    #     return None

    """
    Match the test image to its corresponding reference image
    by taking the beginning of the test image name.
    E.g., "I00_00_00.jpg" -> "I00.jpg"
    """
    reference_prefix = test_image_name.split("_")[0]  # e.g. "I00"
    reference_name = f"{reference_prefix}.png"
    reference_path = os.path.join(reference_dir, reference_name)
    if os.path.exists(reference_path):
        return reference_path
    else:
        return None

#############################################
# IMAGE PREPROCESSING
#############################################

def ensure_reference_resolution(reference_image_path, target_size):
    """
    Check if the given reference image is the correct resolution.
    If not, resize it in-place to match the given target_size (i.e., the test image's resolution).
    """
    with Image.open(reference_image_path) as img:
        if img.size != target_size:
            print(f"Resizing reference {os.path.basename(reference_image_path)} "
                  f"from {img.size} to {target_size}")
            img_resized = img.resize(target_size, Image.LANCZOS)
            img_resized.save(reference_image_path)  # Overwrite in-place

#############################################
# IMAGE QUALITY METRICS
#############################################

def calculate_psnr(test_image, reference_image):
    test_gray = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2GRAY)
    mse = np.mean((reference_gray - test_gray) ** 2)
    if mse == 0:
        return float('inf')  # Return infinity for identical images
    return 10 * np.log10((255 ** 2) / mse)

def calculate_ssim(test_image, reference_image):
    test_gray = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(np.array(reference_image), cv2.COLOR_RGB2GRAY)
    return ssim(reference_gray, test_gray, data_range=255)


#############################################
# MAIN EVALUATION FUNCTION
#############################################

def evaluate_full_reference_metrics(test_dir, reference_dir, output_csv):
    """
    Evaluate full reference metrics (PSNR, SSIM) for all test images
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Image", "Reference Image", "PSNR", "SSIM", "Evaluation Time (s)"])

        for test_image_name in os.listdir(test_dir):
            test_image_path = os.path.join(test_dir, test_image_name)
            if not os.path.isfile(test_image_path):
                continue

            try:
                # Load test image
                test_image = Image.open(test_image_path).convert('RGB')
                # Get the test image size (width, height)
                test_size = test_image.size

                # Find matching reference image
                reference_image_path = match_reference_image(test_image_name, reference_dir)
                if not reference_image_path:
                    print(f"No reference image found for {test_image_name}. Skipping...")
                    continue

                # Ensure reference image matches the test image's resolution
                ensure_reference_resolution(reference_image_path, target_size=test_size)

                # Load the (possibly resized) reference image
                reference_image = Image.open(reference_image_path).convert('RGB')

                # Start timer
                start_time = time.time()

                # Compute metrics
                psnr_value = calculate_psnr(test_image, reference_image)
                ssim_value = calculate_ssim(test_image, reference_image)


                # End timer
                eval_time = time.time() - start_time

                # Write results to CSV
                writer.writerow([
                    test_image_name, os.path.basename(reference_image_path),
                    psnr_value, ssim_value, eval_time
                ])

            except Exception as e:
                print(f"Error processing {test_image_name}: {e}")
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
        match = re.match(r'(D)(\d+)(_full_reference_metrics_classic\.csv)', filename)
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
    test_directory = "J:/Masters/Datasets/KADID-10K/kadid10k (2)/kadid10k/images"

    # Directory containing reference images
    reference_directory = "J:/Masters/Datasets/KADID-10K/kadid10k (2)/kadid10k/ref_images"

    base_output_file = "D2_full_reference_metrics_classic.csv"

    # Output CSV file
    output_file = get_unique_filename(base_output_file)

    save_full_reference_legend(os.path.dirname(output_file))

    evaluate_full_reference_metrics(test_directory, reference_directory, output_file)

    print(f"Full-reference metrics evaluation completed. Results saved to {output_file}")
