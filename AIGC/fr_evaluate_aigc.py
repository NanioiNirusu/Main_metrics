import os
import csv
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import pyiqa
import logging

#############################################
# LOGGING CONFIGURATION
#############################################

# def setup_logging(log_file='evaluation.log'):
#     """
#     Configures logging to output messages to both a file and the console.
#     """
#     logging.basicConfig(
#         level=logging.DEBUG,  # Set to DEBUG for detailed trace
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file, mode='a'),
#             logging.StreamHandler()
#         ]
#     )
#     logging.info("Logging initialized.")

#############################################
# LEGEND AND DOCUMENTATION
#############################################

FULL_REFERENCE_AIGC_LEGEND = r"""... Legend for AIGC Full-Reference Metrics:

1. LPIPS (Learned Perceptual Image Patch Similarity):
   - A deep-learning-based metric that measures perceptual similarity between two images.
   - Lower values indicate higher perceptual similarity.
   - Range: Approximately 0 to 1 (0 = identical, 1 = very different).
   - How It Works:
     LPIPS uses a deep neural network (e.g., VGG or AlexNet) to extract perceptual features 
     from both images, then computes a weighted L2 distance in feature space.

2. DISTS (Deep Image Structure and Texture Similarity):
   - A full-reference metric focusing on image structure and texture similarity.
   - Lower values indicate higher similarity.
   - Range: Typically 0 to ~1 (lower = more similar).
   - How It Works:
     DISTS extracts hierarchical feature maps from a pretrained network, comparing both 
     texture and structural information between reference and test images.

3. Evaluation Time (s):
   - Measures how long it takes to compute all metrics for one test/reference image pair.
   - Lower values indicate faster computation.
   - Range: 0 to infinity (seconds).
   - How It Works:
     This is simply the end time minus the start time for computing the metrics.
"""

def save_full_reference_aigc_legend(output_dir):
    """
    Save the AIGC full-reference metrics legend as a text file.
    """
    legend_filename = "legend_full_reference_aigc.txt"
    legend_path = os.path.join(output_dir, legend_filename)
    try:
        with open(legend_path, "w", encoding="utf-8") as f:
            f.write(FULL_REFERENCE_AIGC_LEGEND)
        logging.info(f"AIGC full-reference legend saved to {legend_path}")
    except Exception as e:
        logging.error(f"Failed to save legend to {legend_path}: {e}")

#############################################
# IMAGE MATCHING AND PROCESSING
#############################################

def extract_reference_identifier(test_image_name):
    """
    Match the test image to its corresponding reference image
    by taking the beginning of the test image name.
    E.g., "I00_00_00.jpg" -> "I00.jpg"
    """
    reference_prefix = test_image_name.split("_")[0]  # e.g. "I00"
    reference_name = f"{reference_prefix}.png"
    reference_path = os.path.join(reference_directory, reference_name)
    print(reference_path)
    if os.path.exists(reference_path):
        return reference_path
    else:
        return None

def match_reference_image(test_image_name, reference_dir, possible_extensions=[".jpg", ".png", ".jpeg"]):
    """
    Match the test image to its corresponding reference image
    by using the extracted reference identifier and searching for any valid extension.
    """
    reference_id = extract_reference_identifier(test_image_name)
    if reference_id is None:
        logging.warning(f"Could not extract reference ID from {test_image_name}.")
        return None

    for ext in possible_extensions:
        reference_name = f"{reference_id}"
        reference_path = os.path.join(reference_dir, reference_name)
        if os.path.exists(reference_path):
            return reference_path
    logging.warning(f"No matching reference image found for {test_image_name} with extensions {possible_extensions}.")
    return None

#############################################
# IMAGE PREPROCESSING
#############################################

def ensure_reference_resolution(reference_image_path, target_size):
    """
    Check if the given reference image is the correct resolution.
    If not, resize
    """
    try:
        with Image.open(reference_image_path) as img:
            if img.size != target_size:
                img_resized = img.resize(target_size, Image.LANCZOS)
                img_resized.save(reference_image_path)
            else:
                pass
    except Exception as e:
        logging.error(f"Failed to process reference image {reference_image_path}: {e}")

def get_possible_extensions(reference_dir):
    """
    Determine the possible file extensions for reference images.
    """
    extensions = set()
    for file in os.listdir(reference_dir):
        if os.path.isfile(os.path.join(reference_dir, file)):
            _, ext = os.path.splitext(file)
            if ext:
                extensions.add(ext.lower())
    if not extensions:
        extensions.add(".jpg")
    return list(extensions)

#############################################
# METRICS EVALUATION
#############################################

def evaluate_full_reference_aigc_metrics(test_dir, reference_dir, output_csv, possible_extensions=[".jpg", ".png", ".jpeg"]):
    """
    Evaluate full-reference AIGC metrics (LPIPS, DISTS) for all test images
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        lpips_metric = pyiqa.create_metric('lpips', device=device)
        logging.info("Initialized LPIPS metric.")
    except Exception as e:
        logging.error(f"Failed to initialize LPIPS metric: {e}")
        return

    try:
        dists_metric = pyiqa.create_metric('dists', device=device)
        logging.info("Initialized DISTS metric.")
    except Exception as e:
        logging.error(f"Failed to initialize DISTS metric: {e}")
        return

    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ["Test Image", "Reference Image", "LPIPS", "DISTS", "Evaluation Time (s)"]
            writer.writerow(header)
            logging.info(f"CSV header written: {header}")

            # Iterate over test images
            for test_image_name in os.listdir(test_dir):
                test_image_path = os.path.join(test_dir, test_image_name)
                if not os.path.isfile(test_image_path):
                    continue

                try:
                    # Load test image
                    with Image.open(test_image_path) as test_image:
                        test_image = test_image.convert('RGB')
                        test_size = test_image.size  # (width, height)
                        # logging.debug(f"Loaded test image {test_image_name} with size {test_size}")

                    # Match reference image
                    reference_image_path = match_reference_image(test_image_name, reference_dir, possible_extensions)
                    if reference_image_path is None:
                        logging.warning(f"No reference image found for {test_image_name}. Skipping...")
                        continue

                    # Ensure reference image resolution matches test image
                    ensure_reference_resolution(reference_image_path, target_size=test_size)

                    # Load reference image
                    with Image.open(reference_image_path) as reference_image:
                        reference_image = reference_image.convert('RGB')

                    # Start timer
                    start_time = time.time()

                    # Convert images to tensors for pyiqa metrics
                    transform_pyiqa = transforms.ToTensor()
                    test_tensor = transform_pyiqa(test_image).unsqueeze(0).to(device)
                    ref_tensor = transform_pyiqa(reference_image).unsqueeze(0).to(device)

                    # Compute pyiqa metrics
                    try:
                        lpips_val = lpips_metric(test_tensor, ref_tensor).item()
                    except Exception as e:
                        logging.error(f"Error computing LPIPS for {test_image_name}: {e}")
                        lpips_val = "Error"

                    try:
                        dists_val = dists_metric(test_tensor, ref_tensor).item()
                    except Exception as e:
                        logging.error(f"Error computing DISTS for {test_image_name}: {e}")
                        dists_val = "Error"

                    # End timer
                    eval_time = time.time() - start_time

                    # Write to CSV
                    writer.writerow([
                        test_image_name,
                        os.path.basename(reference_image_path),
                        lpips_val,
                        dists_val,
                        round(eval_time, 4)
                    ])

                except Exception as e:
                    logging.error(f"Error processing {test_image_name}: {e}")
                    continue

    except Exception as e:
        logging.error(f"Failed to write to CSV {output_csv}: {e}")

#############################################
# MAIN SCRIPT EXECUTION
#############################################

if __name__ == "__main__":
    # Directory containing generated (test) images
    test_directory = "J:/Masters/Datasets/KADID-10K/kadid10k_(2)/kadid10k/images"

    # Directory containing reference images
    reference_directory = "J:/Masters/Datasets/KADID-10K/kadid10k_(2)/kadid10k/ref_images"

    # Output CSV file
    output_file = "full_reference_aigc_metrics.csv"

    # Determine the possible reference image extensions
    possible_extensions = get_possible_extensions(reference_directory)
    logging.info(f"Reference images are expected to have the following extensions: {possible_extensions}")

    # Save the AIGC full-reference legend
    save_full_reference_aigc_legend(os.path.dirname(os.path.abspath(output_file)))

    # Evaluate AIGC metrics
    evaluate_full_reference_aigc_metrics(test_directory, reference_directory, output_file, possible_extensions)

    logging.info(f"AIGC full-reference metrics evaluation completed. Results saved to {output_file}.")
    print(f"AIGC full-reference metrics evaluation completed. Results saved to {output_file}.")
