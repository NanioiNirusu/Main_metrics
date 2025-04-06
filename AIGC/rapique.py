import os
import copy  # Import copy module
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import scipy.stats  # Ensure scipy is installed
import logging

#############################################
# CONFIGURATION AND SETUP
#############################################

# Determine the absolute path for the log file
current_dir = os.path.dirname(os.path.abspath(__file__))
# log_file_path = os.path.join(current_dir, 'rapique_debug.log')

# Configure logging within rapique.py
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file_path, mode='a'),
#         logging.StreamHandler()  # This outputs logs to the console
#     ]
# )
#
# # Confirm where logs are being written
# print(f"Logging to: {log_file_path}")

#############################################
# RAPIQUE CLASS DEFINITION
#############################################

class RAPIQUE:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Load pre-trained ResNet-50 model with updated weights parameter
        weights = models.ResNet50_Weights.IMAGENET1K_V1  # Update to the appropriate weights
        # logging.debug(f"Loaded weights.meta: {weights.meta}")  # Debugging line
        self.model = models.resnet50(weights=weights).to(self.device).eval()

        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

        # Define layers to extract features from
        self.feature_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        self.features = {}
        self._register_hooks()

        # Define image transformations
        # Check if 'mean' and 'std' exist in weights.meta; if not, use default ImageNet values
        if 'mean' in weights.meta and 'std' in weights.meta:
            mean = weights.meta['mean']
            std = weights.meta['std']
            logging.info("Using mean and std from weights.meta.")
        else:
            mean = [0.485, 0.456, 0.406]  # Standard ImageNet mean
            std = [0.229, 0.224, 0.225]  # Standard ImageNet std
            logging.warning("'mean' and/or 'std' not found in weights.meta. Using default ImageNet values.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=std),
        ])

    #############################################
    # FEATURE EXTRACTION HOOKS
    #############################################

    def _register_hooks(self):
        """
        Register hooks to capture feature maps from specified layers.
        """
        for layer_name in self.feature_layers:
            layer = getattr(self.model, layer_name)
            layer.register_forward_hook(self._get_hook(layer_name))

    def _get_hook(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output.detach().cpu().numpy()

        return hook

    #############################################
    # FEATURE EXTRACTION FUNCTIONS
    #############################################

    def extract_features(self, img):
        """
        Extract features from specified layers.
        """
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(img_tensor)
        return copy.deepcopy(self.features)  # Return a deep copy

    #############################################
    # QUALITY ASSESSMENT FUNCTIONS
    #############################################

    def compute_quality_score(self, test_image, reference_image):
        """
        Compute the RAPIQUE quality score between test and reference images.
        """
        # Extract features
        test_features = self.extract_features(test_image)
        ref_features = self.extract_features(reference_image)

        # Initialize score
        total_score = 0.0
        feature_count = 0

        # Iterate over each layer's features
        for layer in self.feature_layers:
            test_feat = test_features[layer]
            ref_feat = ref_features[layer]

            # Compute feature difference
            feat_diff = np.abs(test_feat - ref_feat)

            # Check for NaNs or Infs in feat_diff
            if not np.isfinite(feat_diff).all():
                logging.error(f"Non-finite values found in feat_diff for layer {layer}.")
                return np.nan  # Early exit if invalid values are found

            # Compute statistical measures
            mean_diff = np.mean(feat_diff)
            var_diff = np.var(feat_diff)
            skew_diff = scipy.stats.skew(feat_diff.flatten())
            kurt_diff = scipy.stats.kurtosis(feat_diff.flatten())

            # Handle skew_diff and kurt_diff being nan
            if np.isnan(skew_diff):
                logging.warning(f"skew_diff is NaN for layer {layer}. Assigning 0.")
                skew_diff = 0.0
            if np.isnan(kurt_diff):
                logging.warning(f"kurt_diff is NaN for layer {layer}. Assigning 0.")
                kurt_diff = 0.0

            # Check for NaNs in statistical measures after handling
            if not all(np.isfinite([mean_diff, var_diff, skew_diff, kurt_diff])):
                logging.error(f"Statistical measures contain non-finite values in layer {layer}.")
                return np.nan

            # Aggregate scores (customize weights as needed)
            layer_score = (mean_diff + var_diff + abs(skew_diff) + abs(kurt_diff)) / 4

            total_score += layer_score
            feature_count += 1

        # Normalize the total score
        if feature_count == 0:
            logging.error("Feature count is zero. Cannot normalize score.")
            return np.nan

        normalized_score = total_score / feature_count

        return normalized_score

#############################################
# MAIN SCRIPT EXECUTION
#############################################

if __name__ == "__main__":
    rapique = RAPIQUE()
    test_img_path = 'J:/Masters/Datasets/PIPAL/Distortion_1/Distortion_1/A0001_00_00.bmp'
    ref_img_path = 'J:/Masters/Datasets/PIPAL/train_ref1/A0001.bmp'

    if not os.path.exists(test_img_path):
        logging.error(f"Test image not found at path: {test_img_path}")
        print(f"Test image not found at path: {test_img_path}")
        exit(1)

    if not os.path.exists(ref_img_path):
        logging.error(f"Reference image not found at path: {ref_img_path}")
        print(f"Reference image not found at path: {ref_img_path}")
        exit(1)

    try:
        test_img = Image.open(test_img_path).convert('RGB')
    except Exception as e:
        logging.error(f"Failed to open test image: {e}")
        print(f"Failed to open test image: {e}")
        exit(1)

    try:
        ref_img = Image.open(ref_img_path).convert('RGB')
    except Exception as e:
        logging.error(f"Failed to open reference image: {e}")
        print(f"Failed to open reference image: {e}")
        exit(1)

    rapique_score = rapique.compute_quality_score(test_img, ref_img)
    print(f"RAPIQUE Score: {rapique_score}")
