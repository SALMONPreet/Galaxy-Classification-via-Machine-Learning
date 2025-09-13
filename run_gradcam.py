import torch
from model import get_resnet18
from gradcam_utils import visualize_gradcam
import os
import numpy as np
import torch.serialization

# -----------------------------
# 1. Device setup
# -----------------------------

import os

#test_path = os.path.join("data", "images_test_rev1", "291463.jpg")
#print("Exists?", os.path.exists(test_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "best_model.pth"

model = get_resnet18(num_classes=37, pretrained=False).to(device)

try:
    # Safe loading for old-style checkpoints
    with torch.serialization.safe_globals([np._core.multiarray.scalar]):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded checkpoint successfully.")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded old-style checkpoint successfully.")

except Exception as e:
    raise RuntimeError(f"Failed to load checkpoint: {e}")


model.eval()

# -----------------------------
# 3. Set the path to a sample image
# -----------------------------

sample_galaxy_id = input("Enter GalaxyID : ").strip()

image_path = os.path.join("data", "images_test_rev1", f"{sample_galaxy_id}.jpg")
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}")


# -----------------------------
# 4. Run Grad-CAM
# -----------------------------
visualize_gradcam(model, image_path, target_class=None, save_path="gradcam_output.jpg")
print(f"Grad-CAM saved to gradcam_output.jpg for GalaxyID {sample_galaxy_id}")
