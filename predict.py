import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F

from galaxy_dataset import GalaxyZooDataset
from model import get_resnet18

# ------------------------
# 1. Parameters
# ------------------------
image_dir = "data/images_test_rev1"
checkpoint_path = "best_model.pth"
num_samples = 5000        # how many test images to predict
random_sample = True      # True = random N, False = first N

# ------------------------
# 2. Load test IDs CSV
# ------------------------
all_test_ids_path = "data/all_test_ids.csv"
all_ids_df = pd.read_csv(all_test_ids_path)

if num_samples < len(all_ids_df):
    if random_sample:
        selected_ids = all_ids_df.sample(n=num_samples, random_state=42)["GalaxyID"].tolist()
    else:
        selected_ids = all_ids_df.iloc[:num_samples]["GalaxyID"].tolist()
else:
    selected_ids = all_ids_df["GalaxyID"].tolist()

# ------------------------
# 3. Data preparation
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = GalaxyZooDataset(image_dir, labels_df=None, transform=transform, galaxy_ids=selected_ids)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ------------------------
# 4. Load model
# ------------------------
import torch
import torch.serialization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet18(num_classes=37, pretrained=False).to(device)

try:
    # Safe loading for old-style weights
    import numpy as np
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

# ------------------------
# 5. Run inference
# ------------------------
results = []
with torch.no_grad():
    for images, galaxy_ids in test_loader:
        images = images.to(device)
        outputs = model(images)
        #outputs = F.softmax(outputs, dim=1)  # <-- Apply softmax to get valid probabilities
        outputs = np.clip(outputs, 0, 1)
        outputs = outputs.cpu().numpy()
        for gid, probs in zip(galaxy_ids, outputs):
            results.append([gid] + list(probs))

# ------------------------
# 6. Save predictions
# ------------------------
cols = ["GalaxyID"] + [f"Class{i+1}" for i in range(37)]
df_out = pd.DataFrame(results, columns=cols)
df_out.to_csv("predictions.csv", index=False)
print(f"Predictions saved to predictions.csv ({len(df_out)} rows)")
