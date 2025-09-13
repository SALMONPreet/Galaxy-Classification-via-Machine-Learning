import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import os
import multiprocessing

from galaxy_dataset import GalaxyZooDataset
from model import get_resnet18

# ------------------------
# 1. Data preparation
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Sample N images for training/testing
labels_df = pd.read_csv("data/training_solutions_rev1.csv").sample(n=5000)
train_df = labels_df.sample(frac=0.8, random_state=42)
val_df = labels_df.drop(train_df.index)

train_dataset = GalaxyZooDataset("data/images_training_rev1", train_df, transform)
val_dataset = GalaxyZooDataset("data/images_training_rev1", val_df, transform)

# ------------------------
# DataLoader workers
# ------------------------
loader_workers = min(0, multiprocessing.cpu_count())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=loader_workers)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=loader_workers)

# ------------------------
# 2. Model
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet18(num_classes=37, pretrained=True).to(device)

# ------------------------
# 2a. Load previous checkpoint (if any)
# ------------------------
best_val_loss = float("inf")
best_rmse = None
best_train_loss = None

if os.path.exists("best_model.pth"):
    try:
        checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
            best_rmse = checkpoint.get("best_rmse", best_rmse)
            best_train_loss = checkpoint.get("best_train_loss", best_train_loss)
            print(f"Loaded checkpoint. Previous best val_loss: {best_val_loss}, RMSE: {best_rmse}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded old-style checkpoint. Starting with these weights.")
    except Exception as e:
        print("Failed to load checkpoint:", e)
        print("Starting training from scratch.")
else:
    print("No checkpoint found, starting training from scratch.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ------------------------
# 3. Training Loop
# ------------------------
def train_model(epochs=10):
    global best_val_loss, best_rmse, best_train_loss

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        preds = np.vstack(all_preds)
        truths = np.vstack(all_targets)

        # Manual RMSE calculation
        rmse = np.sqrt(np.mean((truths - preds) ** 2))
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Val Loss: {avg_val_loss:.4f} "
              f"RMSE: {rmse:.4f}")

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_rmse = rmse
            best_train_loss = avg_train_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "best_rmse": best_rmse,
                "best_train_loss": best_train_loss
            }, "best_model.pth")
            print(f"Saved model at epoch {epoch+1}")
        

    print("Training complete. Best val loss:", best_val_loss)

# ------------------------
# 4. Save validation predictions for best epoch
# ------------------------
def save_best_predictions():
    checkpoint = torch.load("best_model.pth", map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            outputs = np.clip(outputs, 0, 1)
            for gid, pred in zip(val_dataset.galaxy_ids, outputs):
                results.append([gid] + list(pred))

    cols = ["GalaxyID"] + [f"Class{i+1}" for i in range(37)]
    df_out = pd.DataFrame(results, columns=cols)
    df_out.to_csv("val_predictions_best_epoch.csv", index=False)
    print("Validation predictions saved to val_predictions_best_epoch.csv")

# ------------------------
# 5. Run training and save predictions
# ------------------------
if __name__ == "__main__":
    train_model(epochs=5)
    #save_best_predictions()
