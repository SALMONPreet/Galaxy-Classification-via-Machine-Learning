
![mlgc - Made with PosterMyWall](https://github.com/user-attachments/assets/3acb9947-ab10-4c69-9a59-a82e15f244c7)

# 🌌 Galaxy Classifier

## 1. Introduction
Understanding galaxy morphology is one of the fundamental problems in observational astronomy.  
The **Galaxy Zoo project** provides a large collection of galaxy images classified by citizen scientists into morphological categories.  
These morphologies are linked to important astrophysical properties such as stellar population, star formation rate, and galaxy evolution.  

In this project, we develop a **deep learning pipeline** to classify galaxies into 37 categories using the Galaxy Zoo dataset.  
Our approach leverages a **ResNet18 convolutional neural network** with a custom classification head, trained using PyTorch.  
We also implement **Grad-CAM visualizations** to interpret model predictions and highlight the image regions that drive classification decisions.  

---

## 2. Dataset

### 2.1 Source
- Data obtained from the **[Galaxy Zoo Challenge on Kaggle](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)**.  
- The dataset consists of galaxy images classified by citizen scientists into 37 morphological categories.  

### 2.2 Structure
- **Training set:** ~60,000 galaxy images with corresponding classification probabilities.  
- **Test set:** ~70,000 unlabeled galaxy images.  
- **Labels:** Provided as a CSV file (`training_solutions_rev1.csv`) with 37 floating-point values for each galaxy.  
- **Image format:** PNG/JPEG, size varies, later resized to 224×224 for model input.  

### 2.3 Labels
- Each galaxy has **37 target values** corresponding to different morphological questions (e.g., presence of spiral arms, bar structure, disk orientation, bulge prominence).  
- These are **probabilistic labels**, meaning each output is a probability rather than a hard classification.  
- Training is therefore treated as a **regression problem** using Mean Squared Error (MSE) loss.  
  

---

## 3. Methodology

### 3.1 Data Preprocessing
- Images are resized to **224×224** pixels.  
- Data augmentation includes:  
  - Random horizontal flips  
  - Random rotations (±20°)  
- Normalization applied using ImageNet mean and standard deviation.  

### 3.2 Dataset Class
Implemented a custom PyTorch dataset (`GalaxyZooDataset`) that:  
- Loads images by GalaxyID.  
- Returns `(image, labels)` during training/validation.  
- Supports inference mode (returns `(image, GalaxyID)` only).  

### 3.3 Model Architecture
- **Base model:** ResNet18 (`torchvision.models.resnet18`)  
- **Classifier head:**  
  - Linear layer → ReLU → Dropout → Linear layer → 37 outputs  
- Pretrained ImageNet weights are used to accelerate convergence.  

### 3.4 Training Setup
- Loss function: **Mean Squared Error (MSE)**  
- Optimizer: **Adam (lr=1e-3)**  
- Batch size: 32  
- Hardware: GPU (if available)  

During training:  
- Model evaluated on validation set at the end of each epoch.  
- Metrics recorded: **Training Loss, Validation Loss, RMSE**.  
- Best model checkpoint saved automatically (`best_model.pth`).  

---

## 4. Implementation

### 4.1 Training
The training pipeline is implemented in **`train.py`**:  
- Loads dataset and applies transformations.  
- Trains ResNet18 for a fixed number of epochs.  
- Saves best-performing model checkpoint.  

### 4.2 Inference
The inference script **`predict.py`**:  
- Loads test GalaxyIDs.  
- Applies the trained model on unseen images.  
- Saves predictions in `predictions.csv` with 37 class probabilities per galaxy.  

### 4.3 Interpretability
We use **Grad-CAM** to visualize which image regions influenced the model’s decision.  
- Implemented in `gradcam_utils.py`.  
- Script `run_gradcam.py` allows users to input a `GalaxyID` and generate a heatmap overlay.  

This helps assess whether the model focuses on relevant features (e.g., spiral arms, bars, galaxy cores).  

---

## 5. Results

### 5.1 Training Performance
- Training Loss (best): `xx.xx`  
- Validation Loss (best): `xx.xx`  
- RMSE (best): `xx.xx`  

*(Exact values depend on dataset subset and number of epochs.)*  

### 5.2 Predictions
- Predictions stored in `predictions.csv`.  
- Each row: `GalaxyID, Class1, Class2, ..., Class37`.  

### 5.3 Visualization
Example Grad-CAM visualization:  

![gradcam_output](https://github.com/user-attachments/assets/4e33bbe0-3ce7-4a04-a647-46fbfa182fe8)   ![193219](https://github.com/user-attachments/assets/2a0dea50-c9c4-449b-a5dd-bd63b398f90e)


This shows which parts of the galaxy image the model relied on for its classification.  

---

## 6. Discussion
- The ResNet18 model achieves reasonable predictive accuracy on a subset of 5000 galaxies.  
- Training on the **full dataset** is expected to improve generalization.  
- The use of **probabilistic labels** makes the task closer to regression than classification.  
- Grad-CAM visualizations confirm that the model often attends to key galaxy features (arms, bulges, bars).  

---

## 7. Future Work
- Experiment with deeper architectures (ResNet50, EfficientNet, Vision Transformers).  
- Incorporate full training dataset instead of a 5k sample.  
- Use **multi-label classification loss functions** (e.g., KL Divergence) instead of plain MSE.  
- Deploy as an interactive **web application** for galaxy exploration.  
- Explore **self-supervised pretraining** for astronomical image data.  

---

## 8. References
- [Galaxy Zoo Data Release](https://data.galaxyzoo.org/)  
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016  
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017  

---

## 9. Prerequisites

Before running the project, ensure you have the following:

### 9.1 Hardware
- A computer with **Python 3.8+** installed  
- **GPU with CUDA support (optional but recommended)** for faster training  
- At least **8 GB RAM** (more recommended for handling large datasets like Galaxy Zoo)  

### 9.2 Software & Libraries
Install the following Python packages:

- `torch` (PyTorch)  
- `torchvision`  
- `numpy`  
- `pandas`  
- `scikit-learn`  
- `opencv-python`  
- `matplotlib`  
- `Pillow`  

You can install all dependencies via:

```bash
pip install -r requirements.txt
