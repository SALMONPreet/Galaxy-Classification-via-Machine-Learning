import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Grad-CAM Implementation
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        """
        model: PyTorch model
        target_layer: convolutional layer to visualize
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        input_tensor: torch tensor of shape [1, C, H, W]
        target_class: integer class index (optional)
        """
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        loss = output[:, target_class]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        gradcam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        gradcam_map = torch.relu(gradcam_map)

        # Normalize to [0,1] and resize to input size
        gradcam_map = gradcam_map.squeeze().cpu().numpy()
        gradcam_map = cv2.resize(gradcam_map, (input_tensor.shape[3], input_tensor.shape[2]))
        gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)

        return gradcam_map


# -----------------------------
# Utility to visualize Grad-CAM
# -----------------------------
def visualize_gradcam(model, img_path, target_class=None, save_path="gradcam_result.jpg"):
    """
    model: PyTorch model
    img_path: path to input image
    target_class: optional class index
    save_path: where to save the Grad-CAM overlay
    """
    device = next(model.parameters()).device

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load image
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dim

    # Pick the last conv layer of ResNet18
    target_layer = model.layer4[-1].conv2

    gradcam = GradCAM(model, target_layer)
    cam_map = gradcam.generate(input_tensor, target_class=target_class)

    heatmap = cv2.applyColorMap(np.uint8(255 * (1.0 - cam_map)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Original image (unnormalized)
    img_np = np.array(image.resize((224, 224))) / 255.0

    overlay = 0.4 * heatmap + 0.6 * img_np
    overlay = np.clip(overlay, 0, 1)

    # Save & display
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM Visualization")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    print(f"Grad-CAM saved to {save_path}")
