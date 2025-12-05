import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Configuration
ENCODER = 'efficientnet-b4'
ENCODER_WEIGHTS = 'imagenet'    # Pretrained on ImageNet
CLASSES = ['plant']             # Binary segmentation (single class: plant)
ACTIVATION = 'sigmoid'          # Use sigmoid for binary mask output
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

input_dir = os.path.join(os.getcwd(), '..', 'new')
output_dir = os.path.join(os.getcwd(), 'unet_wheat')
os.makedirs(output_dir, exist_ok=True)

# Model Initialization
model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)
model = model.to(DEVICE)
model.eval()

def preprocess(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (384, 384))
    image_norm = image_resized / 255.0
    image_transposed = np.transpose(image_norm, (2, 0, 1))
    tensor = torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0)
    return tensor, image_rgb

for file in os.listdir(input_dir):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    IMAGE_PATH = os.path.join(input_dir, file)
    input_tensor, original_image = preprocess(IMAGE_PATH)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255  # Threshold

    # Resize mask back to original size
    binary_mask_resized = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create Output Image with Background Blacked Out
    mask_3ch = np.stack([binary_mask_resized]*3, axis=-1)
    masked_image = np.where(mask_3ch == 255, original_image, 0)

    # Save the Masked Image
    OUTPUT_MASKED_IMAGE_PATH = os.path.join(output_dir, file)
    cv2.imwrite(OUTPUT_MASKED_IMAGE_PATH, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    print("Segmentation is complete")
