import os.path
from torchvision import models, transforms
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
input_dir = os.path.join(os.getcwd(), '..', 'new')
output_dir = os.path.join(os.getcwd(), 'R-CNN_wheat')
os.makedirs(output_dir, exist_ok=True)

# Load Mask R-CNN Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()
model = model.to(device)


def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image_rgb)
    return image, tensor

SCORE_THRESH = 0.5

for file in os.listdir(input_dir):
    INPUT_PATH = os.path.join(input_dir, file)
    orig_img, img_tensor = load_and_preprocess(INPUT_PATH)
    img_tensor = img_tensor.unsqueeze(0).to(device)


    with torch.no_grad():
        outputs = model(img_tensor)

    masks = outputs[0]['masks']  # (N, 1, H, W)
    scores = outputs[0]['scores']

    # Combine Masks of Instances
    binary_mask = np.zeros(masks.shape[-2:], dtype=np.uint8)
    for i in range(len(masks)):
        if scores[i] >= SCORE_THRESH:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()  # [0,255]
            binary_mask = np.maximum(binary_mask, (mask > 127).astype(np.uint8) * 255)

    # Ensure binary_mask is same size as orig_img
    if binary_mask.shape != orig_img.shape[:2]:
        binary_mask = cv2.resize(binary_mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Convert mask to boolean, expand to 3 channels
    mask_bool = binary_mask == 255
    masked_img = np.zeros_like(orig_img)
    masked_img[mask_bool] = orig_img[mask_bool]

    # Save the Mask
    save_path=os.path.join(output_dir, file)
    cv2.imwrite(save_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
    print(f"Segmentation mask saved")
