import os
import torch
import torchvision
import cv2
import numpy as np


base_dir = os.getcwd()

input_dir = os.path.join(base_dir, '..', 'Maize104', 'data')
output_dir = os.path.join(base_dir, 'deeplabv3_maize')
os.makedirs(output_dir, exist_ok=True)

# Parameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).to(DEVICE)
model.eval()


def preprocess_and_get_original(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image at {img_path}. Skipping.")
        return None, None

    resized_bgr_image = cv2.resize(image, (512, 512))

    image_rgb = cv2.cvtColor(resized_bgr_image, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_normalized - mean) / std

    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_tensor = torch.tensor(image_transposed, dtype=torch.float).unsqueeze(0)

    return image_tensor, resized_bgr_image



print(f"Processing images from: {input_dir}")
print(f"Saving original segmented parts to: {output_dir}")

for file in os.listdir(input_dir):
    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    input_path = os.path.join(input_dir, file)
    print(f"Processing {file}...")

    input_tensor, original_image = preprocess_and_get_original(input_path)

    if input_tensor is None:
        continue

    input_tensor = input_tensor.to(DEVICE)


    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        pred_mask = output.argmax(0).byte().cpu().numpy()


    final_image = np.zeros_like(original_image)


    object_mask = pred_mask != 0

    final_image[object_mask] = original_image[object_mask]

    # Save the Result
    output_path = os.path.join(output_dir, file)
    cv2.imwrite(output_path, final_image)

print("\nProcessing complete.")
