import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import os
from skimage import morphology


model_type = "vit_b"
checkpoint_path = "sam_vit_b_01ec64.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
predictor = SamPredictor(sam)

img_dir=os.path.join(os.getcwd(), '..', 'Maize104', 'data')
out_dir=os.path.join(os.getcwd(), '..', 'segmented_maize')
os.makedirs(out_dir, exist_ok=True)

for filename in os.listdir(img_dir):
    image_path = os.path.join(img_dir, filename)
    image = cv2.imread(image_path)  # Load image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    predictor.set_image(image)

    # Input point for SAM
    input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
    input_label = np.array([1])


    # Predict mask using SAM
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    mask = masks[0].astype(np.uint8)

    # Apply mask to image: remove background
    segmented = image.copy()
    segmented[mask == 0] = 0  # Zero out background
    output_path = os.path.join(out_dir, f"segmented_{filename}")
    cv2.imwrite(output_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    plt.figure(figsize=(10, 10))
    plt.title("Left-click: Plant | Right-click: Background | Close window to segment")
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
    plt.show()
    print("Saved file")

