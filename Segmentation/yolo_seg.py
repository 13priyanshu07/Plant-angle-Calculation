import cv2
import os
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")  # or your preferred YOLO-Seg model

# Load image
input_dir = os.path.join(os.getcwd(), '..', 'new')
output_dir = os.path.join(os.getcwd(), 'yoloseg_wheat')
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    INPUT_PATH = os.path.join(input_dir, file)

    results = model.predict(INPUT_PATH, task="segment", save=False)
    if results[0].masks is None:
        continue

    # Load the original image
    original_img = cv2.imread(INPUT_PATH)
    original_img = cv2.resize(original_img, (results[0].orig_shape[1], results[0].orig_shape[0]))

    # Create a black background
    black_bg = np.zeros_like(original_img)

    # Loop through masks for each detected object
    for mask in results[0].masks.data:

        mask_np = mask.cpu().numpy().astype(np.uint8)
        mask_np = cv2.resize(mask_np, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_np = mask_np > 0.5

        black_bg[mask_np] = original_img[mask_np]

    # Save result
    save_path = os.path.join(output_dir, file)
    cv2.imwrite(save_path, black_bg)
    print("Segmentation complete")
