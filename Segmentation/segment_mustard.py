import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
import torch

# Initialize SAM
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# Directories
input_dir = os.path.join(os.getcwd(), '..', 'Maize104', 'data')  # Directory with original images
output_dir = os.path.join(os.getcwd(), '..', 'segmented_maize')  # Directory to save results
os.makedirs(output_dir, exist_ok=True)


# Function to collect point prompts for an image
def get_user_prompts(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Left-click: Plant | Right-click: Background | Close window to segment")
    points, labels = [], []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if event.button == 1:  # Left click (plant)
                points.append([x, y])
                labels.append(1)
                plt.scatter(x, y, color='green', marker='o', s=100)
            elif event.button == 3:  # Right click (background)
                points.append([x, y])
                labels.append(0)
                plt.scatter(x, y, color='red', marker='x', s=100)
            plt.draw()

    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return np.array(points), np.array(labels)


# Process each image
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Get prompts for the current image
        print(f"Select points for: {filename}")
        points_np, labels_np = get_user_prompts(image)

        # Generate mask
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=False,  # Only 1 mask
        )
        mask = masks[0]

        # Refine mask
        kernel = np.ones((3, 3), np.uint8)
        refined_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Apply mask
        segmented_image = image.copy()
        segmented_image[~refined_mask.astype(bool)] = [0, 0, 0]  # Black background

        # Save result
        output_path = os.path.join(output_dir, f"segmented_{filename}")
        cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

print("All images segmented!")