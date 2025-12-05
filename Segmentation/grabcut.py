import numpy as np
import cv2
import os

# img_dir=os.path.join(os.getcwd(), 'new')
# out_dir=os.path.join(os.getcwd(), 'out')
#
# for filename in os.listdir(img_dir):
# image_path = os.path.join(img_dir, filename)
image = cv2.imread("save_path.png")  # Load image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# Create an initial mask
mask = np.zeros(image.shape[:2], np.uint8)  # Mask size = same as image, initialized to 0

rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

bgd_model = np.zeros((1, 65), np.float64)  # Background model (65 values)
fgd_model = np.zeros((1, 65), np.float64)  # Foreground model (65 values)

# Apply GrabCut
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Convert mask
final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

# Apply mask to the original image
grabcut_output = image * final_mask[:, :, np.newaxis]  # Keep RGB format

# Save the output
# save_path = os.path.join(out_dir, filename)
cv2.imwrite("out.png", cv2.cvtColor(grabcut_output, cv2.COLOR_BGR2RGB))