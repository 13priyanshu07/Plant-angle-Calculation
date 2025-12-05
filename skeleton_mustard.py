from skimage.morphology import skeletonize
from skimage.io import imread
import numpy as np
import cv2
import os

in_dir = os.path.join(os.getcwd(), 'segmented_mustard')
out_dir = os.path.join(os.getcwd(), 'skeleton_mustard')
os.makedirs(out_dir, exist_ok=True)

for filename in os.listdir(in_dir):
    img_path = os.path.join(in_dir, filename)

    # Load and binarize
    img = imread(img_path, as_gray=True)

    binary_mask = img > 0.5  # Adjust threshold if needed

    # Compute skeleton
    skeleton= skeletonize(binary_mask)
    skeleton = skeleton.astype(np.uint8) * 255  # Convert to 0-255


    save_path = os.path.join(out_dir, filename)
    cv2.imwrite(save_path, skeleton)
    print("saved image")



