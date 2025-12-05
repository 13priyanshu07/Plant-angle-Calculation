import cv2
import os


INPUT_PATH = os.path.join(os.getcwd(), 'images', 'Mustard', 'plant3_SAM.jpg')  # Your uploaded image
OUTPUT_PATH = 'plant3_SAM.png'  # Save resized image here

TARGET_WIDTH = 1836
TARGET_HEIGHT = 4080


# Load image
image = cv2.imread(INPUT_PATH)

if image is None:
    print(f"Error: Could not load image from '{INPUT_PATH}'")
else:
    # Resize image
    resized = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    # Save resized image
    cv2.imwrite(OUTPUT_PATH, resized)
    print(f"Image resized and saved as '{OUTPUT_PATH}' with shape {resized.shape[1]}x{resized.shape[0]}")
