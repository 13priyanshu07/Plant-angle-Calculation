import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

IMAGE_DIR = os.path.join(os.getcwd(), 'images', 'Mustard')
PLANT_NAMES = ['plant1', 'plant2', 'plant3']

METHOD_NAMES = ['Ground Truth', 'Input', 'SAM']
COLUMN_TITLES = METHOD_NAMES
IMAGE_EXTENSION = '.png'
OUTPUT_FILENAME = 'wheat_plant_comparison.pdf'

FIG_WIDTH = 4
FIG_HEIGHT = len(PLANT_NAMES) * 2
FIG_DPI = 300


def create_comparison_grid():
    num_rows = len(PLANT_NAMES)
    num_cols = len(METHOD_NAMES)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH, FIG_HEIGHT))

    if num_rows == 1:
        axes = [axes]

    for i, plant in enumerate(PLANT_NAMES):
        for j, method in enumerate(METHOD_NAMES):
            ax = axes[i][j] if num_rows > 1 else axes[0][j]
            ax.axis('off')

            # Set column title
            if i == 0:
                ax.set_title(COLUMN_TITLES[j], fontsize=10)

            filename = f"{plant}_{method}{IMAGE_EXTENSION}"
            image_path = os.path.join(IMAGE_DIR, filename)

            try:
                img = mpimg.imread(image_path)
                ax.imshow(img)
            except FileNotFoundError:
                print(f"Warning: Image not found at {image_path}")
                ax.text(0.5, 0.5, 'Not Found', ha='center', va='center',
                        transform=ax.transAxes, color='red', fontsize=8)

    # Adjust subplot parameters
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    plt.savefig(OUTPUT_FILENAME, dpi=FIG_DPI, bbox_inches='tight')
    print(f"Saved figure as '{OUTPUT_FILENAME}'")

    # plt.show()



if not os.path.isdir(IMAGE_DIR):
    print(f"Error: Directory '{IMAGE_DIR}' not found.")
else:
    create_comparison_grid()
