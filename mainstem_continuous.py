import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
import os

def find_main_stem(skeleton):
    points = np.column_stack(np.where(skeleton > 0))
    start_point = points[np.argmax(points[:, 0])]
    end_point = points[np.argmin(points[:, 0])]

    cost = np.ones(skeleton.shape, dtype=np.float32) * 1e6
    cost[skeleton > 0] = 1
    for y in range(cost.shape[0]):
        cost[y, :] -= (y / cost.shape[0])

    indices, _ = route_through_array(cost,
                                     start=(start_point[0], start_point[1]),
                                     end=(end_point[0], end_point[1]),
                                     fully_connected=True)
    indices = np.array(indices)

    main_stem = np.zeros_like(skeleton, dtype=np.uint8)
    main_stem[indices[:, 0], indices[:, 1]] = 255

    return main_stem, indices

def is_branch_point(skeleton, y, x):
    neighbors = skeleton[max(0, y-1):y+2, max(0, x-1):x+2]
    # Count how many separate branches are connected
    count = 0
    if neighbors[0,1]: count += 1
    if neighbors[2,1]: count += 1
    if neighbors[1,0]: count += 1
    if neighbors[1,2]: count += 1
    if neighbors[0,0]: count += 1
    if neighbors[0,2]: count += 1
    if neighbors[2,0]: count += 1
    if neighbors[2,2]: count += 1
    return count >= 3  # Only if there are 3 or more active connections


def boxes_overlap(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    return not (x12 < x21 or x11 > x22 or y12 < y21 or y11 > y22)

in_dir=os.path.join(os.getcwd(), 'skeleton_maize')
out_dir = os.path.join(os.getcwd(), 'maize_boxes')

for filename in os.listdir(in_dir):
    img_path = os.path.join(in_dir, filename)

    # Load and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    skeleton = binary > 0

    # Find main stem and coordinates
    main_stem_mask, stem_coords = find_main_stem(skeleton)

    # Convert to RGB
    skeleton_rgb = np.stack([skeleton * 255] * 3, axis=-1)

    # Draw red stem
    for y, x in stem_coords:
        skeleton_rgb[y, x] = [255, 0, 0]

    # Track drawn boxes
    drawn_boxes = []
    box_size = 50
    half_box = box_size // 2

    # Check each branch point
    for y, x in stem_coords:
        if is_branch_point(skeleton, y, x):
            top = max(y - half_box, 0)
            bottom = min(y + half_box, skeleton.shape[0] - 1)
            left = max(x - half_box, 0)
            right = min(x + half_box, skeleton.shape[1] - 1)

            new_box = (left, top, right, bottom)

            # Check for overlap with existing boxes
            if any(boxes_overlap(new_box, b) for b in drawn_boxes):
                continue  # Skip overlapping box

            # Check number of branch points inside box
            region = skeleton[top:bottom + 1, left:right + 1]
            branch_count = 0
            for i in range(region.shape[0]):
                for j in range(region.shape[1]):
                    if region[i, j]:
                        if is_branch_point(skeleton, top + i, left + j):
                            branch_count += 1

            if branch_count > 4:
                continue  # Too many branches inside -> skip this box

            drawn_boxes.append(new_box)
            if len(drawn_boxes) > 1:
                prev_box = drawn_boxes[-2]
                prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
                curr_center = ((new_box[0] + new_box[2]) // 2, (new_box[1] + new_box[3]) // 2)
                distance = np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])

                if distance < 100:  # If two boxes are very close, it's probably noise
                    drawn_boxes.pop()
                    continue
            skeleton_rgb[y, x] = [0, 0, 255]  # Blue dot at branch point

            # Draw green box
            skeleton_rgb[top:bottom+1, [left, right]] = [0, 255, 0]
            skeleton_rgb[[top, bottom], left:right+1] = [0, 255, 0]

    save_path_np=os.path.join(out_dir, f"{filename[:-4]}.npy")
    save_path=os.path.join(out_dir, f"{filename}")
    np.save(save_path_np, drawn_boxes)
    cv2.imwrite(save_path, skeleton_rgb)
    print(f"Saved filtered bounding box coordinates ")

    # Plot
    plt.figure(figsize=(6, 10))
    plt.imshow(skeleton_rgb)
    plt.title('Main Stem (Red), Branch Points (Blue), Non-Overlapping Boxes (Green)')
    plt.axis('off')
    plt.show()
