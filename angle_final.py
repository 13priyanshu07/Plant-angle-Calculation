import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import io, filters, img_as_ubyte
import matplotlib.pyplot as plt
from math import atan2, degrees
import os
from scipy.spatial import distance
import pandas as pd

# Global variables
rois = []
skeleton = None


def load_and_preprocess(image_path):
    try:
        image = io.imread(image_path, as_gray=True)
    except FileNotFoundError:
        print(f"Error: File '{image_path}' not found")
        return None

    image = img_as_ubyte(image)
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    return skeletonize(binary)


def analyze_branches():
    results = []

    for i, roi in enumerate(rois):
        x1, y1 = min(roi[0][0], roi[1][0]), min(roi[0][1], roi[1][1])
        x2, y2 = max(roi[0][0], roi[1][0]), max(roi[0][1], roi[1][1])
        roi_img = (skeleton[y1:y2, x1:x2] * 255).astype(np.uint8)

        skeleton_points = np.argwhere(roi_img > 0)
        if len(skeleton_points) < 2:
            continue

        # Sort by y to try to get bottom-most
        sorted_pts = sorted(skeleton_points, key=lambda p: p[0], reverse=True)
        junction_pt = (sorted_pts[0][1], sorted_pts[0][0])  # (x, y)

        # Trace main stem and one branch from the junction
        main_path = trace_line_from_point(roi_img, junction_pt, max_length=20)

        branch_path = []
        min_distance = 30

        for test_pt in sorted_pts[1:]:
            test_xy = (test_pt[1], test_pt[0])
            if distance.euclidean(junction_pt, test_xy) > min_distance:
                branch_path = trace_line_from_point(roi_img, test_xy, max_length=30)
                if len(branch_path) > 5:
                    break

        # Compute angles from these paths
        if len(main_path) > 5 and len(branch_path) > 5:
            def get_vector(path):
                x0, y0 = path[0]
                x1, y1 = path[-1]
                return np.array([x1 - x0, y1 - y0])

            v1 = get_vector(main_path)
            v2 = get_vector(branch_path)
            unit_v1 = v1 / np.linalg.norm(v1)
            unit_v2 = v2 / np.linalg.norm(v2)
            dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = round(np.degrees(angle_rad), 1)
        else:
            angle_deg = None

        lines = cv2.HoughLinesP(roi_img, 1, np.pi / 180, 10, minLineLength=10, maxLineGap=3)

        if lines is None or len(lines) < 2:
            results.append((i + 1, [angle_deg] if angle_deg else [], None, []))
            continue

        lines = [l[0] for l in lines]
        bottom_line = max(lines, key=lambda l: max(l[1], l[3]))  # vertical-ish line
        ref_angle = abs(degrees(atan2(bottom_line[3] - bottom_line[1], bottom_line[2] - bottom_line[0])))

        best_branch = None
        best_score = -1

        for line in lines:
            if np.array_equal(line, bottom_line):
                continue

            x0, y0, x1_, y1_ = line
            branch_angle = abs(degrees(atan2(y1_ - y0, x1_ - x0)))
            angle_diff = abs(branch_angle - ref_angle)
            angle_diff = min(angle_diff, 360 - angle_diff)

            if 10 < angle_diff < 170:
                line_len = np.hypot(x1_ - x0, y1_ - y0)
                score = angle_diff * line_len

                if score > best_score:
                    best_score = score
                    best_branch = {
                        "angle": round(angle_diff, 1),
                        "start": np.array([x0 + x1, y0 + y1]),
                        "end": np.array([x1_ + x1, y1_ + y1]),
                        "length": line_len
                    }

        if best_branch:
            angles = [best_branch["angle"]]
            other_lines = [(tuple(best_branch["start"]), tuple(best_branch["end"]))]
        else:
            angles = []
            other_lines = []

        # Merge both angle calculations
        if angle_deg is not None:
            angles.append(angle_deg)

        results.append((i + 1,
                        angles,
                        ((x1 + bottom_line[0], y1 + bottom_line[1]),
                         (x1 + bottom_line[2], y1 + bottom_line[3])),
                        other_lines))

    return results

def trace_line_from_point(skel_img, start, max_length):
    visited = set()
    path = [start]
    h, w = skel_img.shape
    for _ in range(max_length):
        x, y = path[-1]
        visited.add((x, y))
        found = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and
                    skel_img[ny, nx] and (nx, ny) not in visited):
                    path.append((nx, ny))
                    found = True
                    break
            if found:
                break
        if not found:
            break
    return path


def plot_and_save_results(results, image_save_path, csv_save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(skeleton, cmap='gray')

    junction_infos = []

    junction_positions = []
    for idx, roi in enumerate(rois):
        x1, y1 = min(roi[0][0], roi[1][0]), min(roi[0][1], roi[1][1])
        x2, y2 = max(roi[0][0], roi[1][0]), max(roi[0][1], roi[1][1])
        center_y = (y1 + y2) / 2
        junction_positions.append((idx, center_y))

    # Sort junctions by y coordinate descending
    sorted_junctions = sorted(junction_positions, key=lambda x: x[1], reverse=True)

    # Mapping: old idx -> new label
    idx_to_label = {old_idx: new_label for new_label, (old_idx, _) in enumerate(sorted_junctions, start=1)}

    for result in results:
        roi_num, angles, ref_line, other_lines = result
        if ref_line is None:
            continue

        ax.plot([ref_line[0][0], ref_line[1][0]],
                [ref_line[0][1], ref_line[1][1]],
                'r-', linewidth=2)

        for line, angle in zip(other_lines, angles):
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], 'b-', linewidth=1)
            mid_x = (line[0][0] + line[1][0]) / 2
            mid_y = (line[0][1] + line[1][1]) / 2
            ax.text(mid_x, mid_y, f"{angle}°", color='yellow', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

        x1 = min(rois[roi_num - 1][0][0], rois[roi_num - 1][1][0])
        y1 = min(rois[roi_num - 1][0][1], rois[roi_num - 1][1][1])
        new_label = idx_to_label[roi_num - 1]
        ax.text(x1, y1 - 10, f"Junction {new_label}",
                color='green', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, pad=2))

        # Save to CSV info
        if angles:
            junction_infos.append({"Junction": new_label, "Angle": angles[0]})

    ax.set_title("Branch Angle Analysis")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save CSV
    df = pd.DataFrame(junction_infos)
    if not df.empty:
        df = df.sort_values("Junction")
    df.to_csv(csv_save_path, index=False)


# Main
in_dir1=os.path.join(os.getcwd(), 'mustard_boxes')
in_dir2=os.path.join(os.getcwd(), 'skeleton_mustard')
out_dir = os.path.join(os.getcwd(), 'mustard_angles')


for filename1, filename2 in zip(os.listdir(in_dir1), os.listdir(in_dir2)):
    boxes_path = os.path.join(in_dir1, filename1)
    img_path2 = os.path.join(in_dir2, filename2)
    skeleton = load_and_preprocess(img_path2)
    if skeleton is None:
        exit()

    if not os.path.exists(boxes_path):
        print(f"Error: Bounding box file not found at {boxes_path}")
        exit()

    # Convert bounding box two points format
    raw_boxes = np.load(boxes_path, allow_pickle=True)
    rois = [ [(box[0], box[1]), (box[2], box[3])] for box in raw_boxes.tolist() ]

    print("Automatically loaded bounding boxes. Starting analysis...")
    results = analyze_branches()

    image_output_path = os.path.join(out_dir, filename2.replace('.png', '_labeled.png'))
    csv_output_path = os.path.join(out_dir, filename2.replace('.png', '_angles.csv'))

    # Save the results
    plot_and_save_results(results, image_output_path, csv_output_path)
