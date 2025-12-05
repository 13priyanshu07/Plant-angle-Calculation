# 🌿 Plant Angle Calculation Pipeline

This repository contains a **computer vision pipeline** designed to detect the **main stem**, **branch junctions**, and extract **Plant Angles** . The project focuses on converting segmented plant images into structured geometric information through skeletonization, graph traversal, and bounding-box extraction.

> ⚠️ **Note:** Plant images used in this project cannot be shared due to dataset restrictions.  
> This repository contains the code only — users must provide their own segmentation masks or skeleton images.

---

## 🚀 Key Features

- Converts segmented rice/mustard plant masks into **single-pixel skeletons**
- Automatically detects:
  - Main stem path
  - Branch junctions
  - Bounding-box regions for angle measurement
- Exports:
  - Debug visualization images
  - `.npy` files containing coordinates of detected ROIs
- Modular design — each stage can run independently

---

## 🛠 Tech Stack

| Component | Technology |
|----------|------------|
| Language | Python |
| Core Libraries | `numpy`, `opencv-python`, `scikit-image`, `matplotlib` |
| Image Processing | Skeletonization,  ROI extraction |

---
## 🧠 High-Level Pipeline

1. **Input**: Segmented plant masks (binary images).
2. **Skeletonization**:
   - Convert plant masks to single-pixel-wide skeletons.
3. **Main Stem Detection**:
   - Treat skeleton as a graph.
   - Use a cost function + shortest path algorithm to trace the main stem from base to tip.
4. **Branch Point Detection**:
   - Traverse the stem and detect pixels with multiple connections (junctions).
5. **ROI Generation**:
   - Place bounding boxes around each branch junction.
   - Filter overlapping or noisy boxes.
   - Save final filtered boxes for angle analysis.
6. **Angle Calculation**:
   - Extract skeleton segments within each ROI to isolate the main stem direction and the corresponding branch direction.
   - Compute the angle between the two vectors
   - Store or visualize the computed angles
---




