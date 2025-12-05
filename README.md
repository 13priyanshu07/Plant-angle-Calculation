# Plant Angle Calculation Pipeline 🌱

This repository contains a classical computer vision pipeline to **calculate plant branch angles from images**.  
The project was built as part of a lab-based research project using datasets provided by an agricultural research organization (images are **not included** due to data-sharing restrictions).

The pipeline takes segmented plant images as input, skeletonizes them, identifies the **main stem** and **branch junctions**, and generates **regions of interest (ROIs)** around each junction so that angles can be measured reliably.

---

## ✨ Highlights

- End-to-end image processing pipeline for plant phenotyping.
- Uses **image skeletonization** to convert plant masks into one-pixel-wide structures.
- Automatically detects:
  - Main stem path
  - Branch points along the stem
  - Non-overlapping bounding boxes (ROIs) around each junction
- Saves ROIs as both:
  - Annotated images (for visualization)
  - `.npy` files (for downstream angle computation scripts)

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

Angle computation itself can be performed in a later step using the skeleton inside each ROI.

---

## 🛠 Tech Stack

- **Language**: Python
- **Libraries**:
  - `numpy`
  - `opencv-python`
  - `scikit-image`
  - `matplotlib`

---

## 📁 Repository Structure

```text
Plant-angle-Calculation/
│
├── Segmentation/              # (Optional) Segmentation-related code / notebooks (if added)
├── skeleton_mustard.py        # Script to skeletonize segmented mustard plant images
├── mainstem_continuous.py     # Script to detect main stem, branch points & ROIs from skeletons
├── README.md                  # Project documentation (this file)
│
├── segmented_mustard/         # (Expected) Input: binary/segmented mustard images (not tracked)
├── skeleton_mustard/          # Output: skeletonized mustard images (created by skeleton_mustard.py)
├── skeleton_maize/            # Input: skeletonized maize images (for main stem & ROI detection)
└── maize_boxes/               # Output: annotated images + .npy ROI files (created by mainstem_continuous.py)

