# Indoor 3D Reconstruction & Virtual Tour (CS436)

This project implements an Incremental Structure-from-Motion (SfM) pipeline to reconstruct a 3D scene from 2D images and visualizes the result in an interactive WebGL tour.

## Project Structure
* **`pipeline/`**: Python scripts for Feature Extraction, Matching, PnP, and Bundle Adjustment.
* **`web_viewer/`**: A Three.js application combining the dense point cloud with aligned camera poses.

## How to Run the Viewer
1.  Navigate to the `web_viewer` folder.
2.  Start a local server (browser security requires this):
    ```bash
    python -m http.server
    ```
3.  Open `localhost:8000` in your browser.
4.  **Controls**:
    * `WASD`: Fly
    * `Arrows`: Look around
    * `Click Blue Dots`: Auto-navigate to image overlay

## Dependencies (note: since we didnt push .ply file, we cannot run it, if you want to run, please ask for.ply from metashape file, please and thankyou)

```text
CS436_Final_Project/
│
├── README.md                # Project documentation and usage guide
├── requirements.txt         # List of required Python libraries
│
├── pipeline/                # Python SfM Reconstruction Pipeline
│   ├── Group_28_week1.py    # Feature Detection & Matching (SIFT)
│   ├── Group_28_week2.py    # Two-View Reconstruction & Triangulation
│   ├── Group_28_week3.py    # Incremental SfM Loop
│
├── web_viewer/              # Interactive Virtual Tour (Three.js)
│   ├── index.html           # Main viewer application
│   ├── cameras.json         # Aligned Camera Poses
│   └── cameras_final.json
│
└── dataset/             # Original Input Image Sequence
    ├── test1.jpeg
    ├── test2.jpeg
    └── ...
```