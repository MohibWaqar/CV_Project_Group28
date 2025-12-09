import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import glob
import os


%matplotlib inline

def get_manual_intrinsics(image):
    h, w = image.shape[:2]
    K = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    return K

print("Libraries loaded ")
import os
import glob

print("--- Debugging Path ---")

# 1. Check if we should use 'MyDrive' or 'My Drive'
base_path_nospace = '/content/drive/MyDrive'
base_path_space = '/content/drive/My Drive'

if os.path.exists(base_path_nospace):
    base = base_path_nospace
    print(f" Found Drive at: {base}")
elif os.path.exists(base_path_space):
    base = base_path_space
    print(f"Found Drive at: {base}")
else:
    print("Critical Error: Could not find 'MyDrive'. Is Drive mounted?")
    base = None

if base:
    # 2. Check for 'Cv pics' folder (Case Sensitive!)
    cv_pics_path = os.path.join(base, 'Cv pics')

    if os.path.exists(cv_pics_path):
        print(f"Found folder: 'Cv pics'")

        # 3. Check for 'cv_new' folder
        cv_new_path = os.path.join(cv_pics_path, 'cv_new')
        if os.path.exists(cv_new_path):
            print(f"Found folder: 'cv_new'")

            # 4. List the first few files to check extensions
            files = os.listdir(cv_new_path)
            print(f" Files inside 'cv_new' ({len(files)} total):")
            print(files[:5]) # Show first 5
        else:
            print(f"❌ Error: Could not find 'cv_new' inside '{cv_pics_path}'")
            print("Contents of 'Cv pics':", os.listdir(cv_pics_path))
    else:
        print(f"❌ Error: Could not find 'Cv pics' inside '{base}'")
        # List what IS there to help you spot the typo
        # print("Folders in Drive:", os.listdir(base))


# ---------------------------------------------------------
# 1. Mount Google Drive
# ---------------------------------------------------------
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted successfully.")
except:
    print("Not running in Colab (or Drive already mounted). Skipping mount.")

# ---------------------------------------------------------
# 2. Define Path to 'cv_new'
# ---------------------------------------------------------

folder_path = '/content/drive/MyDrive/Cv pics/cv_new/*.jpeg'

# ---------------------------------------------------------
# 3. Get File Paths and Sort
# ---------------------------------------------------------
image_paths = sorted(glob.glob(folder_path))


print(f"\nFound {len(image_paths)} images.")
if len(image_paths) > 0:
    print("First all images in sequence:")
    for p in image_paths[:]:
        print(os.path.basename(p))
else:
    print(" ERROR: No images found! Check the folder path and file extension.")

# ---------------------------------------------------------
# 4. Load and Resize Images
# ---------------------------------------------------------
images = []
scale = 0.5  # Consistent with your Week 2 settings

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not read {path}")
        continue

    img = cv2.resize(img, None, fx=scale, fy=scale)
    images.append(img)

print(f"\n✅ Successfully loaded {len(images)} images into list 'images'.")
print(f"Image 1 is images[0], Image 2 is images[1], etc.")


# --- 1. Get Intrinsics ---
def get_intrinsics(image):
    h, w = image.shape[:2]
    # Approximation: Focal length = Image Width
    K = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    return K

# --- 2. Match Features ---
def get_matches(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Flann Matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    pts1 = []
    pts2 = []

    # Lowe's Ratio Test
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    return pts1, pts2, kp1, kp2, des1, des2, good_matches

# --- 3. Triangulate Points ---
def triangulate_points(P1, P2, pts1, pts2):
    # Triangulate
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert Homogeneous (4D) to Cartesian (3D)
    pts3D = pts4D[:3] / pts4D[3]
    pts3D = pts3D.T

    return pts3D

import matplotlib.pyplot as plt

print("--- PHASE 1: Initialization (Image 0 & Image 1) ---")

# 1. Setup
img0 = images[0]
img1 = images[1]
K = get_intrinsics(img0)

# 2. Get Matches
pts0, pts1, kp0, kp1, des0, des1, matches_0_1 = get_matches(img0, img1)
print(f"Matches found: {len(matches_0_1)}")

# 3. Essential Matrix
E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Filter by RANSAC mask
mask = mask.ravel() == 1
pts0_inliers = pts0[mask]
pts1_inliers = pts1[mask]

# Get descriptors for inliers
matches_inliers = [matches_0_1[i] for i in range(len(matches_0_1)) if mask[i]]
des1_inliers = np.array([des1[m.trainIdx] for m in matches_inliers])

# 4. Recover Pose (R, t)
_, R, t, mask_pose = cv2.recoverPose(E, pts0_inliers, pts1_inliers, K)

# 5. Triangulate Initial Cloud
P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = K @ np.hstack((R, t))

points_3d = triangulate_points(P0, P1, pts0_inliers, pts1_inliers)

# 6. Filter garbage points (Negative depth or too far)
valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 50)

pts1_inliers = pts1_inliers[valid_mask] 
points_3d = points_3d[valid_mask]
des1_inliers = des1_inliers[valid_mask]

# --- SAVE GLOBAL STATE FOR PHASE 2 LOOP ---
global_points_3d = points_3d
global_descriptors = des1_inliers
# We also save pts1_inliers to help build the registry in Phase 2
global_pts1_inliers = pts1_inliers

camera_poses = [np.eye(4), np.vstack((np.hstack((R, t)), [0,0,0,1]))]

print(f"Initialization Complete.")
print(f"Initial Point Cloud: {global_points_3d.shape[0]} points.")
print(f"Saved {len(global_descriptors)} descriptors for PnP matching.")

# --- VISUALIZATION ---
plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.scatter(points_3d[:,0], points_3d[:,2], points_3d[:,1], s=1, c='blue')
ax.set_title("Initial 3D Seed (Week 2 Result)")
ax.set_xlabel('X'); ax.set_ylabel('Z (Depth)'); ax.set_zlabel('Y')
ax.invert_zaxis()
plt.show()

# --- FIX: CREATE THE MISSING REGISTRY ---
# We need to build 'map_2d_3d' which maps:
# [Feature Index in Image 1] -> [Index in global_points_3d]

print("Building 2D-3D Registry...")

map_2d_3d = {}

# We need to trace back which matches survived all the filters
# 1. matches_0_1: All matches
# 2. mask: RANSAC inliers
# 3. valid_mask: Depth/Cheirality valid points

# Re-construct the list of matches that made it into the final cloud
final_matches = []

# Filter 1: RANSAC (Same logic as Phase 1)
matches_ransac = [matches_0_1[i] for i in range(len(matches_0_1)) if mask.ravel()[i]]

# Filter 2: Depth Validation (Same logic as Phase 1)
# Note: 'valid_mask' in Phase 1 corresponds to the indices of 'matches_ransac'
count = 0
for i, is_valid in enumerate(valid_mask):
    if is_valid:
        # This match survived both RANSAC and Depth check
        m = matches_ransac[i]

        # Map the Feature Index (trainIdx is Image 1) to the 3D Point Index
        # The 3D Point Index is 'count' because global_points_3d is a compact list of valid points
        map_2d_3d[m.trainIdx] = count

        count += 1

print(f"✅ Registry created: {len(map_2d_3d)} connections ready for Phase 2.")



print("\n--- PHASE 2: Incremental Reconstruction Loop ---")

# =========================================================
# STEP 0: Bridge Phase 1 to Phase 2 (Create the Map)
# =========================================================

# We need to map: [Feature Index in Image 1] -> [Index in global_points_3d]
# Since we don't have the original indices easily from Phase 1's filtered arrays,
# we will use a KD-Tree (Flann) to reverse-lookup the keypoints.

current_map_2d_3d = {}

# Detect features in Image 1 again to get the full list of indices
kp1_full, des1_full = cv2.SIFT_create().detectAndCompute(images[1], None)

# Find which keypoint in 'kp1_full' is close to 'pts1_inliers'
# This connects our 3D points back to the standard feature indices
for i in range(len(pts1_inliers)):
    # pts1_inliers[i] is the 2D coordinate of global_points_3d[i]
    pt = pts1_inliers[i]

    # Brute force search for the index (Sequence is small, so it's fast enough)
    # In a better system, we'd track indices, but this works for homework.
    for k, kp in enumerate(kp1_full):
        if np.linalg.norm(np.array(kp.pt) - pt) < 0.1: # If coordinates match
            current_map_2d_3d[k] = i # Feature k maps to 3D point i
            break

print(f"Registry initialized with {len(current_map_2d_3d)} correspondences.")

# Setup for Loop
prev_img = images[1]
prev_kp = kp1_full
prev_des = des1_full
prev_P = P1 # From Phase 1

# =========================================================
# THE LOOP
# =========================================================
for i in range(2, len(images)):
    curr_img = images[i]
    print(f"\nProcessing Image {i} ({os.path.basename(image_paths[i])})...")

    # 1. Match Features (Previous vs Current)
    sift = cv2.SIFT_create()
    kp_curr, des_curr = sift.detectAndCompute(curr_img, None)

    # Match
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(prev_des, des_curr, k=2)

    # 2. Build PnP Arrays
    object_points = []
    image_points = []
    pnp_matches = [] # To track which match corresponds to which PnP point

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            # If the feature in Prev Image has a known 3D point
            if m.queryIdx in current_map_2d_3d:
                global_idx = current_map_2d_3d[m.queryIdx]
                object_points.append(global_points_3d[global_idx])
                image_points.append(kp_curr[m.trainIdx].pt)
                pnp_matches.append(m)

    object_points = np.array(object_points)
    image_points = np.array(image_points)

    print(f"  - Found {len(object_points)} correspondences for PnP.")

    if len(object_points) < 6:
        print("   Not enough points for PnP. Skipping frame.")
        continue

    # 3. Solve PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, None,
        reprojectionError=8.0, confidence=0.99
    )

    if not success:
        print("   PnP Failed.")
        continue

    R, _ = cv2.Rodrigues(rvec)
    t = tvec
    current_P = K @ np.hstack((R, t))

    pose_4x4 = np.eye(4)
    pose_4x4[:3, :3] = R
    pose_4x4[:3, 3] = t.ravel()
    camera_poses.append(pose_4x4)

    # 4. Update Registry (Carry over PnP Inliers)
    new_map_2d_3d = {}

    if inliers is not None:
        for idx in inliers.ravel():
            # 'idx' is the index in 'object_points'/'pnp_matches'
            m = pnp_matches[idx]

            # The feature in Previous (m.queryIdx) mapped to a global 3D point
            global_3d_idx = current_map_2d_3d[m.queryIdx]

            # Now, the feature in Current (m.trainIdx) maps to that SAME global point
            new_map_2d_3d[m.trainIdx] = global_3d_idx

    # 5. Triangulate NEW Points
    # Find matches that are GOOD but were NOT in the existing map
    pts_prev_new = []
    pts_curr_new = []
    triangulation_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            # If it wasn't in the map, it's a candidate for a NEW 3D point
            if m.queryIdx not in current_map_2d_3d:
                pts_prev_new.append(prev_kp[m.queryIdx].pt)
                pts_curr_new.append(kp_curr[m.trainIdx].pt)
                triangulation_matches.append(m)

    if len(pts_prev_new) > 0:
        pts_prev_new = np.array(pts_prev_new)
        pts_curr_new = np.array(pts_curr_new)

        pts4d = cv2.triangulatePoints(prev_P, current_P, pts_prev_new.T, pts_curr_new.T)
        pts3d = (pts4d[:3] / pts4d[3]).T

        # Filter
        valid_mask = (pts3d[:, 2] > 0) & (pts3d[:, 2] < 50)
        clean_pts3d = pts3d[valid_mask]

        # Add to Global Cloud
        start_idx = len(global_points_3d)
        global_points_3d = np.vstack((global_points_3d, clean_pts3d))

        # Update Registry with NEW points
        filtered_indices = np.where(valid_mask)[0] # Indices in pts3d that are valid

        for k in filtered_indices:
            m = triangulation_matches[k]
            new_global_idx = start_idx + list(filtered_indices).index(k) # Simple counter logic
            # Actually, simpler: start_idx maps to clean_pts3d[0], start_idx+1 maps to [1]...

            real_global_idx = start_idx + np.where(filtered_indices == k)[0][0]
            new_map_2d_3d[m.trainIdx] = real_global_idx

        print(f"  - Added {len(clean_pts3d)} new points. Total: {len(global_points_3d)}")

    # 6. Handover for next iteration
    current_map_2d_3d = new_map_2d_3d
    prev_kp = kp_curr
    prev_des = des_curr
    prev_P = current_P

print("\n Reconstruction Complete.")



print("--- PHASE 3: Visualization & Export ---")

# =========================================================
# 1. Extract Camera Trajectory
# =========================================================
camera_positions = []

print("Calculating camera positions...")
for pose in camera_poses:
    # pose is a 4x4 Matrix: [R | t]
    # This transforms World Points -> Camera Frame.
    # To get the Camera Center in World Frame, we calculate: C = -R^T * t
    R = pose[:3, :3]
    t = pose[:3, 3]

    # Calculate position
    center = -R.T @ t
    camera_positions.append(center)

camera_positions = np.array(camera_positions)

# =========================================================
# 2. Matplotlib Visualization (Interactive-ish)
# =========================================================
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# A. Plot the Point Cloud
# We plot every 2nd point to keep the viewer responsive
step = 2
pts = global_points_3d[::step]

# Filter visual noise for the plot (optional)
# Just to make the plot look cleaner, we ignore extreme outliers
mask = (pts[:, 2] < np.percentile(pts[:, 2], 95)) & (pts[:, 2] > np.percentile(pts[:, 2], 5))
pts = pts[mask]

ax.scatter(pts[:,0], pts[:,2], pts[:,1], s=0.5, c='blue', alpha=0.2, label='Sparse Cloud')

# B. Plot the Camera Path
ax.plot(camera_positions[:,0], camera_positions[:,2], camera_positions[:,1],
        c='red', linewidth=2, marker='o', markersize=6, label='Camera Path')

# Mark Start and End
ax.text(camera_positions[0,0], camera_positions[0,2], camera_positions[0,1], "Start", color='green', fontsize=12)
ax.text(camera_positions[-1,0], camera_positions[-1,2], camera_positions[-1,1], "End", color='red', fontsize=12)

# C. Formatting
ax.set_title(f"Final Reconstruction: {len(camera_poses)} Cameras, {len(global_points_3d)} Points")
ax.set_xlabel('X (Horizontal)')
ax.set_ylabel('Z (Depth)')
ax.set_zlabel('Y (Vertical)')
ax.legend()

# Invert axes to match standard vision coordinate systems
ax.invert_zaxis()
plt.show()

# =========================================================
# 3. Export to PLY (For MeshLab / Open3D)
# =========================================================
output_filename = "reconstruction_week3.ply"

# Create Open3D object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(global_points_3d)

# Save
o3d.io.write_point_cloud(output_filename, pcd)
print(f"\n✅ Saved 3D point cloud to '{output_filename}'")

# Download Trigger (Colab)
try:
    from google.colab import files
    files.download(output_filename)
    print("Download prompt started. Open this file in MeshLab or Open3D viewer!")
except ImportError:
    print("Check your local folder for the .ply file.")


filename = "crazy_cloud.obj" 
print(f"Parsing {filename} manually...")

vertices = []
colors = []

# 1. Custom Parser for Point Cloud OBJ
try:
    with open(filename, 'r') as f:
        for line in f:
            # OBJ lines for vertices start with 'v'
            if line.startswith('v '):
                parts = line.strip().split()
                # usually: v x y z [r g b]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])

                # Check if there are extra numbers for colors (some software does this)
                if len(parts) >= 7:
                    r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                    colors.append([r, g, b])

    vertices = np.array(vertices)
    print(f"Successfully parsed {len(vertices)} points.")

    if len(vertices) == 0:
        print(" Error: File opened, but no vertices found (lines starting with 'v ')")
    else:
        # 2. Visualize
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Subsample for performance
        step = 1 if len(vertices) < 10000 else 5
        pts = vertices[::step]

        # Handle colors
        if len(colors) == len(vertices):
            col = np.array(colors)[::step]
            # Normalize colors if they are 0-255 instead of 0-1
            if col.max() > 1.0:
                col = col / 255.0
        else:
            col = 'blue'

        ax.scatter(pts[:,0], pts[:,2], pts[:,1], s=0.5, c=col, alpha=0.5)

        ax.set_title(f"Custom OBJ Viewer: {len(vertices)} Points")
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.invert_zaxis() 
        plt.show()

except FileNotFoundError:
    print(f" Error: Could not find file '{filename}'. Check the name!")
except Exception as e:
    print(f" Unexpected Error: {e}")

