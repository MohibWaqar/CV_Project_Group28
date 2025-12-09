# %% [markdown]
# # CS436 Project: Phase 1 - Two-View Reconstruction
# **Student Names:** Shahzaib Ali, Mohib Waqar
# 
# 
# ## Overview
# This notebook implements a Two-View Structure from Motion (SfM) pipeline. It takes a pair of images, extracts features, matches them, estimates camera pose, and triangulates a sparse 3D point cloud.
# 
# **Inputs:** Two images with parallax (`test14.jpg`, `test15.jpg`).  
# **Outputs:** 3D Point Cloud(`.ply`), 2D projection plots, and feature visualizations.

# %%
!pip install open3d

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d

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

# %% [markdown]
# ## 1. Image Loading
# We load the image pair (`test14.jpg` and `test15.jpg`) and resize them to ensure faster processing for feature extraction.

# %%
img1_path = 'test14.jpeg'
img2_path = 'test15.jpeg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    raise ValueError(f"Error: Could not load images. Check that '{img1_path}' and '{img2_path}' exist.")

scale = 0.5
img1 = cv2.resize(img1, None, fx=scale, fy=scale)
img2 = cv2.resize(img2, None, fx=scale, fy=scale)

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1); plt.imshow(img1_rgb); plt.title("Image 1 (Left)")
plt.subplot(1, 2, 2); plt.imshow(img2_rgb); plt.title("Image 2 (Right)")
plt.show()

# %% [markdown]
# ## 2. Feature Detection & Matching
# We use **SIFT** to detect invariant features and **FLANN** to match them. We apply **Lowe's Ratio Test** (ratio = 0.75) to filter out weak or ambiguous matches

# %%
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
pts1 = []
pts2 = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)

print(f"Total Matches: {len(matches)}")
print(f"Good Matches after Ratio Test: {len(good_matches)}")

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title("Feature Matches (Filtered)")
plt.axis('off')
plt.show()

# %% [markdown]
# ## 3. Essential Matrix & Pose Recovery
# We calculate the **Essential Matrix (E)** using RANSAC to handle outliers. From E, we decompose the relative camera pose (Rotation $R$ and Translation $t$)

# %%
K = get_manual_intrinsics(img1)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

pts1_inliers = pts1[mask.ravel() == 1]
pts2_inliers = pts2[mask.ravel() == 1]

_, R, t, mask_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, K)

print("Rotation Matrix R:\n", R);
print("Translation Vector t:\n", t);
print(f"Inliers used for pose: {pts1_inliers.shape[0]}")

# %% [markdown]
# ## 4. Triangulation
# We project the matched 2D points into 3D space using `cv2.triangulatePoints`. We then filter the cloud to remove points that are behind the camera (negative Z) or too far away

# %%
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

P2 = K @ np.hstack((R, t))

pts4D = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)

pts3D = pts4D[:3] / pts4D[3]
pts3D = pts3D.T

valid_mask = (pts3D[:, 2] > 0) & (pts3D[:, 2] < 50)
clean_pts3D = pts3D[valid_mask]

print(f"Points Triangulated: {pts3D.shape[0]}")
print(f"Valid Points (after cleanup): {clean_pts3D.shape[0]}")

# %% [markdown]
# ## 5. Visualizations
# ### Part A: 2D Scatter Plots (Top-Down & Front View)
# Per the requirements, we plot the 3D cloud projected onto 2D planes.

# %%
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(clean_pts3D[:, 0], clean_pts3D[:, 2], s=1, c='blue', alpha=0.5)
plt.title("Top-Down View (X-Z)")
plt.xlabel("X (Horizontal)")
plt.ylabel("Z (Depth)")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(clean_pts3D[:, 0], clean_pts3D[:, 1], s=1, c='red', alpha=0.5)
plt.title("Front View (X-Y)")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().invert_yaxis()
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Part B: 3D Point Cloud (Open3D)
# This cell generates an interactive 3D visualization and saves the result to `reconstruction.ply`.
# **Note:** This will open an external window. Please see the screenshot below for the result.

# %%
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(clean_pts3D[:, 0], clean_pts3D[:, 2], clean_pts3D[:, 1],
           s=1, c='blue', alpha=0.5)

ax.set_xlabel('X (Horizontal)')
ax.set_ylabel('Z (Depth)')
ax.set_zlabel('Y (Vertical)')

ax.invert_zaxis()

ax.set_title("3D Reconstruction (Matplotlib)")
plt.show()


