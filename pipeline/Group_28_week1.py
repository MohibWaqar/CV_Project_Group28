# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#big plots for the images
plt.rcParams['figure.figsize'] = (20, 10)

print("Imported Libz")

# %%
data_folder = '/kaggle/input/actual-images/Data_Set1_CV_project'

my_images_list = [
    'test1.jpeg', 'test2.jpeg', 'test3.jpeg', 'test4.jpeg',
    'test5.jpeg', 'test6.jpeg', 'test7.jpeg', 'test8.jpeg',
    'test9.jpeg', 'test10.jpeg',
]

image_paths = []
for f in my_images_list:
    image_paths.append(os.path.join(data_folder, f))


if not os.path.exists(image_paths[0]):
    print(f"Image nai mil rhi: {image_paths[0]}")
else:
    print(f"Milgayi")

# %%
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle('Sample Images', fontsize=24)

for i in range(10):
    row = i // 5
    col = i % 5
    
    img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
    
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(img_rgb)
        axes[row, col].set_title(my_images_list[i])
        axes[row, col].axis('off') 
    else:
        print(f"!! Failed to load {my_images_list[i]}")

plt.tight_layout()
plt.show()

# %%
def find_and_draw_matches(img_path1, img_path2):
    
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Error loading images {img_path1} or {img_path2}")
        return None, 0

    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    img_matches_viz = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        good_matches, 
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return img_matches_viz, len(good_matches)

print("Done")

# %%
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
fig.suptitle('Consecutive Image Feature Matching', fontsize=24)

for i in range(4):
    print(f"Matching pair {i+1}...")
    
    row = i // 2
    col = i % 2
    
    path1 = image_paths[i]
    path2 = image_paths[i+1]
    
    name1 = my_images_list[i]
    name2 = my_images_list[i+1]
    
    match_img, match_count = find_and_draw_matches(path1, path2)
    
    if match_img is not None:
        axes[row, col].imshow(match_img)
        axes[row, col].set_title(f"Matches: {name1} & {name2}\n(Found: {match_count} matches)")
        axes[row, col].axis('off')
    else:
        axes[row, col].set_title(f"Error matching {name1} & {name2}")

plt.tight_layout()
plt.show()

print("Done")


