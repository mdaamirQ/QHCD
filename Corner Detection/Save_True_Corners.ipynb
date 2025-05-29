import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

# Define dataset paths
dataset_path = Path('./drive/MyDrive/QHCD/UrbanDataset')
images_path = dataset_path / 'Images'
gt_path = dataset_path / 'Ground Truth'

# List all image files
image_files = sorted(images_path.glob('*.*'))  # supports jpg, png, etc.

for image_file in image_files:
    # Load image in grayscale
    img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)

    # Load corresponding ground truth corner file
    gt_file = gt_path / (image_file.stem + '.txt')
    if not gt_file.exists():
        print(f"Ground truth file missing for: {image_file.name}")
        continue

    # Load corner coordinates
    try:
        corners = np.loadtxt(gt_file)
        if corners.ndim == 1:  # Single point case
            corners = corners[np.newaxis, :]
    except Exception as e:
        print(f"Failed to read {gt_file.name}: {e}")
        continue

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.scatter(corners[:, 1], corners[:, 0], c='green', s=20, marker='o', label='True Corners')
    ax.set_title(f"Ground Truth Corners: {image_file.name}")
    ax.axis('off')
    ax.legend()

    # Save the figure in Ground Truth/ folder
    output_file = gt_path / f"{image_file.stem}_ground_truth.jpg"
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_file}")
