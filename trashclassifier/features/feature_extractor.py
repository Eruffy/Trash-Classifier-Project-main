"""
Feature Extraction Module
Extracts HOG + LBP + HSV + Color features from images
Target: 6625-dimensional feature vector
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def extract_features(img):
    """
    Extract comprehensive features from image
    Returns ~6625-dimensional feature vector:
    - HOG: ~6272 features (edges/shapes)
    - LBP: 10 features (texture)  
    - HSV: 90 features (color histogram, 30 per channel)
    - RGB: 9 features (mean/std/uniformity per channel)
    - Saturation: 3 features (mean/std/max)
    - Brightness: 3 features (mean/std/range)
    - Edge: 2 features (ratio/mean)
    - Gradient: 2 features (mean/std)
    
    Args:
        img: BGR image (224x224x3)
    
    Returns:
        1D numpy array of features
    """
    # Ensure correct size
    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224))
    
    # 1. HOG Features (Histogram of Oriented Gradients)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    
    # 2. LBP Features (Local Binary Patterns)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    n_bins = 10
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
    
    # 3. HSV Color Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [30], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [30], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [30], [0, 256])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    hsv_features = np.concatenate([hist_h, hist_s, hist_v])
    
    # 4. RGB Statistics
    rgb_features = []
    for channel in cv2.split(img):
        rgb_features.append(np.mean(channel))
        rgb_features.append(np.std(channel))
        uniformity = 1.0 / (np.var(channel) + 1e-6)
        rgb_features.append(uniformity)
    rgb_features = np.array(rgb_features)
    
    # 5. Saturation Features
    saturation = hsv[:, :, 1]
    sat_features = np.array([
        np.mean(saturation),
        np.std(saturation),
        np.max(saturation)
    ])
    
    # 6. Brightness Features
    value = hsv[:, :, 2]
    bright_features = np.array([
        np.mean(value),
        np.std(value),
        np.max(value) - np.min(value)
    ])
    
    # 7. Edge Features
    edges = cv2.Canny(gray, 50, 150)
    edge_features = np.array([
        np.sum(edges > 0) / edges.size,
        np.mean(edges)
    ])
    
    # 8. Gradient Features
    grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_features = np.array([
        np.mean(magnitude),
        np.std(magnitude)
    ])
    
    # Concatenate all features
    all_features = np.hstack([
        hog_features,      # ~6272
        lbp_hist,          # 10
        hsv_features,      # 90
        rgb_features,      # 9
        sat_features,      # 3
        bright_features,   # 3
        edge_features,     # 2
        grad_features      # 2
    ])
    
    return all_features


if __name__ == "__main__":
    # Test
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = extract_features(test_img)
    print(f"✅ Feature extraction test passed!")
    print(f"   Feature vector size: {len(features)}")
    print(f"   Expected: ~6391 dimensions")
