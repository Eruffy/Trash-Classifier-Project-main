"""
Visualize Predictions on Test Images
Shows images with predictions overlaid for analysis
"""
import sys
import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.ensemble_predictor import EnsemblePredictor


def visualize_predictions(image_dir, output_dir=None):
    """Visualize predictions on all test images"""
    
    # Get all images
    images = sorted([f for f in os.listdir(image_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not images:
        print(f"[ERROR] No images found in {image_dir}")
        return
    
    print(f"[INFO] Found {len(images)} images")
    print("[INFO] Loading models...")
    predictor = EnsemblePredictor()
    print("[INFO] Models loaded!\n")
    
    # Create figure
    n_images = len(images)
    cols = 3
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_name in enumerate(images):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            ax.text(0.5, 0.5, f'Failed to load\n{img_name}', 
                   ha='center', va='center')
            ax.axis('off')
            continue
        
        # Get predictions
        results = predictor.predict_all(img)
        
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display image
        ax.imshow(img_rgb)
        ax.axis('off')
        
        # Create title with predictions
        svm_class = results['svm']['class']
        svm_conf = results['svm']['confidence']
        knn_class = results['knn']['class']
        knn_conf = results['knn']['confidence']
        ensemble_class = results['ensemble']['class']
        ensemble_conf = results['ensemble']['confidence']
        
        agree = "✓" if svm_class == knn_class else "✗"
        
        title = f"{img_name[:30]}...\n"
        title += f"SVM: {svm_class} ({svm_conf:.0f}%)\n"
        title += f"kNN: {knn_class} ({knn_conf:.0f}%)\n"
        title += f"Final: {ensemble_class} ({ensemble_conf:.0f}%) {agree}"
        
        ax.set_title(title, fontsize=9, pad=10)
        
        # Add colored border based on agreement
        if svm_class == knn_class:
            color = 'green'
        else:
            color = 'red'
        
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    # Hide empty subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'predictions_visualization.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[INFO] Visualization saved to: {output_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("ANALYSIS TIPS:")
    print("="*70)
    print("🟢 Green border = Models AGREE (more confident)")
    print("🔴 Red border = Models DISAGREE (uncertain)")
    print("\nIf predictions seem wrong, the models may need:")
    print("  1. More diverse training data (different angles, lighting)")
    print("  2. Less aggressive augmentation (may be creating artifacts)")
    print("  3. Retraining with these test images added to dataset")
    print("="*70)


def main():
    test_dir = r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\test_with"
    output_dir = r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\trashclassifier\saved_models"
    
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    
    if not os.path.exists(test_dir):
        print(f"[ERROR] Directory not found: {test_dir}")
        return 1
    
    visualize_predictions(test_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
