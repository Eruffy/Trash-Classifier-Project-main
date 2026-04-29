"""
Test Custom Images
Test images from a custom directory and compare predictions
"""
import sys
import os
import cv2

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.ensemble_predictor import EnsemblePredictor


def test_images(image_dir):
    """Test all images in a directory"""
    
    print("="*70)
    print("CUSTOM IMAGE TESTING".center(70))
    print("="*70)
    print(f"\nTest Directory: {image_dir}\n")
    
    # Get all images
    images = [f for f in os.listdir(image_dir) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not images:
        print(f"[ERROR] No images found in {image_dir}")
        return
    
    print(f"[INFO] Found {len(images)} images")
    
    # Load models
    print("[INFO] Loading models...")
    predictor = EnsemblePredictor()
    print("[INFO] Models loaded successfully!\n")
    
    # Process each image
    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(image_dir, img_name)
        
        print("="*70)
        print(f"IMAGE {i}/{len(images)}: {img_name}")
        print("="*70)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not load image")
            continue
        
        # Get predictions
        results = predictor.predict_all(img)
        
        # Display results
        print(f"\n📊 SVM Model:")
        print(f"   Class: {results['svm']['class']}")
        print(f"   Confidence: {results['svm']['confidence']:.2f}%")
        
        print(f"\n📊 k-NN Model:")
        print(f"   Class: {results['knn']['class']}")
        print(f"   Confidence: {results['knn']['confidence']:.2f}%")
        
        print(f"\n🎯 Ensemble (Final Prediction):")
        print(f"   Class: {results['ensemble']['class']}")
        print(f"   Confidence: {results['ensemble']['confidence']:.2f}%")
        
        # Agreement
        if results['svm']['class'] == results['knn']['class']:
            print(f"\n✅ Models AGREE on classification")
        else:
            print(f"\n⚠️  Models DISAGREE on classification")
        
        print()
    
    print("="*70)
    print("TESTING COMPLETE".center(70))
    print("="*70)


def main():
    # Default test directory
    test_dir = r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\test_with"
    
    # Allow custom directory from command line
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    
    if not os.path.exists(test_dir):
        print(f"[ERROR] Directory not found: {test_dir}")
        return 1
    
    test_images(test_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
