"""
Single Image Prediction Script
Predicts the class of a single image using both SVM and k-NN models.
"""
import sys
import os
import argparse
import cv2

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.ensemble_predictor import EnsemblePredictor
from config import Config


def main():
    parser = argparse.ArgumentParser(description='Predict waste class for a single image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--show', action='store_true', help='Display the image')
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found: {args.image_path}")
        sys.exit(1)
    
    # Load image
    print(f"[INFO] Loading image: {args.image_path}")
    image = cv2.imread(args.image_path)
    
    if image is None:
        print(f"[ERROR] Could not load image: {args.image_path}")
        sys.exit(1)
    
    # Load predictor
    print("[INFO] Loading models...")
    predictor = EnsemblePredictor()
    
    # Get predictions
    print("[INFO] Making predictions...")
    results = predictor.predict_all(image)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS".center(60))
    print("="*60)
    
    print("\n📊 SVM Model:")
    print(f"   Class: {results['svm']['class']}")
    print(f"   Confidence: {results['svm']['confidence']:.2f}%")
    print(f"   Class ID: {results['svm']['class_id']}")
    
    print("\n📊 k-NN Model:")
    print(f"   Class: {results['knn']['class']}")
    print(f"   Confidence: {results['knn']['confidence']:.2f}%")
    print(f"   Class ID: {results['knn']['class_id']}")
    
    print("\n🎯 Ensemble (Final Prediction):")
    print(f"   Class: {results['ensemble']['class']}")
    print(f"   Confidence: {results['ensemble']['confidence']:.2f}%")
    print(f"   Class ID: {results['ensemble']['class_id']}")
    
    # Agreement status
    if results['svm']['class'] == results['knn']['class']:
        print("\n✅ Models AGREE on classification")
    else:
        print("\n⚠️  Models DISAGREE on classification")
    
    print("="*60 + "\n")
    
    # Display image if requested
    if args.show:
        # Add prediction text to image
        display_img = image.copy()
        text = f"{results['ensemble']['class']} ({results['ensemble']['confidence']:.1f}%)"
        cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Prediction', display_img)
        print("[INFO] Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
