"""
Test Custom Images with Ground Truth Analysis
Compare predictions against actual labels (from filenames)
"""
import sys
import os
import cv2
import re

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.ensemble_predictor import EnsemblePredictor


def extract_true_label(filename):
    """Extract true class from filename"""
    filename_lower = filename.lower()
    
    # Map filename patterns to classes
    if 'glass' in filename_lower:
        return 'Glass'
    elif 'cardboard' in filename_lower:
        return 'Cardboard'
    elif 'paper' in filename_lower:
        return 'Paper'
    elif 'plastic' in filename_lower and 'unknown' not in filename_lower:
        return 'Plastic'
    elif 'metal' in filename_lower:
        return 'Metal'
    elif 'trash' in filename_lower:
        return 'Trash'
    elif 'unknown' in filename_lower or 'rubber' in filename_lower:
        return 'Unknown'
    else:
        return None


def test_with_ground_truth(image_dir):
    """Test images and compare with ground truth from filenames"""
    
    print("="*80)
    print("CUSTOM IMAGE TESTING WITH GROUND TRUTH".center(80))
    print("="*80)
    print(f"\nTest Directory: {image_dir}\n")
    
    # Get all images
    images = sorted([f for f in os.listdir(image_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not images:
        print(f"[ERROR] No images found in {image_dir}")
        return
    
    print(f"[INFO] Found {len(images)} images")
    
    # Load models
    print("[INFO] Loading models...")
    predictor = EnsemblePredictor()
    print("[INFO] Models loaded successfully!\n")
    
    # Track accuracy
    results = {
        'svm_correct': 0,
        'knn_correct': 0,
        'ensemble_correct': 0,
        'total': 0,
        'unknown_images': 0,
        'details': []
    }
    
    # Process each image
    for i, img_name in enumerate(images, 1):
        img_path = os.path.join(image_dir, img_name)
        true_label = extract_true_label(img_name)
        
        print("="*80)
        print(f"IMAGE {i}/{len(images)}: {img_name}")
        print("="*80)
        print(f"📋 EXPECTED: {true_label if true_label else 'Could not parse'}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not load image\n")
            continue
        
        # Get predictions
        predictions = predictor.predict_all(img)
        
        svm_class = predictions['svm']['class']
        svm_conf = predictions['svm']['confidence']
        knn_class = predictions['knn']['class']
        knn_conf = predictions['knn']['confidence']
        ensemble_class = predictions['ensemble']['class']
        ensemble_conf = predictions['ensemble']['confidence']
        
        # Display predictions
        print(f"\n📊 SVM Model:      {svm_class} ({svm_conf:.1f}%)")
        print(f"📊 k-NN Model:     {knn_class} ({knn_conf:.1f}%)")
        print(f"🎯 Ensemble Final: {ensemble_class} ({ensemble_conf:.1f}%)")
        
        # Check correctness
        if true_label and true_label != 'Unknown':
            results['total'] += 1
            
            svm_correct = (svm_class == true_label)
            knn_correct = (knn_class == true_label)
            ensemble_correct = (ensemble_class == true_label)
            
            if svm_correct:
                results['svm_correct'] += 1
            if knn_correct:
                results['knn_correct'] += 1
            if ensemble_correct:
                results['ensemble_correct'] += 1
            
            print("\n" + "-"*80)
            print(f"SVM:      {'✅ CORRECT' if svm_correct else '❌ WRONG'}")
            print(f"k-NN:     {'✅ CORRECT' if knn_correct else '❌ WRONG'}")
            print(f"Ensemble: {'✅ CORRECT' if ensemble_correct else '❌ WRONG'}")
            print("-"*80)
            
            results['details'].append({
                'filename': img_name,
                'true_label': true_label,
                'svm': {'class': svm_class, 'conf': svm_conf, 'correct': svm_correct},
                'knn': {'class': knn_class, 'conf': knn_conf, 'correct': knn_correct},
                'ensemble': {'class': ensemble_class, 'conf': ensemble_conf, 'correct': ensemble_correct}
            })
        else:
            results['unknown_images'] += 1
            print(f"\n⚠️  Skipped accuracy calculation (Unknown/Rubber image)")
        
        print()
    
    # Print summary
    print("\n" + "="*80)
    print("ACCURACY SUMMARY".center(80))
    print("="*80)
    
    if results['total'] > 0:
        svm_acc = results['svm_correct'] / results['total'] * 100
        knn_acc = results['knn_correct'] / results['total'] * 100
        ensemble_acc = results['ensemble_correct'] / results['total'] * 100
        
        print(f"\nTested: {results['total']} images (excluding {results['unknown_images']} unknown)")
        print(f"\n📊 SVM Accuracy:      {results['svm_correct']}/{results['total']} = {svm_acc:.1f}%")
        print(f"📊 k-NN Accuracy:     {results['knn_correct']}/{results['total']} = {knn_acc:.1f}%")
        print(f"🎯 Ensemble Accuracy: {results['ensemble_correct']}/{results['total']} = {ensemble_acc:.1f}%")
        
        # Detailed breakdown
        print("\n" + "-"*80)
        print("DETAILED RESULTS:")
        print("-"*80)
        for detail in results['details']:
            status = "✅" if detail['ensemble']['correct'] else "❌"
            print(f"{status} {detail['filename'][:40]:40} | True: {detail['true_label']:10} | Pred: {detail['ensemble']['class']:10} ({detail['ensemble']['conf']:.0f}%)")
        
        # Best model
        print("\n" + "="*80)
        best_acc = max(svm_acc, knn_acc, ensemble_acc)
        if ensemble_acc == best_acc:
            print("🏆 BEST MODEL: Ensemble")
        elif svm_acc == best_acc:
            print("🏆 BEST MODEL: SVM")
        else:
            print("🏆 BEST MODEL: k-NN")
        print("="*80)
    else:
        print("\n⚠️  No testable images found (all were unknown/rubber)")
    
    return results


def main():
    test_dir = r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\test_with"
    
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]
    
    if not os.path.exists(test_dir):
        print(f"[ERROR] Directory not found: {test_dir}")
        return 1
    
    results = test_with_ground_truth(test_dir)
    
    # Provide recommendations
    if results['total'] > 0:
        ensemble_acc = results['ensemble_correct'] / results['total'] * 100
        
        print("\n" + "="*80)
        print("RECOMMENDATIONS:")
        print("="*80)
        
        if ensemble_acc < 50:
            print("❌ POOR PERFORMANCE (<50%)")
            print("   → Models severely struggle with real-world images")
            print("   → Add these test images to training dataset")
            print("   → Collect 50+ more real-world images for retraining")
        elif ensemble_acc < 70:
            print("⚠️  MODERATE PERFORMANCE (50-70%)")
            print("   → Models need more diverse training data")
            print("   → Add these test images to dataset and retrain")
        elif ensemble_acc < 90:
            print("✅ GOOD PERFORMANCE (70-90%)")
            print("   → Models work reasonably well")
            print("   → Minor improvements possible by adding test images")
        else:
            print("🏆 EXCELLENT PERFORMANCE (>90%)")
            print("   → Models generalize well to real-world images!")
        
        print("\nNext Steps:")
        print("  1. Copy correctly labeled images to dataset folders")
        print("  2. Retrain: python train_all.py --skip-augmentation")
        print("  3. Retest to verify improvement")
        print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
