"""
Test All Models on Custom Test Data
Tests both SVM and KNN strategies (original + augmented combinations) on images in test_with folder
"""
import os
import sys
import cv2
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config
from scripts.hybrid_predictor import HybridPredictor


def test_on_folder(test_folder):
    """Test all images in the test folder with both strategies"""
    
    print("\n" + "="*70)
    print("TESTING MODELS ON CUSTOM TEST DATA".center(70))
    print("="*70)
    print(f"\nTest Folder: {test_folder}\n")
    
    # Get all image files
    image_files = [f for f in os.listdir(test_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("[ERROR] No image files found in test folder!")
        return
    
    print(f"Found {len(image_files)} test images\n")
    
    # Initialize predictor
    predictor = HybridPredictor()
    
    # Store results
    results = []
    
    # Test each image
    print("="*70)
    print("PREDICTIONS".center(70))
    print("="*70 + "\n")
    
    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(test_folder, img_file)
        
        print(f"[{idx}/{len(image_files)}] {img_file}")
        print("-" * 70)
        
        # Get predictions from both strategies
        prediction = predictor.predict_both(img_path)
        
        if 'error' in prediction.get('svm', {}):
            print(f"  ERROR: {prediction['svm']['error']}")
            print()
            continue
        
        svm = prediction['svm']
        knn = prediction['knn']
        
        # Display results
        print(f"  SVM (All Aug):        {svm['predicted_class']:<12} | "
              f"Super: {svm['super_class']:<12} | Conf: {svm['final_confidence']:>6.1%}")
        print(f"  KNN (Orig+Aug):       {knn['predicted_class']:<12} | "
              f"Super: {knn['super_class']:<12} | Conf: {knn['final_confidence']:>6.1%}")
        
        # Show agreement
        if prediction['agreement']:
            print(f"  ✅ Agreement: {svm['predicted_class']}")
        else:
            print(f"  ⚠️  Disagreement: SVM={svm['predicted_class']} vs KNN={knn['predicted_class']}")
        
        print()
        
        # Store results
        results.append({
            'filename': img_file,
            'svm': svm,
            'knn': knn,
            'agreement': prediction['agreement']
        })
    
    # Generate summary
    print("\n" + "="*70)
    print("SUMMARY".center(70))
    print("="*70 + "\n")
    
    # Agreement rate
    agreements = sum(1 for r in results if r['agreement'])
    agreement_rate = agreements / len(results) * 100 if results else 0
    
    print(f"Total Images Tested: {len(results)}")
    print(f"Agreement Rate: {agreements}/{len(results)} ({agreement_rate:.1f}%)")
    print()
    
    # Class distribution by strategy
    print("Predictions by Strategy:")
    print("-" * 70)
    
    svm_predictions = defaultdict(int)
    knn_predictions = defaultdict(int)
    
    for r in results:
        svm_predictions[r['svm']['predicted_class']] += 1
        knn_predictions[r['knn']['predicted_class']] += 1
    
    all_classes = sorted(set(list(svm_predictions.keys()) + list(knn_predictions.keys())))
    
    print(f"{'Class':<15} {'SVM (All Aug)':<15} {'KNN (Orig+Aug)':<15}")
    print("-" * 70)
    for cls in all_classes:
        svm_count = svm_predictions.get(cls, 0)
        knn_count = knn_predictions.get(cls, 0)
        print(f"{cls:<15} {svm_count:<15} {knn_count:<15}")
    
    print()
    
    # Average confidence
    svm_avg_conf = np.mean([r['svm']['final_confidence'] for r in results])
    knn_avg_conf = np.mean([r['knn']['final_confidence'] for r in results])
    
    print("Average Confidence:")
    print(f"  SVM Strategy:  {svm_avg_conf:.1%}")
    print(f"  KNN Strategy:  {knn_avg_conf:.1%}")
    
    # Show disagreements
    disagreements = [r for r in results if not r['agreement']]
    if disagreements:
        print(f"\nDisagreements ({len(disagreements)}):")
        print("-" * 70)
        for r in disagreements:
            print(f"  {r['filename']}")
            print(f"    SVM: {r['svm']['predicted_class']} ({r['svm']['final_confidence']:.1%})")
            print(f"    KNN: {r['knn']['predicted_class']} ({r['knn']['final_confidence']:.1%})")
    
    print("\n" + "="*70)
    print("[INFO] Testing complete!")
    print("="*70 + "\n")
    
    return results


def compare_strategies(results):
    """Generate detailed comparison between strategies"""
    
    print("\n" + "="*70)
    print("STRATEGY COMPARISON".center(70))
    print("="*70 + "\n")
    
    print("SVM Strategy (All Augmented):")
    print("  - Super-class: Augmented SVM (75.33% validation accuracy)")
    print("  - Fiber: Augmented SVM (96.50% validation accuracy)")
    print("  - Rigid: Augmented SVM (96.50% validation accuracy)")
    print()
    
    print("KNN Strategy (Original Super + Augmented Fine):")
    print("  - Super-class: Original KNN (77.83% validation accuracy)")
    print("  - Fiber: Augmented KNN (87.50% validation accuracy)")
    print("  - Rigid: Augmented KNN (94.00% validation accuracy)")
    print()
    
    # Confidence comparison
    svm_confidences = [r['svm']['final_confidence'] for r in results]
    knn_confidences = [r['knn']['final_confidence'] for r in results]
    
    print("Test Set Performance:")
    print(f"  SVM Average Confidence: {np.mean(svm_confidences):.1%}")
    print(f"  KNN Average Confidence: {np.mean(knn_confidences):.1%}")
    print(f"  Agreement Rate: {sum(1 for r in results if r['agreement'])/len(results)*100:.1f}%")
    
    print("\nRecommendation:")
    if np.mean(svm_confidences) > np.mean(knn_confidences):
        print("  ✅ SVM Strategy shows higher confidence on this test set")
    elif np.mean(knn_confidences) > np.mean(svm_confidences):
        print("  ✅ KNN Strategy shows higher confidence on this test set")
    else:
        print("  ⚖️  Both strategies show similar confidence")
    
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test models on custom test data')
    parser.add_argument('--test-folder', type=str, 
                       default=r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\test_with",
                       help='Path to test folder with images')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed comparison')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.test_folder):
        print(f"[ERROR] Test folder not found: {args.test_folder}")
        sys.exit(1)
    
    # Run tests
    results = test_on_folder(args.test_folder)
    
    # Show detailed comparison if requested
    if args.detailed and results:
        compare_strategies(results)


if __name__ == "__main__":
    main()
