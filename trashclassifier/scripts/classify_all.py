"""
Classify All Images in Dataset
Processes every image in the dataset and generates detailed classification report.
Similar to MSI_Project's comprehensive analysis.
"""
import sys
import os
import cv2
import numpy as np
from collections import defaultdict
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.ensemble_predictor import EnsemblePredictor
from config import Config


def classify_all_images():
    """Classify every image in the dataset and generate comprehensive report"""
    
    print("="*70)
    print("COMPREHENSIVE IMAGE CLASSIFICATION".center(70))
    print("="*70)
    print(f"\nDataset: {Config.DATASET_PATH}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize predictor
    print("[INFO] Loading models...")
    predictor = EnsemblePredictor()
    print("[INFO] Models loaded successfully!\n")
    
    # Reverse mapping: ID -> class name
    id_to_class = {v: k for k, v in Config.CLASS_NAMES.items()}
    
    # Storage for results
    results = {
        'total_images': 0,
        'correct': 0,
        'incorrect': 0,
        'unknown': 0,
        'per_class': {},
        'misclassifications': []
    }
    
    # Process each class
    for true_class_name in Config.CLASS_NAMES.keys():
        true_class_id = Config.CLASS_NAMES[true_class_name]
        class_path = os.path.join(Config.DATASET_PATH, true_class_name)
        
        if not os.path.exists(class_path):
            print(f"[WARNING] Class folder not found: {class_path}")
            continue
        
        print(f"[INFO] Processing {true_class_name.upper()}...")
        
        # Get all images
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Initialize class stats
        class_stats = {
            'total': len(images),
            'correct': 0,
            'incorrect': 0,
            'unknown': 0,
            'confused_with': defaultdict(int),
            'misclassified_files': []
        }
        
        # Process each image
        for i, img_name in enumerate(images, 1):
            img_path = os.path.join(class_path, img_name)
            
            # Progress indicator
            if i % 100 == 0 or i == len(images):
                print(f"  Progress: {i}/{len(images)} images processed...")
            
            # Load and predict
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARNING] Could not load: {img_name}")
                continue
            
            # Get predictions
            predictions = predictor.predict_all(img)
            
            # Use ensemble prediction
            pred_class_id = predictions['ensemble']['class_id']
            pred_confidence = predictions['ensemble']['confidence']
            svm_class = predictions['svm']['class']
            knn_class = predictions['knn']['class']
            
            results['total_images'] += 1
            
            # Check if correct
            if pred_class_id == true_class_id:
                results['correct'] += 1
                class_stats['correct'] += 1
            elif pred_class_id == 6:  # Unknown
                results['unknown'] += 1
                class_stats['unknown'] += 1
                class_stats['misclassified_files'].append({
                    'file': img_name,
                    'predicted': 'UNKNOWN',
                    'confidence': pred_confidence,
                    'svm': svm_class,
                    'knn': knn_class
                })
            else:
                results['incorrect'] += 1
                class_stats['incorrect'] += 1
                pred_class_name = id_to_class.get(pred_class_id, 'UNKNOWN')
                class_stats['confused_with'][pred_class_name] += 1
                
                # Store misclassification details
                misclass_info = {
                    'file': img_name,
                    'true_class': true_class_name,
                    'predicted': pred_class_name,
                    'confidence': pred_confidence,
                    'svm_pred': svm_class,
                    'knn_pred': knn_class,
                    'models_agree': svm_class == knn_class
                }
                class_stats['misclassified_files'].append(misclass_info)
                results['misclassifications'].append(misclass_info)
        
        # Calculate class accuracy
        class_stats['accuracy'] = (class_stats['correct'] / class_stats['total'] * 100) if class_stats['total'] > 0 else 0
        results['per_class'][true_class_name] = class_stats
        
        print(f"  ✅ {true_class_name}: {class_stats['correct']}/{class_stats['total']} correct ({class_stats['accuracy']:.2f}%)\n")
    
    # Calculate overall accuracy
    results['overall_accuracy'] = (results['correct'] / results['total_images'] * 100) if results['total_images'] > 0 else 0
    
    # Print comprehensive report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT".center(70))
    print("="*70)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total Images: {results['total_images']}")
    print(f"  Correct: {results['correct']} ({results['correct']/results['total_images']*100:.2f}%)")
    print(f"  Incorrect: {results['incorrect']} ({results['incorrect']/results['total_images']*100:.2f}%)")
    print(f"  Unknown: {results['unknown']} ({results['unknown']/results['total_images']*100:.2f}%)")
    print(f"  Overall Accuracy: {results['overall_accuracy']:.2f}%")
    
    print("\n" + "-"*70)
    print("📈 PER-CLASS PERFORMANCE:")
    print("-"*70)
    
    for class_name, stats in results['per_class'].items():
        print(f"\n{class_name.upper()}:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']} ({stats['accuracy']:.2f}%)")
        print(f"  Incorrect: {stats['incorrect']}")
        print(f"  Unknown: {stats['unknown']}")
        
        if stats['confused_with']:
            print(f"  Confused with:")
            for confused_class, count in sorted(stats['confused_with'].items(), key=lambda x: x[1], reverse=True):
                print(f"    → {confused_class}: {count} times")
    
    # Print misclassification details
    if results['misclassifications']:
        print("\n" + "-"*70)
        print(f"❌ MISCLASSIFIED IMAGES ({len(results['misclassifications'])} total):")
        print("-"*70)
        
        # Group by true class
        misclass_by_class = defaultdict(list)
        for m in results['misclassifications']:
            misclass_by_class[m['true_class']].append(m)
        
        for true_class, misclasses in sorted(misclass_by_class.items()):
            print(f"\n{true_class.upper()} misclassified as:")
            for m in misclasses:
                agree_marker = "✓" if m['models_agree'] else "✗"
                print(f"  {agree_marker} {m['file']}")
                print(f"     Predicted: {m['predicted']} ({m['confidence']:.1f}%)")
                print(f"     SVM: {m['svm_pred']} | k-NN: {m['knn_pred']}")
    
    # Save detailed report to JSON
    output_file = os.path.join(Config.SAVED_MODELS_PATH, 'classification_report.json')
    
    # Prepare JSON-serializable version
    json_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_path': Config.DATASET_PATH,
        'total_images': results['total_images'],
        'correct': results['correct'],
        'incorrect': results['incorrect'],
        'unknown': results['unknown'],
        'overall_accuracy': results['overall_accuracy'],
        'per_class': {}
    }
    
    for class_name, stats in results['per_class'].items():
        json_results['per_class'][class_name] = {
            'total': stats['total'],
            'correct': stats['correct'],
            'incorrect': stats['incorrect'],
            'unknown': stats['unknown'],
            'accuracy': stats['accuracy'],
            'confused_with': dict(stats['confused_with']),
            'misclassified_files': stats['misclassified_files']
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print("\n" + "="*70)
    print(f"✅ Detailed report saved to: {output_file}")
    print("="*70)
    
    return results


def main():
    try:
        results = classify_all_images()
        
        # Print summary
        print(f"\n🎯 FINAL ACCURACY: {results['overall_accuracy']:.2f}%")
        print(f"✅ {results['correct']} correct out of {results['total_images']} total images")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
