"""
Master Training Script
Orchestrates the entire training pipeline with hyperparameter tuning.
"""
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.augmentation import augment_dataset
from training.train_super import train_super_models
from training.train_fine import train_fine_models
from training.hyperparameter_tuner import tune_super_svm, tune_super_knn, tune_fine_svm, tune_fine_knn
from config import Config
import argparse


def print_header(text):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train the complete waste classification system')
    parser.add_argument('--skip-augmentation', action='store_true', 
                       help='Skip data augmentation (use existing augmented data)')
    parser.add_argument('--skip-tuning', action='store_true',
                       help='Skip hyperparameter tuning (use default parameters)')
    parser.add_argument('--tune-only', action='store_true',
                       help='Only run hyperparameter tuning, do not train final models')
    args = parser.parse_args()
    
    start_time = time.time()
    
    print_header("WASTE CLASSIFICATION SYSTEM - TRAINING PIPELINE")
    print(f"Dataset: {Config.DATASET_PATH}")
    print(f"Output: {Config.SAVED_MODELS_PATH}")
    print(f"Target samples per class: {Config.TARGET_SAMPLES_PER_CLASS}")
    
    # Step 1: Data Augmentation
    if not args.skip_augmentation:
        print_header("STEP 1: DATA AUGMENTATION")
        print(f"Augmenting dataset to {Config.TARGET_SAMPLES_PER_CLASS} samples per class...")
        try:
            augment_dataset()
            print("✅ Data augmentation completed successfully!")
        except Exception as e:
            print(f"❌ Data augmentation failed: {e}")
            return
    else:
        print_header("STEP 1: DATA AUGMENTATION (SKIPPED)")
    
    # Step 2: Hyperparameter Tuning (Optional)
    best_params = {
        'super_svm': None,
        'super_knn': None,
        'fiber_svm': None,
        'fiber_knn': None,
        'rigid_svm': None,
        'rigid_knn': None
    }
    
    if not args.skip_tuning:
        print_header("STEP 2: HYPERPARAMETER TUNING")
        
        print("\n🔍 Tuning Super-Class SVM...")
        try:
            best_params['super_svm'] = tune_super_svm()
            print(f"✅ Best SVM params: {best_params['super_svm']}")
        except Exception as e:
            print(f"⚠️  Super SVM tuning failed: {e}")
        
        print("\n🔍 Tuning Super-Class k-NN...")
        try:
            best_params['super_knn'] = tune_super_knn()
            print(f"✅ Best k-NN params: {best_params['super_knn']}")
        except Exception as e:
            print(f"⚠️  Super k-NN tuning failed: {e}")
        
        print("\n🔍 Tuning Fiber Fine-Class SVM...")
        try:
            best_params['fiber_svm'] = tune_fine_svm('fiber')
            print(f"✅ Best Fiber SVM params: {best_params['fiber_svm']}")
        except Exception as e:
            print(f"⚠️  Fiber SVM tuning failed: {e}")
        
        print("\n🔍 Tuning Fiber Fine-Class k-NN...")
        try:
            best_params['fiber_knn'] = tune_fine_knn('fiber')
            print(f"✅ Best Fiber k-NN params: {best_params['fiber_knn']}")
        except Exception as e:
            print(f"⚠️  Fiber k-NN tuning failed: {e}")
        
        print("\n🔍 Tuning Rigid Fine-Class SVM...")
        try:
            best_params['rigid_svm'] = tune_fine_svm('rigid')
            print(f"✅ Best Rigid SVM params: {best_params['rigid_svm']}")
        except Exception as e:
            print(f"⚠️  Rigid SVM tuning failed: {e}")
        
        print("\n🔍 Tuning Rigid Fine-Class k-NN...")
        try:
            best_params['rigid_knn'] = tune_fine_knn('rigid')
            print(f"✅ Best Rigid k-NN params: {best_params['rigid_knn']}")
        except Exception as e:
            print(f"⚠️  Rigid k-NN tuning failed: {e}")
        
        print("\n✅ Hyperparameter tuning completed!")
        
    else:
        print_header("STEP 2: HYPERPARAMETER TUNING (SKIPPED)")
    
    if args.tune_only:
        print_header("TUNING COMPLETE - EXITING (--tune-only flag)")
        return
    
    # Step 3: Train Super-Class Models
    print_header("STEP 3: TRAINING SUPER-CLASS CLASSIFIERS")
    try:
        train_super_models(
            svm_params=best_params['super_svm'],
            knn_params=best_params['super_knn']
        )
        print("✅ Super-class models trained successfully!")
    except Exception as e:
        print(f"❌ Super-class training failed: {e}")
        return
    
    # Step 4: Train Fine-Class Models
    print_header("STEP 4: TRAINING FINE-CLASS CLASSIFIERS")
    try:
        train_fine_models(
            fiber_svm_params=best_params['fiber_svm'],
            fiber_knn_params=best_params['fiber_knn'],
            rigid_svm_params=best_params['rigid_svm'],
            rigid_knn_params=best_params['rigid_knn']
        )
        print("✅ Fine-class models trained successfully!")
    except Exception as e:
        print(f"❌ Fine-class training failed: {e}")
        return
    
    # Summary
    elapsed_time = time.time() - start_time
    print_header("TRAINING COMPLETE!")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"\n✅ All models saved to: {Config.SAVED_MODELS_PATH}")
    print("\nYou can now run:")
    print("  - Real-time camera: python app/live_camera.py")
    print("  - Single image: python scripts/predict_image.py <image_path>")
    print("  - Evaluation: python scripts/evaluate.py")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
