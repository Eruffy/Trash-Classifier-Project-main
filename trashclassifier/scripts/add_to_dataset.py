"""
Add test images to training dataset for improved generalization
"""
import shutil
import os


def add_to_dataset():
    """Copy test images to their correct dataset folders"""
    
    test_dir = r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\test_with"
    dataset_dir = r"C:\Users\ahmed\Desktop\MACHINE_LEARNING\dataset"
    
    # Mapping: test image -> (true class, copy to)
    mappings = [
        ("cardboard.jpeg", "cardboard"),
        ("glass.jpeg", "glass"),
        ("glass (2).jpeg", "glass"),
        ("plastic.jpeg", "plastic"),
        ("plastic (2).jpeg", "plastic"),
        ("plastic (3).jpeg", "plastic"),
    ]
    
    print("="*80)
    print("ADDING TEST IMAGES TO DATASET".center(80))
    print("="*80)
    print()
    
    added = 0
    for img_name, class_folder in mappings:
        src = os.path.join(test_dir, img_name)
        dst_folder = os.path.join(dataset_dir, class_folder)
        
        # Create unique name (avoid overwriting)
        base_name = f"realworld_{img_name.replace(' ', '_')}"
        dst = os.path.join(dst_folder, base_name)
        
        if not os.path.exists(src):
            print(f"⚠️  Source not found: {img_name}")
            continue
        
        if not os.path.exists(dst_folder):
            print(f"⚠️  Destination folder not found: {class_folder}")
            continue
        
        try:
            shutil.copy2(src, dst)
            print(f"✅ Added: {img_name:30} → {class_folder}/")
            added += 1
        except Exception as e:
            print(f"❌ Failed: {img_name} - {e}")
    
    print()
    print("="*80)
    print(f"✅ Successfully added {added}/{len(mappings)} images to dataset")
    print("="*80)
    print()
    print("Next step: python train_all.py --skip-augmentation")
    print("           (Uses existing optimized hyperparameters)")
    print()


if __name__ == "__main__":
    add_to_dataset()
