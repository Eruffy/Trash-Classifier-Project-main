"""
Data augmentation module to increase dataset size by 30%+.
Targets 700 samples per class with quality augmentation techniques.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import shutil

from config.config import (
    DATASET_PATH, AUGMENTED_PATH, TARGET_SAMPLES_PER_CLASS,
    DATASET_FOLDERS, BOOST_AUGMENTATION_FOR, IMAGE_SIZE
)


class DataAugmentor:
    """Handles data augmentation for the trash classification dataset."""
    
    def __init__(self, source_path=DATASET_PATH, target_path=AUGMENTED_PATH, 
                 target_samples=TARGET_SAMPLES_PER_CLASS):
        self.source_path = source_path
        self.target_path = target_path
        self.target_samples = target_samples
        
        # Define augmentation pipeline using Albumentation
        self.augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ])
        
        # More aggressive augmentation for problematic classes
        self.boost_augment = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=45, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.8),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=10, p=0.7),
            A.GaussNoise(var_limit=(10.0, 70.0), p=0.4),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.7),
            A.Perspective(scale=(0.05, 0.15), p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        ])
    
    def is_corrupt(self, img_path):
        """Check if image is corrupt."""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return True
            return False
        except Exception:
            return True
    
    def augment_image(self, image, use_boost=False):
        """Apply augmentation to a single image."""
        if use_boost:
            augmented = self.boost_augment(image=image)
        else:
            augmented = self.augment(image=image)
        return augmented['image']
    
    def augment_class(self, class_name, class_id, use_boost=False):
        """Augment all images for a specific class."""
        source_folder = Path(self.source_path) / class_name
        target_folder = Path(self.target_path) / class_name
        target_folder.mkdir(parents=True, exist_ok=True)
        
        # Get all valid images
        valid_images = []
        for img_path in source_folder.glob("*.jpg"):
            if not self.is_corrupt(img_path):
                valid_images.append(img_path)
        
        if not valid_images:
            print(f"[WARNING] No valid images found for class {class_name}")
            return 0
        
        current_count = len(valid_images)
        augmentations_needed = max(0, self.target_samples - current_count)
        
        print(f"\n📦 Class: {class_name} (ID: {class_id})")
        print(f"   Original: {current_count} images")
        print(f"   Target: {self.target_samples} images")
        print(f"   Augmentations needed: {augmentations_needed}")
        print(f"   Boost mode: {use_boost}")
        
        # Copy original images first
        for img_path in tqdm(valid_images, desc=f"  Copying originals"):
            shutil.copy(str(img_path), str(target_folder / img_path.name))
        
        # Generate augmented images
        if augmentations_needed > 0:
            aug_per_image = (augmentations_needed // current_count) + 1
            
            for img_path in tqdm(valid_images, desc=f"  Augmenting"):
                # Read image
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate augmentations
                for i in range(aug_per_image):
                    if current_count >= self.target_samples:
                        break
                    
                    # Apply augmentation
                    aug_image = self.augment_image(image, use_boost=use_boost)
                    
                    # Save augmented image
                    aug_filename = f"{img_path.stem}_aug{i}{img_path.suffix}"
                    aug_path = target_folder / aug_filename
                    
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(aug_path), aug_image_bgr)
                    
                    current_count += 1
                    
                    if current_count >= self.target_samples:
                        break
        
        final_count = len(list(target_folder.glob("*.jpg")))
        print(f"   ✓ Final count: {final_count} images")
        
        return final_count
    
    def augment_all_classes(self):
        """Augment all classes in the dataset."""
        print("=" * 70)
        print("DATA AUGMENTATION PIPELINE")
        print("=" * 70)
        print(f"Source: {self.source_path}")
        print(f"Target: {self.target_path}")
        print(f"Target samples per class: {self.target_samples}")
        print(f"Boost augmentation for: {BOOST_AUGMENTATION_FOR}")
        print("=" * 70)
        
        # Create target directory
        Path(self.target_path).mkdir(parents=True, exist_ok=True)
        
        total_augmented = 0
        class_stats = {}
        
        # Augment each class
        for class_id, class_name in DATASET_FOLDERS.items():
            use_boost = class_name in BOOST_AUGMENTATION_FOR
            count = self.augment_class(class_name, class_id, use_boost=use_boost)
            class_stats[class_name] = count
            total_augmented += count
        
        # Print summary
        print("\n" + "=" * 70)
        print("AUGMENTATION SUMMARY")
        print("=" * 70)
        for class_name, count in class_stats.items():
            percentage = (count / self.target_samples) * 100 if self.target_samples > 0 else 0
            print(f"{class_name:12}: {count:4} images ({percentage:5.1f}% of target)")
        print(f"\nTotal augmented images: {total_augmented}")
        print(f"Average per class: {total_augmented / len(class_stats):.1f}")
        print("=" * 70)
        
        return class_stats


def main():
    """Run data augmentation."""
    augmentor = DataAugmentor()
    augmentor.augment_all_classes()


if __name__ == "__main__":
    main()
