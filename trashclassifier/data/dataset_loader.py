"""
Dataset loader and splitter for training and validation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config.config import (
    AUGMENTED_PATH, DATASET_FOLDERS, IMAGE_SIZE,
    VALIDATION_SPLIT, RANDOM_STATE
)
from data.label_mapping import get_fine_class_id_from_folder


class DatasetLoader:
    """Loads and splits the augmented dataset."""
    
    def __init__(self, dataset_path=AUGMENTED_PATH, val_split=VALIDATION_SPLIT):
        self.dataset_path = dataset_path
        self.val_split = val_split
        self.random_state = RANDOM_STATE
    
    def load_images_from_folder(self, folder_path, class_id):
        """Load all valid images from a folder."""
        images = []
        labels = []
        skipped = 0
        
        for img_path in Path(folder_path).glob("*.jpg"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue
                
                # Resize to standard size
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(class_id)
                
            except Exception as e:
                skipped += 1
                continue
        
        if skipped > 0:
            print(f"   [WARNING] Skipped {skipped} corrupt images")
        
        return images, labels
    
    def load_dataset(self, verbose=True):
        """Load the complete dataset."""
        if verbose:
            print("=" * 70)
            print("LOADING DATASET")
            print("=" * 70)
            print(f"Source: {self.dataset_path}")
        
        all_images = []
        all_labels = []
        class_counts = {}
        
        for class_id, folder_name in DATASET_FOLDERS.items():
            folder_path = Path(self.dataset_path) / folder_name
            
            if not folder_path.exists():
                if verbose:
                    print(f"[WARNING] Folder not found: {folder_name}")
                continue
            
            if verbose:
                print(f"\nLoading: {folder_name} (ID: {class_id})...")
            
            images, labels = self.load_images_from_folder(folder_path, class_id)
            
            all_images.extend(images)
            all_labels.extend(labels)
            class_counts[folder_name] = len(images)
            
            if verbose:
                print(f"   Loaded: {len(images)} images")
        
        if verbose:
            print("\n" + "=" * 70)
            print("DATASET SUMMARY")
            print("=" * 70)
            for class_name, count in class_counts.items():
                print(f"{class_name:12}: {count:4} images")
            print(f"\nTotal images: {len(all_images)}")
            print("=" * 70)
        
        return np.array(all_images), np.array(all_labels), class_counts
    
    def split_dataset(self, images, labels, verbose=True):
        """Split dataset into training and validation sets."""
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels,
            test_size=self.val_split,
            random_state=self.random_state,
            stratify=labels  # Ensure balanced split
        )
        
        if verbose:
            print("\n" + "=" * 70)
            print("TRAIN/VALIDATION SPLIT")
            print("=" * 70)
            print(f"Training set: {len(X_train)} images ({(1-self.val_split)*100:.0f}%)")
            print(f"Validation set: {len(X_val)} images ({self.val_split*100:.0f}%)")
            
            # Show class distribution
            unique, train_counts = np.unique(y_train, return_counts=True)
            _, val_counts = np.unique(y_val, return_counts=True)
            
            print("\nClass distribution:")
            print(f"{'Class':12} {'Train':>8} {'Val':>8}")
            for cls_id, folder_name in DATASET_FOLDERS.items():
                t_idx = np.where(unique == cls_id)[0]
                train_count = train_counts[t_idx][0] if len(t_idx) > 0 else 0
                val_count = val_counts[t_idx][0] if len(t_idx) > 0 else 0
                print(f"{folder_name:12} {train_count:8} {val_count:8}")
            print("=" * 70)
        
        return X_train, X_val, y_train, y_val


def main():
    """Test dataset loading."""
    loader = DatasetLoader()
    images, labels, counts = loader.load_dataset()
    X_train, X_val, y_train, y_val = loader.split_dataset(images, labels)


if __name__ == "__main__":
    main()
