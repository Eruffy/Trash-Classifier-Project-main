"""
Data Augmentation Module
Increases dataset size by 30%+ using OpenCV and scikit-image
Target: 700 samples per class with quality augmentations
"""
import os
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config import Config


def rotate_image(img, angle):
    """Rotate image by given angle"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def flip_horizontal(img):
    """Flip image horizontally"""
    return cv2.flip(img, 1)


def adjust_brightness(img, factor):
    """Adjust brightness by factor"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(img, factor):
    """Adjust contrast by factor"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = lab[:, :, 0].mean()
    lab[:, :, 0] = np.clip((lab[:, :, 0] - mean) * factor + mean, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def add_gaussian_noise(img, sigma=10):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def zoom_image(img, factor):
    """Zoom image by factor"""
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    
    if factor > 1:
        # Zoom in - crop center
        resized = cv2.resize(img, (new_w, new_h))
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        return resized[start_y:start_y+h, start_x:start_x+w]
    else:
        # Zoom out - add border
        resized = cv2.resize(img, (new_w, new_h))
        border_y = (h - new_h) // 2
        border_x = (w - new_w) // 2
        return cv2.copyMakeBorder(resized, border_y, border_y, border_x, border_x, 
                                 cv2.BORDER_REFLECT)


def augment_image(img, aug_type):
    """Apply specific augmentation type"""
    if aug_type == 'rotate_10':
        return rotate_image(img, 10)
    elif aug_type == 'rotate_-10':
        return rotate_image(img, -10)
    elif aug_type == 'rotate_20':
        return rotate_image(img, 20)
    elif aug_type == 'rotate_-20':
        return rotate_image(img, -20)
    elif aug_type == 'flip':
        return flip_horizontal(img)
    elif aug_type == 'bright_1.2':
        return adjust_brightness(img, 1.2)
    elif aug_type == 'bright_0.8':
        return adjust_brightness(img, 0.8)
    elif aug_type == 'contrast_1.2':
        return adjust_contrast(img, 1.2)
    elif aug_type == 'contrast_0.8':
        return adjust_contrast(img, 0.8)
    elif aug_type == 'noise':
        return add_gaussian_noise(img, sigma=10)
    elif aug_type == 'zoom_1.1':
        return zoom_image(img, 1.1)
    elif aug_type == 'zoom_0.9':
        return zoom_image(img, 0.9)
    else:
        return img


def augment_dataset():
    """Augment entire dataset to target samples per class"""
    
    print(f"[INFO] Starting data augmentation...")
    print(f"[INFO] Target: {Config.TARGET_SAMPLES_PER_CLASS} samples per class")
    print(f"[INFO] Dataset: {Config.DATASET_PATH}")
    
    # Augmentation techniques
    augmentations = [
        'rotate_10', 'rotate_-10', 'rotate_20', 'rotate_-20',
        'flip', 'bright_1.2', 'bright_0.8', 'contrast_1.2', 
        'contrast_0.8', 'noise', 'zoom_1.1', 'zoom_0.9'
    ]
    
    stats = {}
    
    for class_name in Config.CLASS_NAMES.keys():
        class_path = os.path.join(Config.DATASET_PATH, class_name)
        
        if not os.path.exists(class_path):
            print(f"[WARNING] Class folder not found: {class_path}")
            continue
        
        # Get existing images
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        original_count = len(images)
        current_count = original_count
        
        print(f"\n[INFO] Processing {class_name}: {original_count} original images")
        
        if current_count >= Config.TARGET_SAMPLES_PER_CLASS:
            print(f"  ✅ Already at target ({current_count}/{Config.TARGET_SAMPLES_PER_CLASS})")
            stats[class_name] = {'original': original_count, 'augmented': 0, 'total': current_count}
            continue
        
        # Calculate how many augmentations needed
        needed = Config.TARGET_SAMPLES_PER_CLASS - current_count
        augs_per_image = needed // original_count + 1
        
        print(f"  Need {needed} more images ({augs_per_image} augmentations per image)")
        
        augmented_count = 0
        aug_cycle = 0
        
        while current_count < Config.TARGET_SAMPLES_PER_CLASS:
            for img_name in images:
                if current_count >= Config.TARGET_SAMPLES_PER_CLASS:
                    break
                
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Apply augmentation
                aug_type = augmentations[aug_cycle % len(augmentations)]
                aug_img = augment_image(img, aug_type)
                
                # Save augmented image
                base_name = os.path.splitext(img_name)[0]
                aug_name = f"{base_name}_aug_{aug_type}_{aug_cycle}.jpg"
                aug_path = os.path.join(class_path, aug_name)
                
                cv2.imwrite(aug_path, aug_img)
                augmented_count += 1
                current_count += 1
                
                if augmented_count % 50 == 0:
                    print(f"    Progress: {current_count}/{Config.TARGET_SAMPLES_PER_CLASS}")
            
            aug_cycle += 1
        
        print(f"  ✅ Complete: {original_count} → {current_count} (+{augmented_count} augmented)")
        stats[class_name] = {
            'original': original_count,
            'augmented': augmented_count,
            'total': current_count
        }
    
    # Print summary
    print("\n" + "="*70)
    print("AUGMENTATION SUMMARY".center(70))
    print("="*70)
    
    total_original = sum(s['original'] for s in stats.values())
    total_augmented = sum(s['augmented'] for s in stats.values())
    total_final = sum(s['total'] for s in stats.values())
    
    for class_name, s in stats.items():
        increase = ((s['total'] - s['original']) / s['original'] * 100) if s['original'] > 0 else 0
        print(f"{class_name:12} : {s['original']:4} → {s['total']:4} (+{increase:5.1f}%)")
    
    print("-"*70)
    overall_increase = ((total_final - total_original) / total_original * 100) if total_original > 0 else 0
    print(f"{'TOTAL':12} : {total_original:4} → {total_final:4} (+{overall_increase:5.1f}%)")
    print("="*70 + "\n")
    
    if overall_increase >= 30:
        print(f"✅ SUCCESS: Dataset increased by {overall_increase:.1f}% (target: 30%+)")
    else:
        print(f"⚠️  WARNING: Dataset only increased by {overall_increase:.1f}% (target: 30%+)")


if __name__ == "__main__":
    augment_dataset()
