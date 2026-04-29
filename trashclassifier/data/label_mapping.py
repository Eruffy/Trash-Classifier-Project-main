"""
Label mapping utilities for converting between class IDs and names.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    FINE_CLASS_NAMES, FINE_CLASS_IDS,
    SUPER_CLASS_NAMES, SUPER_CLASS_IDS,
    FINE_TO_SUPER, SUPER_TO_FINE,
    DATASET_FOLDERS
)


def fine_class_id_to_name(class_id):
    """Convert fine-class ID to name."""
    return FINE_CLASS_NAMES.get(class_id, "Unknown")


def fine_class_name_to_id(class_name):
    """Convert fine-class name to ID."""
    return FINE_CLASS_IDS.get(class_name, 6)  # Default to Unknown


def super_class_id_to_name(class_id):
    """Convert super-class ID to name."""
    return SUPER_CLASS_NAMES.get(class_id, "Unknown")


def super_class_name_to_id(class_name):
    """Convert super-class name to ID."""
    return SUPER_CLASS_IDS.get(class_name, -1)


def fine_to_super_class(fine_class_id):
    """Map fine-class ID to super-class ID."""
    return FINE_TO_SUPER.get(fine_class_id, -1)


def super_to_fine_classes(super_class_id):
    """Get list of fine-class IDs for a super-class."""
    return SUPER_TO_FINE.get(super_class_id, [])


def get_dataset_folder_name(fine_class_id):
    """Get dataset folder name for a fine-class ID."""
    return DATASET_FOLDERS.get(fine_class_id, "unknown")


def get_fine_class_id_from_folder(folder_name):
    """Get fine-class ID from dataset folder name."""
    folder_to_id = {v: k for k, v in DATASET_FOLDERS.items()}
    return folder_to_id.get(folder_name.lower(), 6)


def print_class_hierarchy():
    """Print the complete class hierarchy for reference."""
    print("=" * 60)
    print("CLASS HIERARCHY")
    print("=" * 60)
    
    print("\n📊 FINE CLASSES (7 total):")
    for fine_id, fine_name in FINE_CLASS_NAMES.items():
        super_id = FINE_TO_SUPER.get(fine_id, -1)
        super_name = SUPER_CLASS_NAMES.get(super_id, "None")
        print(f"  ID {fine_id}: {fine_name:12} → Super-class: {super_name}")
    
    print("\n🎯 SUPER CLASSES (4 total):")
    for super_id, super_name in SUPER_CLASS_NAMES.items():
        fine_classes = SUPER_TO_FINE.get(super_id, [])
        fine_names = [FINE_CLASS_NAMES[fc] for fc in fine_classes]
        print(f"  ID {super_id}: {super_name:12} → Contains: {', '.join(fine_names)}")
    
    print("\n📁 DATASET FOLDERS:")
    for fine_id, folder_name in DATASET_FOLDERS.items():
        print(f"  {folder_name:12} → Fine-class ID: {fine_id} ({FINE_CLASS_NAMES[fine_id]})")
    
    print("=" * 60)


if __name__ == "__main__":
    print_class_hierarchy()
