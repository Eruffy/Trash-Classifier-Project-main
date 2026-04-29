# Trash Classifier Project

## Repository
🔗 **GitHub:** [https://github.com/Eruffy/Trash-Classifier-Project-main.git](https://github.com/Eruffy/Trash-Classifier-Project-main.git)

## Quick Start

### Train All Models
```bash
python train_all.py
```

### Test Single Image
```bash
python scripts/predict_image.py path/to/image.jpg
```

### Batch Classify Folder
```bash
python scripts/classify_all.py path/to/folder
```

### Evaluate Models
```bash
python scripts/evaluate.py
```

---

## Project Structure

```
trashclassifier/
├── data/                    # Dataset loading & augmentation
├── features/                # Feature extraction (6,203 features)
├── inference/               # Prediction logic
├── models/                  # Training scripts
├── saved_models/            # 24 trained models (SVM + KNN)
├── scripts/                 # Utility scripts
├── training/                # Training modules
├── config/                  # Configuration
└── train_all.py            # Master training script
```

---

## Models

### Two Main Strategies

**1. SVM Augmented** - All trained on augmented data (500 images/class)
- Best fine-class accuracy (96%)
- Models: `super_svm_aug.pkl`, `fiber_svm_aug.pkl`, `rigid_svm_aug.pkl`

**2. KNN Hybrid** - Original super + Augmented fine
- Balanced performance
- Models: `super_knn.pkl`, `fiber_knn_aug.pkl`, `rigid_knn_aug.pkl`

### Class Hierarchy

**Super-classes:**
- Fiber → Paper, Cardboard
- Rigid → Plastic, Metal
- Transparent → Glass
- Garbage → Trash

**Fine-classes:**
- 0: Glass
- 1: Paper
- 2: Cardboard
- 3: Plastic
- 4: Metal
- 5: Trash
- 6: Unknown (low confidence)

---

## Key Scripts

### Training
- `train_all.py` - Train all models from scratch
- `models/train_super_svm.py` - Train super-class SVM
- `models/train_fine_knns.py` - Train fine-class KNN models

### Evaluation & Testing
- `scripts/evaluate.py` - Evaluate model accuracy
- `scripts/predict_image.py` - Test single image
- `scripts/classify_all.py` - Batch classify folder

### Dataset Management
- `scripts/add_to_dataset.py` - Add new images to training set

---

## Configuration

Edit `config/config.py` for:
- Dataset paths
- Model hyperparameters
- Augmentation settings
- Class mappings

---

## Model Performance

### Validation Accuracy (Augmented Dataset)

**SVM:**
- Super-class: 75%
- Fiber: 96% ⭐
- Rigid: 89%

**KNN Hybrid:**
- Super-class: 77% (original)
- Fiber: 94% (augmented)
- Rigid: 89% (augmented)

---

## File Organization

### Keep These Files

**Core Modules:**
- `data/`, `features/`, `inference/`, `training/`
- All needed for system to work

**Saved Models:** (24 files)
- 12 model files: `*_svm_aug.pkl`, `*_knn.pkl`, `*_knn_aug.pkl`
- 12 scaler files: `*_scaler*.pkl`

**Production Scripts:**
- `predict_image.py` - Test single images
- `evaluate.py` - Model evaluation
- `classify_all.py` - Batch processing
- `add_to_dataset.py` - Dataset management

---

## Tips

### When Models Disagree
- Both agree → Trust prediction
- Disagree → Edge case, review manually

### Improving Accuracy
1. Add more real-world training images
2. Retrain with `train_all.py`
3. Collect edge cases (dirty, worn items)
4. Consider deep learning (CNN) for complex cases

### Dataset Paths
- Original: `C:\Users\ahmed\Desktop\MACHINE_LEARNING\dataset`
- Augmented: `C:\Users\ahmed\Desktop\FINAL_ML\augmented_data`

---

## Common Commands

```bash
# Train all models
python train_all.py

# Test image
python scripts/predict_image.py test.jpg

# Evaluate accuracy
python scripts/evaluate.py

# Classify folder
python scripts/classify_all.py test_folder/

# Add images to dataset
python scripts/add_to_dataset.py new_images/ cardboard
```

---

**Project Status:** Production Ready ✅
**Models:** SVM Augmented + KNN Hybrid (24 files loaded)
**Feature Extraction:** 6,203 dimensions (HOG, LBP, Color, Edge)
