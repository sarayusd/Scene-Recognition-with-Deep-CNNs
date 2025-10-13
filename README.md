# Indoor Scene Classification — MIT67

This project focuses on indoor scene classification using the MIT Indoor67 dataset.  
Indoor scene recognition is challenging due to high intra-class variability, clutter and reliance on object-level cues rather than just global layouts.  
We implement and compare multiple deep learning architectures and combine them using ensemble learning to improve accuracy and robustness.

---

## Dataset Description

- **Dataset:** [MIT Indoor67](http://web.mit.edu/torralba/www/indoor.html)  
- **Classes:** 67 indoor categories (e.g., bookstore, kitchen, auditorium)  
- **Images:** 15,620 total  
- **Split:** standard train/val/test  
- **Input:** 224 × 224 RGB images (resized and normalized with ImageNet mean/std)  
- **Output:** probability distribution across 67 classes  
- **Metrics:** Top-1 accuracy, macro/weighted F1, confusion matrix, classification report

### Preprocessing
- Train: RandomResizedCrop, horizontal flip, rotation (±15°), color jitter, normalization  
- Val/Test: resize and normalization only

### Class Imbalance Handling
- Used class weights and label smoothing to counter class imbalance.  
- Applied weighted sampling and targeted finetuning for lowperforming classes.

---
## Model Architectures

### 1. DenseNet-Like Scene Classifier (Teacher)
- 58 convolutional layers, dense connections, 4 dense blocks with transition layers
- **Total params:** ~9.9M  
- **Train Accuracy:** 79.6%  
- **Validation Accuracy:** 60.3%  
- **Test Accuracy:** 62.1%  
- **Macro F1:** 0.60 | **Weighted F1:** 0.62  

**Top performing classes:** cloister, casino, poolinside, florist, grocerystore.  
**Challenging classes:** artstudio, office, hospitalroom.

---

### 2. HybridEfficientCNN (Student + SE Blocks)
- 17 layers with SE attention after block 3
- Total params: ~2.45M
- Lightweight, stable training up to 60 epochs
- **Test Accuracy:** 51.9% | Macro F1: 0.44 | Weighted F1: 0.50
- Good performance on cluttered scenes; struggled on rare classes.

---

### 3. MiniEfficientNet (Student)
- MBConv blocks + SE modules + global pooling
- Total params: ~1.2M
- **With Knowledge Distillation from DenseNet teacher**
- **Test Accuracy:** 61.5% | Macro F1: 0.59 | Weighted F1: 0.63
- **Fine-tuning** on low-F1 classes (F1 < 0.45) improved per-class scores significantly  
  (e.g., artstudio +0.24, museum +0.38, waitingroom +0.33)

---

### 4. FOSNet (Lightweight CNN)
- 13 trainable layers, 1.6M params
- **Test Accuracy:** 49.4%
- Outperformed other models on some niche classes (e.g., bedroom, warehouse) despite lower overall accuracy.

---

## Knowledge Distillation
- **Teacher:** DenseNet-Like  
- **Students:** Hybrid CNN + SE, MiniEfficientNet, FOSNet  
- Improves student generalization without increasing complexity.

---

## Fine-tuning Low-F1 Classes
- Identified underperforming classes (F1 < 0.45)  
- Used:
  - WeightedRandomSampler
  - Strong augmentations
  - Reduced learning rate
- Large F1 boosts for classes like `auditorium`, `livingroom`, `artstudio`, etc.

---

## Ensemble Learning (Soft & Weighted Voting)

| Model                 | Acc | Macro F1 | Weighted F1 | Notes                                |
|-----------------------|-----|----------|------------|---------------------------------------|
| DenseNet (Teacher)    | 62% | 0.60     | 0.62       | Strong single model                  |
| MiniEfficientNet      | 61.5% | 0.59   | 0.63       | High-performing student              |
| FOSNet                | 49% | 0.44     | 0.50       | Lightweight, good on niche classes   |
| **Soft Voting**       | 76% | 0.77     | 0.76       | Boosted rare class recall           |
| **Weighted Voting**   | **77%** | **0.80** | **0.76** | Best overall - balances model strengths |

- **Soft Voting:** average of logits -> strong general improvement.  
- **Weighted Voting:** higher weight to MiniEfficientNet; improved rare class performance.

*Ensemble significantly improved overall accuracy and stabilized class-wise performance*

---

## Performance Highlights

- DenseNet: robust feature extractor  
- MiniEfficientNet + distillation: strong lightweight model  
- FOSNet: niche class coverage  
- Ensemble: lifted performance to 77% accuracy and 0.80 macro recall

---

## Key Challenges & Lessons

- Class imbalance was the biggest bottleneck. Even with class weights + label smoothing, rare classes remained tricky.  
- Training DenseNet (~9.9M params) required long training times (4–5 mins/epoch, 8+ hrs total).  
- Lightweight students trained faster but needed distillation and fine tuning to close the performance gap.  
- Ensemble learning gave the best balance between accuracy, recall, and robustness.

---

## Tech Stack
- Python 3 · PyTorch · Torchvision  
- NumPy · Pandas · scikit-learn  
- Matplotlib · tqdm

---

##  Future Work
- Integrate ViT backbones for richer features.  
- Edge deployment with student models for real-time inference.

---

## References
- MIT Indoor Scene Recognition Dataset  https://web.mit.edu/torralba/www/indoor.html

