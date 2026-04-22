# Skin Lesion Classification via Self-Supervised Learning (SimCLR)
This project applies **SIMCLR-based Self Supervised Learning** for skin lesion classification into 7 categories, without relying on labels during the pretraining phase. 

## Dataset
This project uses the **DermaMNIST** dataset from the MedMNIST collection. The different classes present in the dataset are:

| Code | Full Name |
|------|-----------|
| AKIEC | Actinic Keratoses and Intraepithelial Carcinoma |
| BCC | Basal Cell Carcinoma |
| BKL | Benign Keratosis-like Lesions |
| DF | Dermatofibroma |
| MEL | Melanoma |
| NV | Melanocytic Nevi |
| VASC | Vascular Lesions |

* Image shape : 28 × 28 (RGB) 
* Total Training Samples = 7007
* Total Validation Samples = 1003
* Total Testing Samples = 2005
* DermaMNIST is a lightweight, standardized subset derived from the HAM10000 dataset, designed for benchmarking medical image classification models.

The official repository : [MedMNIST](https://github.com/MedMNIST/MedMNIST)

## Pipeline Overview

```
Raw Images
    │
    ▼
SimCLR Augmentation (two views)
    │
    ▼
Encoder (ResNet-style CNN) ──► Projection Head
    │                               │
    │                        NT-Xent Loss (SSL)
    │
    ▼
Frozen / Unfrozen Encoder
    │
    ▼
Linear Classifier (fc layer trained)
    │
    ▼
Evaluation (Confusion Matrix, Classification Report, t-SNE)
```
## Methodology
### Phase 1 -- SSL Pretraining
The encoder is pretrained using the **SimCLR** framework for **1000** epochs with no labels. Each image is passed through two stochastic augmentation pipelines to produce a postive pair and other images in the batch serve as negatives.

#### Loss -- NT-Xent (Normalized Temperature-scaled Cross Entropy)
The NT-Xent loss encourages representations of positive pairs (two views of the same image) to be close in embedding space, while pushing apart representations of different images (negatives). The loss is computed symmetrically over all pairs in the batch.

#### Architecture
| Component | Details |
|---|---|
| Encoder | CNN backbone outputting 512-dim representations |
| Projection Head | 2-layer MLP mapping 512 → 128-dim for contrastive loss |
| Optimizer | AdamW |
| Augmentations | Random crop, horizontal flip, colour jitter, grayscale |


> The projection head is **discarded** after pretraining — only the encoder weights are carried forward to Phase 2.


### Phase 2 -- Downstream Linear Evaluation

A single fully-connected layer is appended to the pre-trained encoder and trained on labeled data. Three configurations are evaluated to understand the impact of encoder freezing and training duration.

| # | Configuration | Epochs |
|---|---|---|
| 1 | Frozen encoder — linear probe only | 20 |
| 2 | Unfrozen encoder — full fine-tune | 20 |
| 3 | Unfrozen encoder — full fine-tune | 100 |


#### Architecture
| Component | Details |
|---|---|
| Encoder | Frozen or Unfrozen depending on configuration |
| Classifier | Single `nn.Linear(512, 7)` |
| Loss | Cross Entropy |
| Optimizer | AdamW (lr=3e-4, weight_decay=1e-4) |

> Only the linear layer's parameters receive gradients — the encoder serves purely as a fixed feature extractor. If the SSL representations are good, a linear layer alone should be sufficient to separate the classes.


### Phase 3 -- Evaluation
The trained classifier is evaluated on both the held-out train and test set using three methods:

- **Classification Report** — per-class precision, recall, and F1-score
- **Confusion Matrix** — visualises where the model confuses classes
- **t-SNE** — projects the 512-dim encoder outputs to 2D to assess the quality of learned representations

> Full results are in **[Results.md](Results.md)**




## Limitations & Future Work

### Root Causes of Poor Performance

1. **Severe class imbalance** — NV has ~57× more samples than DF; the linear classifier defaults to NV for uncertain inputs
2. **SSL pretraining is label-agnostic** — SimCLR optimises for augmentation invariance, not class separability hence learned features may not align with fine-grained lesion differences
3. **Insufficient fine-tuning in short runs** — unfreezing the encoder for only 20 epochs causes precision gains but recall losses, with no net improvement in macro F1
4. **VASC** is unlearnable at current data volumes — 28 test samples provides no recoverable signal under any configuration

### Findings from the Three Experiments 

The most important finding is that **overall accuracy is a misleading metric here** — all three configurations achieve 70%, yet the unfrozen 100-epoch run is meaningfully better by macro F1 (0.31 vs 0.26). Unfreezing alone without sufficient training duration does not help. Extended fine-tuning is the only configuration where minority class recall genuinely improves, and the t-SNE begins to show emerging structure for the first time.

### Suggested Improvements

| Strategy | Expected Impact |
|---|---|
| **Weighted Cross Entropy / Focal Loss** | Penalises NV over-predictions, improving minority class recall |
| **Oversampling (SMOTE) or undersampling** | Balances the training distribution |
| **Longer downstream training** (more than 100 epochs) | Loss still declining at epoch 100 - more epochs may likely yield better results |
| **Class-balanced batch sampling** | Ensures each class is seen equally per step |
| **Larger Encoder Capacity** | 512-dim representations may be insufficient for 7 fine-grained classes | 


## Project Structure

```
├── Images                      # Contains the images of the Results
|    ├── Downstream_100epochs
|    ├── Frozen Encoder
|    └── Unfrozen Encoder
├── checkpoints/
│   ├── ssl_chkpt.pth                        # SSL pretrained encoder + projection head
│   ├── down_chkpt_frozen.pth                # Frozen encoder, 20 epochs
│   ├── down_chkpt_unfrozen_20.pth           # Unfrozen encoder, 20 epochs
│   └── down_chkpt_unfrozen_100.pth          # Unfrozen encoder, 100 epochs
├── src/
│   ├── Model.py                             # Encoder, SSL (SimCLR) model definition
│   ├── Loss.py                              # NT-Xent contrastive loss
│   ├── Data.py                              # DataLoaders
│   ├── Augment.py                           # SimCLR augmentation pipeline
│   ├── Downstream_model.py                  # LinearClassifier definition
│   ├── Downstream_train.py                  # Training the LinearClassifier  └──
│   ├── Training.py                          # Training the SimCLR model
|   ├── Testing.py                           # Testing the classifier model
|   └── tSNE.py                              # Plotting the tSNE of the 
├── README.md
└── Results.md
```



## Environment and Requirements
### Python Version
Python = 3.9.23

### Dependencies
Please refer to [requirements](requirements.txt)

### Hardware
GPU: Nvidia GeForce RTX 2080 Ti








