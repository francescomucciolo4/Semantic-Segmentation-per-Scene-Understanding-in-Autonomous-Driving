# Semantic Segmentation per Scene Understanding in Autonomous Driving


<p align="center">
  <img src="https://github.com/user-attachments/assets/f732d2a8-a0b5-49a5-b581-096af76bc927" width="700">
</p>


Sistema di segmentazione semantica pixel-wise utilizzando **DeepLabV3+** con encoder **ResNet50** pre-trained sul dataset **CamVid**.  
Il progetto gestisce **class imbalance** (ratio 55:1) tramite **Combined Loss Function** e class weighting.
Il modello viene tetsato su un test set ricavato dal dataset **CamVid**, mentre il video è solo una dimostrazione in tempo reale dei risultati.

---

## Dataset

Il progetto utilizza il **CamVid (Cambridge-driving Labeled Video Database)**, un dataset standard per scene understanding in contesti di guida autonoma.

- **Totale immagini:** 701  
  - Training: 490 (70%)  
  - Validation: 105 (15%)  
  - Test: 106 (15%)  
- **Risoluzione originale:** 960×720 pixel (resize a 384×512 per training)  

Il dataset originale contiene 32 classi, ciascuna con un colore RGB specifico, più la classe **Void** per pixel non annotati o da ignorare.  
Per semplificare il problema, le 32 classi sono state accorpate in **11 macro-classi semantiche**, più la classe **Void** che contiene classi con pochi pixel o UNKNOWN.

**11 classi semantiche e percentuale di pixel:**

- Sky: 15.85%  
- Building: 24.22%  
- Pole: 1.36%  
- Road: 28.99%  
- Pavement: 6.95%  
- Tree: 11.20%  
- SignSymbol: 0.71%  
- Fence: 1.43%  
- Car: 5.05%  
- Pedestrian: 0.70%  
- Bicyclist: 0.52%  

>  Class imbalance: ratio 55:1 tra classe più frequente (**Road**) e più rara (**Bicyclist**).

---

## Architettura del Modello: DeepLabV3+ con ResNet50 Encoder

### Input
- Immagini RGB 384×512 normalizzate ImageNet

### Encoder
- **ResNet50 pre-trained** su ImageNet  
- Estrae features multi-scala a diversi livelli di profondità  
- Transfer learning per convergenza 3× più veloce

### ASPP Module (Atrous Spatial Pyramid Pooling)
- Cattura contesto multi-scala con dilatation rates `[6, 12, 18]`  
- Cruciale per piccoli oggetti (Pole, SignSymbol)

### Decoder
- Upsampling progressivo con skip connections  
- Output: Logits `(B, 11, H, W)` segmentazione pixel-wise  

**Parametri trainable:** 26M

---

## Metodologia

### Gestione Class Imbalance

1. **Combined Loss Function (50% Dice + 50% Focal)**  
   - **Dice Loss:** ottimizza overlap tra predizioni e ground truth, robusta per imbalance  
   - **Focal Loss:** down-weight esempi facili, focus su pixel difficili

2. **Class Weights**  
   - Pesi inversamente proporzionali alla frequenza delle classi  
   - Applicati nella Focal Loss per amplificare errori su classi rare

3. **Data Augmentation**  
   - **Geometric:** `HorizontalFlip`, `Affine` (rotation ±15°, scale 0.8–1.2)  
   - **Color:** `RandomBrightnessContrast`, `HueSaturationValue`, `RandomGamma`  
   - **Noise:** `GaussianBlur`, `MotionBlur`, `GaussNoise`  
   - **Other:** `RandomShadow`, `RandomCrop`

### Training
- Optimizer: **AdamW** (lr=3e-4, weight_decay=1e-4)  
- Scheduler: **ReduceLROnPlateau** (patience=10, factor=0.5)  
- Epochs: 20 (early stopping su validation mIoU)  
- Batch size: 8  
- Hardware: **NVIDIA RTX 4060 (8GB VRAM)**  
- Conversione maschere RGB-numpy offline  
- Gestione pixel **Void (255)** tramite valid mask  
- Normalizzazione ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

---

## Metriche di valutazione
- **mIoU (mean Intersection over Union):** media su tutte le 11 classi  
- **Pixel Accuracy:** percentuale di pixel classificati correttamente  
- **Per-Class IoU:** IoU individuale per ogni classe  
- **Confusion Matrix:** analisi errori tra classi  
- **Training Curves:** evolution di loss e mIoU durante training  

---

## Tecnologie utilizzate
- Python   
- PyTorch 
- segmentation-models-pytorch 
- Albumentations
- OpenCV   
- NumPy & Pandas   
- Matplotlib & Seaborn  
- TensorBoard   
- scikit-learn  

---

## Risultati

### Performance quantitativa

| Metrica           | Validation | Test        |
|------------------|-----------|------------|
| mIoU             | 73.9%     | 69.2%      |
| Pixel Accuracy   | 91.9%     | 89.4%      |
| Training Time    | -         | 45 min  |
| Inference Speed  | -         | ~42 FPS    |

---

### Per-Class IoU (Test Set)

| Classe       | IoU    | Note                                |
|-------------|-------|------------------------------------|
| Road        | 95.1% | Classe dominante, eccellente       |
| Sky         | 91.1% | Regioni ampie, facile              |
| Vehicle         | 84.1% | Oggetti distintivi                  |
| Building    | 84.0% | Texture riconoscibili               |
| Pavement    | 80.9% | Buona separazione da Road           |
| Tree        | 78.2% | Texture complessa ma gestita        |
| Bicyclist   | 67.9% | Ottimo miglioramento               |
| Fence       | 60.9% | Struttura sottile                   |
| Pedestrian  | 55.8% | Piccoli, occlusioni                 |
| SignSymbol  | 49.6% | Rari + piccoli                      |
| Pole        | 41.2% | Oggetti molto sottili               |




## Note

> Questo repository ha finalità esclusivamente illustrative e di portfolio personale.
>
> Parte del codice è stato generato utilizzando Claude.
