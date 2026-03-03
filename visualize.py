import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import test_loader, NUM_CLASSES
from nuova_distribuzione import CLASS_11_NAMES

# Configurazione
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = Path('checkpoints/best_model.pth')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(exist_ok=True)

# Crea subdirectories
(OUTPUT_DIR / 'predictions').mkdir(exist_ok=True)
(OUTPUT_DIR / 'metrics').mkdir(exist_ok=True)
(OUTPUT_DIR / 'analysis').mkdir(exist_ok=True)

# Color palette per classi (colori distintivi)
CLASS_COLORS = np.array([
    [128, 128, 128],  # 0: Sky - Grigio chiaro
    [128, 0, 0],      # 1: Building - Rosso scuro
    [192, 192, 128],  # 2: Pole - Beige
    [128, 64, 128],   # 3: Road - Viola
    [0, 0, 192],      # 4: Pavement - Blu scuro
    [128, 128, 0],    # 5: Tree - Verde oliva
    [192, 128, 128],  # 6: SignSymbol - Rosa
    [64, 64, 128],    # 7: Fence - Blu-viola
    [64, 0, 128],     # 8: Car - Viola scuro
    [64, 64, 0],      # 9: Pedestrian - Verde scuro
    [0, 128, 192],    # 10: Bicyclist - Ciano
])


def load_model():
    """Carica il modello dal checkpoint"""
    print("Loading best model...")
    
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights=None,  # Carico i pesi del checkpoint
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded from {CHECKPOINT_PATH}")
    print(f"✓ Best mIoU: {checkpoint.get('best_miou', 'N/A'):.4f}")
    
    return model, checkpoint


def mask_to_rgb(mask):
    """Converte mask (H, W) con indici 0-10 in immagine RGB"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(NUM_CLASSES):
        rgb[mask == class_idx] = CLASS_COLORS[class_idx]
    
    return rgb


def denormalize_image(image):
    """De-normalizza immagine da ImageNet"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # image è (C, H, W), converti in (H, W, C)
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def create_prediction_visualization(images, masks_true, masks_pred, indices, save_dir):
    
    print("\nCreating prediction visualizations...")
    
    for idx in tqdm(indices, desc="Generating predictions"):
        image = denormalize_image(images[idx])
        mask_true = masks_true[idx].cpu().numpy()
        mask_pred = masks_pred[idx]
        
        # Filtra classe Void (255)
        mask_true_filtered = mask_true.copy()
        mask_true_filtered[mask_true == 255] = 0  # Temporaneo per visualizzazione
        
        # Converti in RGB
        mask_true_rgb = mask_to_rgb(mask_true_filtered)
        mask_pred_rgb = mask_to_rgb(mask_pred)
        
        # Crea figura
        _, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(image)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(mask_true_rgb)
        axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(mask_pred_rgb)
        axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'prediction_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(indices)} prediction visualizations to {save_dir}")


def create_overlay_visualization(images, masks_pred, indices, save_dir):
    """
    Crea visualizzazione con overlay semi-trasparente
    """
    print("\nCreating overlay visualizations...")
    
    for idx in tqdm(indices, desc="Generating overlays"):
        image = denormalize_image(images[idx])
        mask_pred = masks_pred[idx]
        
        # Converti mask in RGB
        mask_rgb = mask_to_rgb(mask_pred)
        
        # Overlay semi-trasparente
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1-alpha, mask_rgb, alpha, 0)
        
        # Crea figura
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(overlay)
        axes[1].set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'overlay_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(indices)} overlay visualizations to {save_dir}")


def create_legend():
    """Crea legenda con colori delle classi"""
    _, ax = plt.subplots(figsize=(8, 6))
    
    # Crea patches colorati
    for i, (name, color) in enumerate(zip(CLASS_11_NAMES, CLASS_COLORS)):
        rect = plt.Rectangle((0, i), 1, 0.8, facecolor=color/255.0)
        ax.add_patch(rect)
        ax.text(1.2, i+0.4, name, va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, len(CLASS_11_NAMES))
    ax.axis('off')
    ax.set_title('Class Legend', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'predictions' / 'legend.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Legend saved")


def calculate_iou_per_class(masks_true, masks_pred):
    """Calcola IoU per ogni classe"""
    iou_per_class = []
    
    for class_idx in range(NUM_CLASSES):
        # Trova pixel di questa classe
        true_mask = (masks_true == class_idx)
        pred_mask = (masks_pred == class_idx)
        
        # Intersection e Union
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        
        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
        
        iou_per_class.append(iou)
    
    return np.array(iou_per_class)


def create_iou_bar_chart(iou_per_class):
    """Crea bar chart con IoU per classe"""
    print("\nCreating IoU bar chart...")
    
    _, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(CLASS_11_NAMES))
    bars = ax.bar(x, iou_per_class * 100, color=CLASS_COLORS/255.0, edgecolor='black', linewidth=1.5)
    
    # Aggiungi valori sopra le barre
    for _, (bar, iou) in enumerate(zip(bars, iou_per_class)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{iou*100:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('IoU (%)', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class IoU on Test Set', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_11_NAMES, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # mIoU medio
    mean_iou = iou_per_class.mean()
    ax.axhline(mean_iou * 100, color='red', linestyle='--', linewidth=2, label=f'Mean IoU: {mean_iou*100:.2f}%')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'metrics' / 'iou_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ IoU bar chart saved (mIoU: {mean_iou*100:.2f}%)")


def create_confusion_matrix(masks_true, masks_pred):
    """Crea confusion matrix"""
    print("\nCreating confusion matrix...")
    
    # Filtra classe Void
    valid_mask = (masks_true != 255)
    masks_true_filtered = masks_true[valid_mask]
    masks_pred_filtered = masks_pred[valid_mask]
    
    # Calcola confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    for true_cls in range(NUM_CLASSES):
        for pred_cls in range(NUM_CLASSES):
            cm[true_cls, pred_cls] = np.sum(
                (masks_true_filtered == true_cls) & (masks_pred_filtered == pred_cls)
            )
    
    # Normalizza per riga (true class)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    # Plot
    _, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_11_NAMES, yticklabels=CLASS_11_NAMES,
                cbar_kws={'label': 'Normalized Frequency'}, ax=ax)
    
    ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'analysis' / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Confusion matrix saved")
    
    return cm


def generate_predictions(model, dataloader):
    """Genera predizioni su tutto il dataloader"""
    print("\nGenerating predictions on test set...")
    
    all_images = []
    all_masks_true = []
    all_masks_pred = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Predicting"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_images.append(images.cpu())
            all_masks_true.append(masks.cpu())
            all_masks_pred.append(preds.cpu())
    
    # Concatena
    all_images = torch.cat(all_images, dim=0)
    all_masks_true = torch.cat(all_masks_true, dim=0)
    all_masks_pred = torch.cat(all_masks_pred, dim=0).numpy()
    
    print(f"✓ Generated predictions for {len(all_images)} images")
    
    return all_images, all_masks_true, all_masks_pred


def create_statistics_report(iou_per_class, cm):
    """Crea report testuale con statistiche"""
    print("\nCreating statistics report...")
    
    report_path = OUTPUT_DIR / 'analysis' / 'statistics.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CAMVID SEMANTIC SEGMENTATION - TEST SET RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*70 + "\n")
        f.write("Architecture: DeepLabV3+\n")
        f.write("Encoder: ResNet50 (ImageNet pre-trained)\n")
        f.write("Classes: 11\n")
        f.write("Input size: 384×512\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*70 + "\n")
        mean_iou = iou_per_class.mean()
        f.write(f"Mean IoU (mIoU): {mean_iou*100:.2f}%\n")
        f.write(f"Best class: {CLASS_11_NAMES[np.argmax(iou_per_class)]} ({np.max(iou_per_class)*100:.2f}%)\n")
        f.write(f"Worst class: {CLASS_11_NAMES[np.argmin(iou_per_class)]} ({np.min(iou_per_class)*100:.2f}%)\n\n")
        
        f.write("PER-CLASS IoU\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Class':<15} {'IoU (%)':<10} {'Performance'}\n")
        f.write("-"*70 + "\n")
        
        for _, (name, iou) in enumerate(zip(CLASS_11_NAMES, iou_per_class)):
            if iou > 0.8:
                perf = "Excellent"
            elif iou > 0.7:
                perf = "Very Good"
            elif iou > 0.6:
                perf = "Good"
            elif iou > 0.5:
                perf = "Fair"
            else:
                perf = "Challenging"
            
            f.write(f"{name:<15} {iou*100:>7.2f}   {perf}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Statistics report saved to {report_path}")


def main():
    print("="*70)
    print("🎨 CAMVID SEGMENTATION VISUALIZATION")
    print("="*70)
    
    # Carica modello
    model, _ = load_model()
    
    # Genera predizioni su test set
    images, masks_true, masks_pred = generate_predictions(model, test_loader)
    
    # Seleziona esempi da visualizzare
    # Best examples (alta IoU)
    num_samples = min(10, len(images))
    indices_best = list(range(0, num_samples))
    
    # Random examples
    import random
    random.seed(42)
    indices_random = random.sample(range(len(images)), min(5, len(images)))
    
    # Crea visualizzazioni
    create_legend()
    
    create_prediction_visualization(
        images, masks_true, masks_pred,
        indices_best,
        OUTPUT_DIR / 'predictions'
    )
    
    create_overlay_visualization(
        images, masks_pred,
        indices_random,
        OUTPUT_DIR / 'predictions'
    )
    
    # Calcola metriche
    masks_true_np = masks_true.numpy()
    iou_per_class = calculate_iou_per_class(masks_true_np, masks_pred)
    
    # Crea grafici
    create_iou_bar_chart(iou_per_class)
    cm = create_confusion_matrix(masks_true_np, masks_pred)
    
    # Report
    create_statistics_report(iou_per_class, cm)
    
    print("\n" + "="*70)
    print("🎉 VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n📁 All results saved to: {OUTPUT_DIR.absolute()}")
    print("\n📊 Generated files:")
    print(f"  • Predictions: {len(list((OUTPUT_DIR / 'predictions').glob('*.png')))} images")
    print(f"  • Metrics: IoU bar chart")
    print(f"  • Analysis: Confusion matrix, statistics report")
    print(f"\n🏆 Test Set mIoU: {iou_per_class.mean()*100:.2f}%")


if __name__ == '__main__':
    main()