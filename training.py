import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp
from pathlib import Path
import time

from dataset import train_loader, val_loader

# Configurazione
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_CLASSES = 11
VOID_CLASS = 255
NUM_EPOCHS = 20
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001

# Paths
CHECKPOINT_DIR = Path('./checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR = Path('./runs')
LOG_DIR.mkdir(exist_ok=True)

# Salvataggio
SAVE_EVERY = 10  # Salva checkpoint ogni N epoche
BEST_MODEL_PATH = CHECKPOINT_DIR / 'best_model.pth'
LAST_MODEL_PATH = CHECKPOINT_DIR / 'last_model.pth'

CLASS_NAMES = [
    'Sky', 'Building', 'Pole', 'Road', 'Pavement',
    'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
]

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss per segmentazione"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        probs = torch.softmax(predictions, dim=1)
        
        # FILTRA 255 prima del one-hot encoding
        targets_filtered = targets.clone()
        targets_filtered[targets == VOID_CLASS] = 0
        
        # One-hot encoding (ora senza 255)
        targets_one_hot = torch.nn.functional.one_hot(targets_filtered, num_classes=NUM_CLASSES)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Maschera per ignorare Void
        valid_mask = (targets != VOID_CLASS).unsqueeze(1).float()
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask
        
        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality = probs.sum(dims) + targets_one_hot.sum(dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice_score
        
        return dice_loss.mean()


class FocalLoss(nn.Module):
    """Focal Loss per class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets,
            ignore_index=VOID_CLASS,
            reduction='none'
        )
        
        p_t = torch.exp(-ce_loss)
        focal_term = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            # FILTRA i pixel Void prima di usare gather
            valid_mask = (targets != VOID_CLASS)
            targets_filtered = targets.clone()
            targets_filtered[~valid_mask] = 0  # Sostituisci 255 → 0
            
            alpha_t = self.alpha.gather(0, targets_filtered.view(-1))
            alpha_t = alpha_t.view_as(targets)
            
            # Azzera alpha per pixel Void
            alpha_t = alpha_t * valid_mask.float()
            
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        # Considera solo pixel validi
        valid_mask = (targets != VOID_CLASS)
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        else:
            return focal_loss.mean()  # Fallback se tutti i pixel sono Void


class CombinedLoss(nn.Module):
    """Combina Dice Loss e Focal Loss"""
    
    def __init__(self, alpha=None, gamma=2.0, lambda_dice=0.5, lambda_focal=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        combined = self.lambda_dice * dice + self.lambda_focal * focal
        return combined, dice, focal


# ============================================================================
# METRICHE
# ============================================================================

class SegmentationMetrics:
    """Calcola metriche per segmentazione semantica"""
    
    def __init__(self, num_classes=11, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(self, predictions, targets):
        predictions = predictions.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        for pred, target in zip(predictions, targets):
            if 0 <= pred < self.num_classes and 0 <= target < self.num_classes:
                self.confusion_matrix[target, pred] += 1
    
    def get_metrics(self):
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) + 
                self.confusion_matrix.sum(axis=0) - 
                intersection)
        
        iou = intersection / (union + 1e-10)
        
        valid_classes = union > 0
        miou = iou[valid_classes].mean()
        
        pixel_acc = intersection.sum() / (self.confusion_matrix.sum() + 1e-10)
        
        class_acc = intersection / (self.confusion_matrix.sum(axis=1) + 1e-10)
        mean_acc = class_acc[valid_classes].mean()
        
        return {
            'mIoU': miou,
            'pixel_acc': pixel_acc,
            'mean_acc': mean_acc,
            'class_iou': iou
        }


# ============================================================================
# TRAINING/VALIDATION
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    
    running_loss = 0.0
    running_dice = 0.0
    running_focal = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
    
    for _, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss, dice, focal = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(outputs, dim=1)
        metrics.update(preds, masks)
        
        running_loss += loss.item()
        running_dice += dice.item()
        running_focal += focal.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}',
            'focal': f'{focal.item():.4f}'
        })
    
    epoch_metrics = metrics.get_metrics()
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_focal = running_focal / len(dataloader)
    
    return {
        'loss': epoch_loss,
        'dice': epoch_dice,
        'focal': epoch_focal,
        **epoch_metrics
    }


def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    
    running_loss = 0.0
    running_dice = 0.0
    running_focal = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]')
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            loss, dice, focal = criterion(outputs, masks)
            
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)
            
            running_loss += loss.item()
            running_dice += dice.item()
            running_focal += focal.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{metrics.get_metrics()["mIoU"]:.4f}'
            })
    
    epoch_metrics = metrics.get_metrics()
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_focal = running_focal / len(dataloader)
    
    return {
        'loss': epoch_loss,
        'dice': epoch_dice,
        'focal': epoch_focal,
        **epoch_metrics
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🚀 CAMVID SEMANTIC SEGMENTATION TRAINING")
    print("="*70)
    
    print(f"\n📱 Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
    
    print("\n🏗️  Creating model...")
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )
    model = model.to(DEVICE)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   Model size: ~{trainable * 4 / 1024**2:.2f} MB")
    
    print("\n⚖️  Calculating class weights...")
    class_pixels = np.array([
        76801167, 117373718, 6588576, 140469061, 33681081,
        54258673, 3434413, 6921061, 24485634, 3379345, 2542545
    ])
    
    total_pixels = class_pixels.sum()
    class_freq = class_pixels / total_pixels
    class_weights = 1.0 / (class_freq + 1e-6)
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    print("   Class weights:")
    for i, (name, weight) in enumerate(zip(CLASS_NAMES, class_weights)):
        print(f"   {i:2d} {name:15s}: {weight:.3f}")
    
    criterion = CombinedLoss(
        alpha=class_weights,
        gamma=2.0,
        lambda_dice=0.5,
        lambda_focal=0.5
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )
    
    writer = SummaryWriter(log_dir=LOG_DIR / f'run_{int(time.time())}')
    
    print("\n" + "="*70)
    print("🔥 STARTING TRAINING")
    print("="*70)
    
    best_miou = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}")
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )
        
        val_metrics = validate(
            model, val_loader, criterion, DEVICE, epoch
        )
        
        scheduler.step(val_metrics['mIoU'])
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('mIoU/train', train_metrics['mIoU'], epoch)
        writer.add_scalar('mIoU/val', val_metrics['mIoU'], epoch)
        writer.add_scalar('PixelAcc/train', train_metrics['pixel_acc'], epoch)
        writer.add_scalar('PixelAcc/val', val_metrics['pixel_acc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train - Loss: {train_metrics['loss']:.4f} | mIoU: {train_metrics['mIoU']:.4f} | PixelAcc: {train_metrics['pixel_acc']:.4f}")
        print(f"   Val   - Loss: {val_metrics['loss']:.4f} | mIoU: {val_metrics['mIoU']:.4f} | PixelAcc: {val_metrics['pixel_acc']:.4f}")
        
        print(f"\n   Per-class IoU (validation):")
        for i, (name, iou) in enumerate(zip(CLASS_NAMES, val_metrics['class_iou'])):
            print(f"   {i:2d} {name:15s}: {iou:.4f}")
        
        if val_metrics['mIoU'] > best_miou:
            best_miou = val_metrics['mIoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'val_metrics': val_metrics
            }, BEST_MODEL_PATH)
            print(f"\n   ✅ New best model saved! mIoU: {best_miou:.4f}")
        
        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_path = CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path.name}")
    
    torch.save({
        'epoch': NUM_EPOCHS - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics
    }, LAST_MODEL_PATH)
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")
    
    writer.close()


if __name__ == '__main__':
    main()
