import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import os
from augmentation import TRAIN_TRANSFORM, VAL_TRANSFORM, TEST_TRANSFORM
from nuova_distribuzione import RGB_TO_CLASS_11 as RGB_TO_CLASS

# ============================================================================
# CONFIGURAZIONI
# ============================================================================

# Paths
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / 'CamVid_organized'

TRAIN_IMG_DIR = DATA_DIR / 'train' / 'images'
TRAIN_LBL_DIR = DATA_DIR / 'train' / 'labels'

VAL_IMG_DIR = DATA_DIR / 'val' / 'images'
VAL_LBL_DIR = DATA_DIR / 'val' / 'labels'

TEST_IMG_DIR = DATA_DIR / 'test' / 'images'
TEST_LBL_DIR = DATA_DIR / 'test' / 'labels'


# Training settings
BATCH_SIZE = 8
NUM_WORKERS = 0
PIN_MEMORY = True

# Class settings
NUM_CLASSES = 11
VOID_CLASS = 255



class CamVidDataset(Dataset):
    
    def __init__(self, images_dir, labels_dir, transform=None, use_preprocessed=True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
        self.transform = transform
        self.use_preprocessed = use_preprocessed
        self.rgb_to_class = RGB_TO_CLASS
        
        # Lista file immagini
        self.images = sorted(list(self.images_dir.glob('*.png')))
        
        # Usa masks preprocessate invece di labels RGB
        if use_preprocessed:
            self.masks_dir = self.labels_dir.parent / 'masks' 
        else:
            self.masks_dir = self.labels_dir
    
        
        # Match masks
        self.masks = []
        for img_path in self.images:
            if use_preprocessed:
                # Cerca file .npy
                mask_path = self.masks_dir / f"{img_path.stem}.npy"
            else:
                # Cerca file .png (vecchio metodo)
                mask_path = self.masks_dir / f"{img_path.stem}_L.png"
                if not mask_path.exists():
                    mask_path = self.masks_dir / f"{img_path.stem}.png"
            
            if mask_path.exists():
                self.masks.append(mask_path)
            else:
                raise FileNotFoundError(f"Mask not found for {img_path.name}")
        
        assert len(self.images) == len(self.masks)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Carica immagine
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Carica mask preprocessata
        mask_path = self.masks[idx]
        
        if self.use_preprocessed:
            mask = np.load(mask_path)
        else:
            # Vecchio metodo (lento)
            label_rgb = cv2.imread(str(mask_path))
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
            mask = self.rgb_to_class_mask(label_rgb)
        
        # Applica augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    
    train_dataset = CamVidDataset(
        images_dir=TRAIN_IMG_DIR,
        labels_dir=TRAIN_LBL_DIR,
        transform=TRAIN_TRANSFORM
    )
    
    
    val_dataset = CamVidDataset(
        images_dir=VAL_IMG_DIR,
        labels_dir=VAL_LBL_DIR,
        transform=VAL_TRANSFORM
    )
   
    
    test_dataset = CamVidDataset(
        images_dir=TEST_IMG_DIR,
        labels_dir=TEST_LBL_DIR,
        transform=TEST_TRANSFORM
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader



train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS
)


if __name__ == '__main__':
    print("="*70)
    print("📦 DATALOADERS CREATED")
    print("="*70)
    print(f"✓ Train dataset: {len(train_loader.dataset)} samples")
    print(f"✓ Val dataset: {len(val_loader.dataset)} samples")
    print(f"✓ Test dataset: {len(test_loader.dataset)} samples")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Num workers: {NUM_WORKERS}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Test un batch
    images, masks = next(iter(train_loader))
    print(f"\n🧪 Test batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")
    print("\n✅ Dataset working correctly!")

