import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from nuova_distribuzione import RGB_TO_CLASS_11 as RGB_TO_CLASS

# Configurazioni
VOID_CLASS = 255

# Directories
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / 'CamVid_organized'

SPLITS = ['train', 'val', 'test']


def rgb_to_class_mask(label_rgb, rgb_to_class):
    """
    Converte label RGB in maschera di classi.
    Versione ottimizzata per preprocessing.
    """
    h, w = label_rgb.shape[:2]
    mask = np.full((h, w), VOID_CLASS, dtype=np.uint8) 
    
    # Per ogni colore RGB conosciuto
    for rgb_color, class_idx in rgb_to_class.items():
        # Match esatto vettorizzato
        match = np.all(label_rgb == rgb_color, axis=2)
        mask[match] = class_idx
    
    return mask


def preprocess_split(split_name):
    """Preprocessa un singolo split (train/val/test)"""
    
    print(f"\n{'='*70}")
    print(f"📂 Processing {split_name.upper()} split")
    print(f"{'='*70}")
    
    # Paths
    labels_dir = DATA_DIR / split_name / 'labels'
    masks_dir = DATA_DIR / split_name / 'masks'
    
    # Crea directory masks
    masks_dir.mkdir(exist_ok=True, parents=True)
    
    # Lista label files
    label_files = sorted(list(labels_dir.glob('*.png')))
    
    if len(label_files) == 0:
        print(f"⚠️  No label files found in {labels_dir}")
        return
    
    print(f"Found {len(label_files)} label files")
    
    # Processa ogni label
    for label_path in tqdm(label_files, desc=f"Converting {split_name}"):
        # Carica label RGB
        label_rgb = cv2.imread(str(label_path))
        label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
        
        # Converti RGB → mask
        mask = rgb_to_class_mask(label_rgb, RGB_TO_CLASS)
        
        # Salva come numpy (molto più veloce da caricare)
        # Rimuovi suffisso _L se presente
        stem = label_path.stem.replace('_L', '')
        mask_path = masks_dir / f"{stem}.npy"
        np.save(mask_path, mask)
    
    print(f"✅ {split_name}: {len(label_files)} masks saved to {masks_dir}")
    
    # Statistiche
    total_size_mb = sum(f.stat().st_size for f in masks_dir.glob('*.npy')) / (1024**2)
    print(f"   Total size: {total_size_mb:.2f} MB")


def verify_masks(split_name, num_samples=3):
    """Verifica che le maschere siano corrette"""
    
    print(f"\n🔍 Verifying {split_name} masks...")
    
    masks_dir = DATA_DIR / split_name / 'masks'
    mask_files = sorted(list(masks_dir.glob('*.npy')))
    
    if len(mask_files) == 0:
        print(f"⚠️  No masks found!")
        return
    
    # Controlla alcuni samples random
    import random
    samples = random.sample(mask_files, min(num_samples, len(mask_files)))
    
    for mask_path in samples:
        mask = np.load(mask_path)
        
        unique_classes = np.unique(mask)
        print(f"   {mask_path.name}: shape={mask.shape}, classes={unique_classes.tolist()}, dtype={mask.dtype}")
        
        # Verifica che non ci siano valori strani
        invalid = unique_classes[(unique_classes > 10) & (unique_classes != 255)]
        if len(invalid) > 0:
            print(f"   ⚠️  WARNING: Invalid class indices found: {invalid}")


def main():
    print("="*70)
    print("🎯 PREPROCESSING CAMVID MASKS")
    print("="*70)
    print(f"\nThis will convert RGB labels to class masks (0-10, 255)")
    print(f"Output format: numpy .npy files (uint8)")
    print(f"Location: {DATA_DIR}")
    
    # Verifica che le directory esistano
    if not DATA_DIR.exists():
        print(f"\n❌ ERROR: {DATA_DIR} not found!")
        return
    
    # Processa ogni split
    for split in SPLITS:
        preprocess_split(split)
    
    # Verifica risultati
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print("="*70)
    
    for split in SPLITS:
        verify_masks(split)


if __name__ == '__main__':
    main()