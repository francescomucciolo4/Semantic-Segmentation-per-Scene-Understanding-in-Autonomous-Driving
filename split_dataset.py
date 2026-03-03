import os
from pathlib import Path
import shutil
import random
import numpy as np
from tqdm import tqdm

# Configurazione
base_root = os.getcwd()
images_dir = Path(base_root) / '701_StillsRaw_full'
labels_dir = Path(base_root) / 'LabeledApproved_full'
output_dir = Path(base_root) / 'CamVid_organized'

# Parametri split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

print("="*70)
print("📦 CAMVID DATASET ORGANIZATION")
print("="*70)
print(f"Source images: {images_dir}")
print(f"Source labels: {labels_dir}")
print(f"Output: {output_dir}")
print(f"Split: {TRAIN_RATIO*100:.0f}% train, {VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test")
print(f"Random seed: {SEED}")


print("\n" + "="*70)
print("🔍 STEP 1: COLLECTING FILES")
print("="*70)

image_files = sorted(list(images_dir.glob('*.png')))
label_files = sorted(list(labels_dir.glob('*.png')))

print(f"Images found: {len(image_files)}")
print(f"Labels found: {len(label_files)}")

image_stems = {f.stem: f for f in image_files}
label_stems = {f.stem.replace('_L', ''): f for f in label_files}

matched = set(image_stems.keys()) & set(label_stems.keys())
matched_files = [(image_stems[s], label_stems[s]) for s in sorted(matched)]

print(f"✅ Matched pairs: {len(matched_files)}")

if len(matched_files) == 0:
    print("❌ No matches found! Check file names.")
    exit()


print("\n" + "="*70)
print("✂️  STEP 2: CREATING SPLITS")
print("="*70)

random.seed(SEED)
np.random.seed(SEED)

files_shuffled = matched_files.copy()
random.shuffle(files_shuffled)

n_total = len(files_shuffled)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

train_files = files_shuffled[:n_train]
val_files = files_shuffled[n_train:n_train + n_val]
test_files = files_shuffled[n_train + n_val:]

print(f"Train: {len(train_files)} samples ({len(train_files)/n_total*100:.1f}%)")
print(f"Val:   {len(val_files)} samples ({len(val_files)/n_total*100:.1f}%)")
print(f"Test:  {len(test_files)} samples ({len(test_files)/n_total*100:.1f}%)")

splits = {
    'train': train_files,
    'val': val_files,
    'test': test_files
}

print("\n" + "="*70)
print("📁 STEP 3: CREATING DIRECTORY STRUCTURE")
print("="*70)

for split_name in ['train', 'val', 'test']:
    split_dir = output_dir / split_name
    (split_dir / 'images').mkdir(parents=True, exist_ok=True)
    (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
    print(f"✓ Created {split_dir}")

# 4. COPIA FILE
print("\n" + "="*70)
print("📋 STEP 4: COPYING FILES")
print("="*70)
print("ℹ️  Original files will be kept intact")

for split_name, files in splits.items():
    print(f"\nCopying {split_name} files...")
    
    for img_path, lbl_path in tqdm(files, desc=f"{split_name}"):
        # Destinazioni
        img_dest = output_dir / split_name / 'images' / img_path.name
        lbl_dest = output_dir / split_name / 'labels' / lbl_path.name
        
        # Copia
        shutil.copy2(str(img_path), str(img_dest))
        shutil.copy2(str(lbl_path), str(lbl_dest))
    
    print(f"✓ Copied {len(files)} pairs")

# 5. SALVA INFO SPLIT
print("\n" + "="*70)
print("💾 STEP 5: SAVING SPLIT INFO")
print("="*70)

info_dir = output_dir / 'split_info'
info_dir.mkdir(exist_ok=True)

for split_name, files in splits.items():
    info_file = info_dir / f'{split_name}_files.txt'
    
    with open(info_file, 'w') as f:
        f.write(f"# {split_name.upper()} SET - {len(files)} samples\n")
        f.write(f"# Seed: {SEED}\n")
        f.write(f"# Ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}\n\n")
        
        for img_path, lbl_path in files:
            f.write(f"{img_path.stem}\n")
    
    print(f"✓ Saved: {info_file}")

# 6. VERIFICA
print("\n" + "="*70)
print("✅ STEP 6: VERIFICATION")
print("="*70)

for split in ['train', 'val', 'test']:
    img_dir = output_dir / split / 'images'
    lbl_dir = output_dir / split / 'labels'
    
    n_img = len(list(img_dir.glob('*.png')))
    n_lbl = len(list(lbl_dir.glob('*.png')))
    
    status = "✅" if n_img == n_lbl else "❌"
    print(f"{status} {split:5s}: {n_img} images, {n_lbl} labels")

# 7. CLEANUP (rimuovi cartelle originali se vuote)
print("\n" + "="*70)
print("🧹 STEP 7: CLEANUP")
print("="*70)

if len(list(images_dir.glob('*.png'))) == 0:
    images_dir.rmdir()
    print(f"✓ Removed empty: {images_dir.name}")

if len(list(labels_dir.glob('*.png'))) == 0:
    labels_dir.rmdir()
    print(f"✓ Removed empty: {labels_dir.name}")

print("\n" + "="*70)
print("🎉 ORGANIZATION COMPLETE!")
print("="*70)
print(f"\nYour organized dataset:")
print(f"📁 {output_dir.absolute()}\n")
print("Next step: Create PyTorch Dataset class!")