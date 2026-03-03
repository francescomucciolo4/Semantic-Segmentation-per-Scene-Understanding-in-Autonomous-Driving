import os
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Configurazione
base_root = os.getcwd()
images_dir = Path(base_root) / '701_StillsRaw_full'
labels_dir = Path(base_root) / 'LabeledApproved_full'

print(f"📁 Images: {images_dir}")
print(f"📁 Labels: {labels_dir}")

# Mapping classi CamVid (32 classi)

class_mapping = {}

with open('label_colors.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        if len(parts) < 4:
            continue
        
        r = int(parts[0])
        g = int(parts[1])
        b = int(parts[2])
        class_name = parts[3]
        
        class_mapping[(r, g, b)] = class_name
 

# 1. CONTA FILE
print("\n" + "="*60)
print("📊 FILE COUNTING")
print("="*60)

image_files = sorted(list(images_dir.glob('*.png')))
label_files = sorted(list(labels_dir.glob('*.png')))

print(f"Images: {len(image_files)}")
print(f"Labels: {len(label_files)}")

# 2. MATCHING
print("\n" + "="*60)
print("🔗 MATCHING")
print("="*60)

# Dizionari che hanno per chiave lo stem (nome file senza estensione) e per valore il path del file
image_stems = {f.stem: f for f in image_files}
label_stems = {f.stem.replace('_L', ''): f for f in label_files}

matched = set(image_stems.keys()) & set(label_stems.keys())
matched_files = [(image_stems[s], label_stems[s]) for s in sorted(matched)]

print(f"✅ Matched pairs: {len(matched_files)}")
print(f"⚠️  Images only: {len(image_stems) - len(matched)}")
print(f"⚠️  Labels only: {len(label_stems) - len(matched)}")

if len(matched_files) == 0:
    print("\n❌ No matching found! Cannot continue.")
    exit()

# 3. DIMENSIONI
print("\n" + "="*60)
print("📏 DIMENSIONS")
print("="*60)

print("Checking all images...")
dimensions = defaultdict(int)
for img_path, _ in tqdm(matched_files, desc="Analyzing dimensions"):
    img = cv2.imread(str(img_path))
    if img is not None:
        h, w = img.shape[:2]
        dimensions[f"{w}x{h}"] += 1

print(f"\nImage dimensions found:")
for dim, count in sorted(dimensions.items(), key=lambda x: x[1], reverse=True):
    pct = (count / len(matched_files)) * 100
    print(f"  {dim}: {count} images ({pct:.1f}%)")

# 4. ANALISI CLASSI RGB
print("\n" + "="*60)
print("🎨 RGB CLASS DISTRIBUTION ANALYSIS")
print("="*60)

print(f"Analyzing ALL {len(matched_files)} labels...")

# Contatori per classe
class_pixel_counts = defaultdict(int)
class_image_counts = defaultdict(int)
total_pixels = 0
unknown_colors = set()

for _, lbl_path in tqdm(matched_files, desc="Analyzing labels"):
    label = cv2.imread(str(lbl_path))
    if label is None:
        continue
    
    label_rgb = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    pixels = label_rgb.reshape(-1, 3)
    total_pixels += len(pixels)
    
    # Trova colori unici in questa immagine
    unique_colors_in_img = np.unique(pixels, axis=0)
    
    for color in unique_colors_in_img:
        color_tuple = tuple(color)
        
        # Conta pixel di questo colore
        mask = np.all(pixels == color, axis=1)
        pixel_count = mask.sum()
        
        if color_tuple in class_mapping:
            class_name = class_mapping[color_tuple]
            class_pixel_counts[class_name] += pixel_count
            class_image_counts[class_name] += 1
        else:
            unknown_colors.add(color_tuple)
            class_pixel_counts['UNKNOWN'] += pixel_count

# Ordina per frequenza
sorted_classes = sorted(
    class_pixel_counts.items(), 
    key=lambda x: x[1], 
    reverse=True
)

print(f"\n✓ Total pixels analyzed: {total_pixels:,}")
print(f"✓ Classes found: {len([c for c in sorted_classes if c[0] != 'UNKNOWN'])}/32")

if unknown_colors:
    print(f"⚠️  Unknown colors found: {len(unknown_colors)}")
    print(f"   Examples: {list(unknown_colors)[:5]}")

print(f"\n{'#':<4} {'Class Name':<25} {'Pixels':<15} {'%':<8} {'In N Images':<12}")
print("-" * 75)

for idx, (class_name, pixel_count) in enumerate(sorted_classes):
    pct = (pixel_count / total_pixels) * 100
    img_count = class_image_counts.get(class_name, 0)
    print(f"{idx:<4} {class_name:<25} {pixel_count:>12,}  {pct:>6.2f}%  {img_count:>8}")

# Identifica classi assenti
present_classes = set(class_pixel_counts.keys()) - {'UNKNOWN'}
all_classes = set(class_mapping.values())
missing_classes = all_classes - present_classes

if missing_classes:
    print(f"\n⚠️  Missing classes (not found in dataset):")
    for cls in sorted(missing_classes):
        print(f"   - {cls}")

# Salva distribuzione classi in file txt
print("\n💾 Saving class distribution to file...")

output_file = 'class_distribution.txt'

with open(output_file, 'w') as f:
    f.write("="*75 + "\n")
    f.write("CAMVID CLASS DISTRIBUTION ANALYSIS\n")
    f.write("="*75 + "\n\n")
    
    f.write(f"Total pixels analyzed: {total_pixels:,}\n")
    f.write(f"Total images: {len(matched_files)}\n")
    f.write(f"Classes found: {len([c for c in sorted_classes if c[0] != 'UNKNOWN'])}/32\n\n")
    
    f.write(f"{'#':<4} {'Class Name':<25} {'Pixels':<15} {'%':<10} {'In N Images':<12}\n")
    f.write("-" * 75 + "\n")
    
    for idx, (class_name, pixel_count) in enumerate(sorted_classes):
        pct = (pixel_count / total_pixels) * 100
        img_count = class_image_counts.get(class_name, 0)
        f.write(f"{idx:<4} {class_name:<25} {pixel_count:>12,}  {pct:>8.2f}%  {img_count:>10}\n")
    
    if missing_classes:
        f.write("\n" + "="*75 + "\n")
        f.write("MISSING CLASSES (not found in dataset):\n")
        f.write("="*75 + "\n")
        for cls in sorted(missing_classes):
            f.write(f"  - {cls}\n")

print(f"✓ Class distribution saved to: {output_file}")

# 5. VISUALIZZAZIONE
print("\n" + "="*60)
print("👁️  SAMPLE VISUALIZATION")
print("="*60)

num_samples = min(6, len(matched_files))
samples = random.sample(matched_files, num_samples)

fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3*num_samples))

if num_samples == 1:
    axes = axes.reshape(1, -1)

for idx, (img_path, lbl_path) in enumerate(samples):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    lbl = cv2.imread(str(lbl_path))
    lbl_rgb = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_rgb, 0.6, lbl_rgb, 0.4, 0)
    
    axes[idx, 0].imshow(img_rgb)
    axes[idx, 0].set_title(f'Image: {img_path.name[:20]}...', fontsize=8)
    axes[idx, 0].axis('off')
    
    axes[idx, 1].imshow(lbl_rgb)
    axes[idx, 1].set_title(f'Label', fontsize=8)
    axes[idx, 1].axis('off')
    
    axes[idx, 2].imshow(overlay)
    axes[idx, 2].set_title('Overlay', fontsize=8)
    axes[idx, 2].axis('off')

plt.tight_layout()
plt.savefig('camvid_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved: camvid_analysis.png")
plt.show()

# 6. SUMMARY
print("\n" + "="*60)
print("📋 ANALYSIS SUMMARY")
print("="*60)
print(f"✅ Total image-label pairs: {len(matched_files)}")
print(f"✅ Image dimensions: {list(dimensions.keys())[0]}")
print(f"✅ Classes present: {len(present_classes)}/32")
print(f"✅ Total pixels: {total_pixels:,}")

# Top 5 classi più frequenti
print(f"\n🔝 Top 5 most frequent classes:")
for idx, (class_name, pixel_count) in enumerate(sorted_classes[:5]):
    pct = (pixel_count / total_pixels) * 100
    print(f"   {idx+1}. {class_name}: {pct:.2f}%")

# Bottom 5 classi più rare
print(f"\n🔻 Top 5 rarest classes:")
for idx, (class_name, pixel_count) in enumerate(list(reversed(sorted_classes))[:5]):
    if class_name == 'UNKNOWN':
        continue
    pct = (pixel_count / total_pixels) * 100
    print(f"   {class_name}: {pct:.4f}%")

print("\n✅ ANALYSIS COMPLETE!\n")