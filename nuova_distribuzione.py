import numpy as np
from collections import defaultdict

# Mapping 32 classi → 11 classi
CLASS_32_TO_11 = {
    'Sky': 0,
    'Building': 1,
    'Wall': 1,
    'Archway': 1,
    'Bridge': 1,
    'Tunnel': 1,
    'Column_Pole': 2,
    'TrafficLight': 2,
    'Road': 3,
    'LaneMkgsDriv': 3,
    'Sidewalk': 4,
    'LaneMkgsNonDriv': 4,
    'ParkingBlock': 4,
    'RoadShoulder': 4,
    'Tree': 5,
    'VegetationMisc': 5,
    'SignSymbol': 6,
    'Misc_Text': 6,
    'TrafficCone': 6,
    'Fence': 7,
    'Car': 8,
    'SUVPickupTruck': 8,
    'Truck_Bus': 8,
    'Train': 8,
    'MotorcycleScooter': 8,
    'OtherMoving': 8,
    'Pedestrian': 9,
    'Child': 9,
    'CartLuggagePram': 9,
    'Bicyclist': 10,
    'Void': 255,
    'Animal': 255,
    'UNKNOWN': 255,
}

CLASS_11_NAMES = [
    'Sky', 'Building', 'Pole', 'Road', 'Pavement',
    'Tree', 'SignSymbol', 'Fence', 'Vehicle', 'Pedestrian', 'Bicyclist'
]

def load_rgb_mapping(filepath='label_colors.txt'):
    """Carica mapping RGB → nome classe da file"""
    class_mapping = {}
    
    with open(filepath, 'r') as f:
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
    
    return class_mapping

def load_class_counts_from_file(filepath='class_distribution.txt'):
    """Legge il file class_distribution.txt e estrae i conteggi"""
    class_counts = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        start_idx = None
        for i, line in enumerate(lines):
            if line.startswith('-'):
                start_idx = i + 1
                break
        
        if start_idx is None:
            raise ValueError("Table not found in file")
        
        for line in lines[start_idx:]:
            if not line.strip() or line.startswith('='):
                break
            
            parts = line.split()
            if len(parts) >= 3:
                class_name = parts[1]
                pixel_str = parts[2]
                pixel_count = int(pixel_str.replace(',', ''))
                class_counts[class_name] = pixel_count
    
    return class_counts


CLASS_MAPPING = load_rgb_mapping('label_colors.txt')

RGB_TO_CLASS_11 = {}
for rgb, class_name in CLASS_MAPPING.items():
    class_11 = CLASS_32_TO_11.get(class_name, 255)
    RGB_TO_CLASS_11[rgb] = class_11



if __name__ == '__main__':
    print(f"✓ Loaded {len(CLASS_MAPPING)} classes from label_colors.txt")
    print("="*60)
    print("🎯 RGB TO 11-CLASS MAPPING CREATED")
    print("="*60)
    print("Mapping - ", RGB_TO_CLASS_11)
    print(f"Total RGB colors mapped: {len(RGB_TO_CLASS_11)}")
    print(f"Classes (0-10): {len(CLASS_11_NAMES)}")
    print(f"Void class: 255")
    
    print("\n" + "="*60)
    print("📊 RECALCULATING DISTRIBUTION WITH 11 CLASSES")
    print("="*60)
    
    class_11_pixel_counts = defaultdict(int)
    
    print("📂 Loading class distribution from file...")
    old_counts = load_class_counts_from_file('class_distribution.txt')
    print(f"✓ Loaded {len(old_counts)} classes from class_distribution.txt")
    
    total_pixels = sum(old_counts.values())
    
    for class_32, count in old_counts.items():
        class_11_idx = CLASS_32_TO_11.get(class_32, 255)
        
        if class_11_idx == 255:
            class_name = 'Void'
        else:
            class_name = CLASS_11_NAMES[class_11_idx]
        
        class_11_pixel_counts[class_name] += count
    
    sorted_11_classes = sorted(
        class_11_pixel_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    print(f"\n{'Class ID':<10} {'Class Name':<15} {'Pixels':<15} {'%':<10}")
    print("-" * 60)
    
    for class_name, pixel_count in sorted_11_classes:
        if class_name == 'Void':
            class_id = 255
        else:
            class_id = CLASS_11_NAMES.index(class_name)
        
        pct = (pixel_count / total_pixels) * 100
        print(f"{class_id:<10} {class_name:<15} {pixel_count:>12,}  {pct:>7.2f}%")
    
    print("\n" + "="*60)
    print("📋 11-CLASS SUMMARY")
    print("="*60)
    print(f"✅ Training classes: 11 (indices 0-10)")
    print(f"✅ Void class: 1 (index 255, ignored in training)")
    print(f"✅ Total pixels: {total_pixels:,}")
    
    max_pct = max([p for n, p in sorted_11_classes if n != 'Void'])
    min_pct = min([p for n, p in sorted_11_classes if n != 'Void'])
    max_name = max([(n, p) for n, p in sorted_11_classes if n != 'Void'], key=lambda x: x[1])[0]
    min_name = min([(n, p) for n, p in sorted_11_classes if n != 'Void'], key=lambda x: x[1])[0]
    
    imbalance_ratio = max_pct / min_pct
    
    print(f"\n⚖️  Class Imbalance:")
    print(f"   Most frequent: {max_name} ({max_pct/total_pixels*100:.2f}%)")
    print(f"   Least frequent: {min_name} ({min_pct/total_pixels*100:.2f}%)")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}x")
    
    print("\n✅ MAPPING COMPLETE!")