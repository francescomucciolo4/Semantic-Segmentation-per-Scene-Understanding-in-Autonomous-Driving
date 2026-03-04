"""
Video Semantic Segmentation
Applica segmentazione semantica frame-by-frame su video
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import time 


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 11
INPUT_SIZE = (384, 512)  # (H, W)

# Colori per visualizzazione (BGR per OpenCV)
CLASS_COLORS = np.array([
    [128, 128, 128],  # 0: Sky
    [128, 0, 0],      # 1: Building
    [192, 192, 128],  # 2: Pole
    [128, 64, 128],   # 3: Road
    [0, 0, 192],      # 4: Pavement
    [128, 128, 0],    # 5: Tree
    [192, 128, 128],  # 6: SignSymbol
    [64, 64, 128],    # 7: Fence
    [64, 0, 128],     # 8: Vehicle
    [64, 64, 0],      # 9: Pedestrian
    [0, 128, 192]     # 10: Bicyclist
], dtype=np.uint8)

CLASS_NAMES = [
    'Sky', 'Building', 'Pole', 'Road', 'Pavement', 'Tree',
    'SignSymbol', 'Fence', 'Vehicle', 'Pedestrian', 'Bicyclist'
]


# ============================================================================
# PREPROCESSING
# ============================================================================

def get_preprocessing():
    """Preprocessing per inference (normalizzazione ImageNet)"""
    return A.Compose([
        A.Resize(INPUT_SIZE[0], INPUT_SIZE[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path):
    """Carica modello da checkpoint"""
    print(f"\n🔄 Loading model from {checkpoint_path}...")
    
    # Crea modello
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )
    
    # Carica pesi con weights_only=False
    checkpoint = torch.load(
        checkpoint_path, 
        map_location=DEVICE,
        weights_only=False  
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"✅ Model loaded successfully on {DEVICE}")
    
    return model


# ============================================================================
# SEGMENTATION FUNCTIONS
# ============================================================================

def mask_to_rgb(mask, colors=CLASS_COLORS):
    """Converte maschera (H, W) con indici 0-10 in RGB"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(NUM_CLASSES):
        rgb[mask == class_idx] = colors[class_idx]
    
    return rgb


def segment_frame(frame, model, preprocessing):
    """
    Segmenta singolo frame
    
    Args:
        frame: numpy array (H, W, 3) BGR
        model: modello PyTorch
        preprocessing: pipeline Albumentations
    
    Returns:
        mask: numpy array (H, W) con indici classe 0-10
    """
    # Converti BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Preprocessing
    augmented = preprocessing(image=frame_rgb)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return pred_mask


def create_overlay(frame, mask, alpha=0.5):
    """
    Crea overlay semi-trasparente
    
    Args:
        frame: frame originale BGR (H, W, 3)
        mask: maschera predetta (H, W)
        alpha: trasparenza overlay (0-1)
    
    Returns:
        overlay: frame con segmentazione sovrapposta
    """
    # Converti maschera in RGB
    mask_rgb = mask_to_rgb(mask)
    
    # Resize maschera a dimensione frame originale
    h, w = frame.shape[:2]
    mask_resized = cv2.resize(mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Converti RGB → BGR per OpenCV
    mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR)
    
    # Blend
    overlay = cv2.addWeighted(frame, 1 - alpha, mask_bgr, alpha, 0)
    
    return overlay


def add_info_overlay(frame, fps, frame_idx, total_frames):
    """Aggiunge informazioni testuali al frame"""
    # Info text
    info_text = [
        f"Frame: {frame_idx}/{total_frames}",
        f"FPS: {fps:.1f}"
    ]
    
    # Background semi-trasparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Testo
    y_offset = 30
    for text in info_text:
        cv2.putText(frame, text, (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    return frame


def add_legend(frame, show_legend=True):
    """Aggiunge legenda classi"""
    if not show_legend:
        return frame
    
    h, w = frame.shape[:2]
    legend_width = 200
    legend_height = NUM_CLASSES * 25 + 20
    
    # Background legenda
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - legend_width - 10, 10), 
                  (w - 10, legend_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Classi
    y_offset = 30
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        # Box colore (converti RGB → BGR)
        color_bgr = color[::-1]
        cv2.rectangle(frame, (w - legend_width, y_offset - 12),
                     (w - legend_width + 20, y_offset + 3), 
                     color_bgr.tolist(), -1)
        
        # Nome classe
        cv2.putText(frame, name, (w - legend_width + 25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 25
    
    return frame


# ============================================================================
# VIDEO PROCESSING
# ============================================================================

def process_video(
    input_path,
    output_path,
    model,
    preprocessing,
    alpha=0.5,
    show_legend=True,
    show_info=True,
    skip_frames=0,
    show_realtime=True,  
    save_output=True      
):
    
    print(f"\n{'='*70}")
    print(f"🎥 PROCESSING VIDEO")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    if save_output:
        print(f"Output: {output_path}")
    if show_realtime:
        print(f"Real-time preview: ENABLED (press 'q' to quit)")
    print(f"Device: {DEVICE}")
    
    # Apri video input
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Proprietà video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo properties:")
    print(f"  Resolution: {width}×{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f}s")
    
    # Setup video output (se richiesto)
    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Setup finestra display (se richiesto)
    if show_realtime:
        window_name = 'Segmentation Preview'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)  # Dimensione finestra
    
    # Processing
    print(f"\n🔄 Processing frames...")
    
    frame_idx = 0
    processed_frames = 0
    times = []
    
    pbar = tqdm(total=total_frames, desc="Processing", unit="frames")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_idx += 1
        
        # Skip frames se richiesto
        if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
            if save_output:
                out.write(frame)  # Scrivi frame originale
            pbar.update(1)
            continue
        
        # Segmentazione
        start_time = time.time()
        mask = segment_frame(frame, model, preprocessing)
        inference_time = time.time() - start_time
        times.append(inference_time)
        
        # Crea overlay
        overlay_frame = create_overlay(frame, mask, alpha)
        
        # Aggiungi info
        if show_info:
            current_fps = 1.0 / inference_time if inference_time > 0 else 0
            overlay_frame = add_info_overlay(
                overlay_frame, current_fps, frame_idx, total_frames
            )
        
        # Aggiungi legenda
        if show_legend:
            overlay_frame = add_legend(overlay_frame, show_legend)
        
        
        if show_realtime:
            cv2.imshow(window_name, overlay_frame)
            
            # Check per quit (ESC o 'q')
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q o ESC
                print("\n User interrupted processing")
                break
        
        
        # Scrivi frame output (se richiesto)
        if save_output:
            out.write(overlay_frame)
        
        processed_frames += 1
        pbar.update(1)
    
    pbar.close()
    
    # Cleanup
    cap.release()
    if save_output and out is not None:
        out.release()
    
    
    if show_realtime:
        cv2.destroyAllWindows()
    
    # Statistics
    if times:
        avg_time = np.mean(times)
        avg_fps = 1.0 / avg_time
        
        print(f"\n{'='*70}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Processed frames: {processed_frames}/{total_frames}")
        print(f"Average inference time: {avg_time*1000:.1f} ms/frame")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total processing time: {sum(times)/60:.1f} minutes")
        if save_output:
            print(f"Output saved: {output_path}")
        print(f"{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function"""
    
    # Path video input
    BASE_DIR = Path(__file__).parent
    input_path = BASE_DIR / 'video1.mp4'
    
    output_path = None
    
    checkpoint_path = Path('checkpoints/best_model.pth')
    
    # Parametri overlay
    alpha = 0.5
    show_legend = True
    show_info = True
    skip_frames = 0
    
    
    show_realtime = True   # Mostra mentre processa
    save_output = True     # Salva anche su file 
    
    
    print(f"\n{'='*70}")
    print(f"🎥 VIDEO SEMANTIC SEGMENTATION")
    print(f"{'='*70}")
    
    # Verifica input
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    print(f"✅ Input video found: {input_path}")
    
    # Setup output
    if save_output:
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_segmented.mp4"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"📂 Output will be saved to: {output_path}")
    else:
        print(f"📂 Real-time preview only (no output file)")
        output_path = None  # Assicurati che sia None
    
    # Verifica checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"🔧 Using checkpoint: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Preprocessing
    preprocessing = get_preprocessing()
    
    # Process video
    process_video(
        input_path=input_path,
        output_path=output_path,
        model=model,
        preprocessing=preprocessing,
        alpha=alpha,
        show_legend=show_legend,
        show_info=show_info,
        skip_frames=skip_frames,
        show_realtime=show_realtime,  
        save_output=save_output        
    )
    
    print(f"\n✅ DONE!")
    if save_output:
        print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    main()