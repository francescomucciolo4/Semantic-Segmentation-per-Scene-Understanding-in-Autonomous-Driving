import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

IMG_HEIGHT = 384
IMG_WIDTH = 512
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_training_augmentation():
    
    train_transform = A.Compose([
        # Resize con padding per mantenere aspect ratio
        A.LongestMaxSize(max_size=max(IMG_SIZE), p=1.0),
        A.PadIfNeeded(
            min_height=IMG_SIZE[0],
            min_width=IMG_SIZE[1],
            border_mode=cv2.BORDER_CONSTANT,
            p=1.0
        ),
        
        # Geometric augmentation
        A.HorizontalFlip(p=0.5),
        
        A.Affine(
            scale=(0.8, 1.2),     # Scale ±20%
            rotate=(-15, 15),     # Rotate ±15°
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # Shift ±10%
            p=0.5
        ),
        
        # Random crop
        A.RandomCrop(
            height=IMG_SIZE[0],
            width=IMG_SIZE[1],
            p=1.0
        ),
        
        # Color augmentation
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=1
            ),
        ], p=0.5),
        

        A.OneOf([
            A.GaussNoise(p=1), 
            A.GaussianBlur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
        ], p=0.3),
        
    
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1),
            num_shadows_limit=(1, 2),  
            shadow_dimension=5,
            p=0.2
        ),
        
        # Normalizzazione ImageNet
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
        
        # Converti in tensore PyTorch
        ToTensorV2(),
    ])
    
    return train_transform


def get_validation_augmentation():
    
    val_transform = A.Compose([
        A.Resize(
            height=IMG_SIZE[0],
            width=IMG_SIZE[1],
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),
        
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
        
        ToTensorV2(),
    ])
    
    return val_transform


def get_test_augmentation():
    return get_validation_augmentation()

TRAIN_TRANSFORM = get_training_augmentation()
VAL_TRANSFORM = get_validation_augmentation()
TEST_TRANSFORM = get_validation_augmentation()
