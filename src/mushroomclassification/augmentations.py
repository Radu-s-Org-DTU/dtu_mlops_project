import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_augmentation_transforms() -> dict:
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.MotionBlur(p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0)
        ], p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return {
        "train": train_transform,
        "val": val_transform,
        "test": test_transform
    }

if __name__ == "__main__":
    augmentations = get_augmentation_transforms()
    print("Train Transformations:", augmentations["train"])
    print("Validation Transformations:", augmentations["val"])
    print("Test Transformations:", augmentations["test"])
