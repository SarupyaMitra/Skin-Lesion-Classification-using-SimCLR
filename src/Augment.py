import torch
from torchvision import transforms


class SimCLRAugment:
    def __init__(self):
        self.transform = transforms.Compose([
            
            # 1. Random crop 
            transforms.RandomResizedCrop(
                size=28,
                scale=(0.7, 1.0)
            ),

            # 2. Flip
            transforms.RandomHorizontalFlip(p=0.5),

            # 3. Color jitter
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            ], p=0.8),

            # 4. Random grayscale
            transforms.RandomGrayscale(p=0.2),

            # 5. Convert to tensor
            #transforms.ToTensor(),

            # 6. Normalize (important)
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __call__(self, x):
        return self.transform(x)