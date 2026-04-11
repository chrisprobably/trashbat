import torchvision.transforms as transforms

LARGE_IMG_SIZE = 128
MEDIUM_IMG_SIZE = 64
SMALL_IMG_SIZE = 32

stretch_transform = transforms.Compose(
    [
        transforms.Resize((MEDIUM_IMG_SIZE, MEDIUM_IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

stretch_transform_small = transforms.Compose(
    [
        transforms.Resize((SMALL_IMG_SIZE, SMALL_IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

# Resize long side to IMG_SIZE preserving aspect ratio, then pad short side
letterbox_transform = transforms.Compose(
    [
        transforms.Resize(MEDIUM_IMG_SIZE),
        transforms.CenterCrop(MEDIUM_IMG_SIZE),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)
