import torchvision.transforms as transforms

LARGE_IMG_SIZE = 128
MEDIUM_IMG_SIZE = 64
SMALL_IMG_SIZE = 32

resize_med = transforms.Compose(
    [
        transforms.Resize((MEDIUM_IMG_SIZE, MEDIUM_IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

resize_med_norm = transforms.Compose(
    [
        transforms.Resize((MEDIUM_IMG_SIZE, MEDIUM_IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)

resize_small_colour = transforms.Compose(
    [
        transforms.Resize((SMALL_IMG_SIZE, SMALL_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

resize_small = transforms.Compose(
    [
        transforms.Resize((SMALL_IMG_SIZE, SMALL_IMG_SIZE)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)

# Resize long side to IMG_SIZE preserving aspect ratio, then pad short side
resize_med_letterbox = transforms.Compose(
    [
        transforms.Resize(MEDIUM_IMG_SIZE),
        transforms.CenterCrop(MEDIUM_IMG_SIZE),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]
)
