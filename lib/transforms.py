from typing import cast

import torch
import torchvision.transforms as transforms
from PIL import Image

LARGE_IMG_SIZE = 128
MEDIUM_IMG_SIZE = 64
SMALL_IMG_SIZE = 32

_small_colour_resize = transforms.Compose(
    [
        transforms.Resize((SMALL_IMG_SIZE, SMALL_IMG_SIZE)),
        transforms.ToTensor(),
    ]
)

_colour_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
        _colour_normalize,
    ]
)


def resize_small_colour_plus_hist(img: Image.Image) -> torch.Tensor:
    full = cast(torch.Tensor, transforms.ToTensor()(img.convert("RGB")))
    hist = torch.cat(
        [
            torch.histc(full[c], bins=16, min=0.0, max=1.0) / full[c].numel()
            for c in range(full.shape[0])
        ]
    )
    brightness = full.mean().unsqueeze(0)
    contrast = full.std().unsqueeze(0)
    resized = cast(torch.Tensor, _small_colour_resize(img))
    normalised = _colour_normalize(resized).view(-1)
    return torch.cat([normalised, hist, brightness, contrast])


def resize_small_colour_plus_hist_plus_sharp(img: Image.Image) -> torch.Tensor:
    full = cast(torch.Tensor, transforms.ToTensor()(img.convert("RGB")))
    hist = torch.cat(
        [
            torch.histc(full[c], bins=16, min=0.0, max=1.0) / full[c].numel()
            for c in range(full.shape[0])
        ]
    )
    brightness = full.mean().unsqueeze(0)
    contrast = full.std().unsqueeze(0)
    blurred = torch.nn.functional.avg_pool2d(
        full.unsqueeze(0), kernel_size=3, stride=1, padding=1
    ).squeeze(0)
    sharpness_score = torch.mean(torch.abs(full - blurred)).unsqueeze(0)
    resized = cast(torch.Tensor, _small_colour_resize(img))
    normalised = _colour_normalize(resized).view(-1)
    return torch.cat([normalised, hist, brightness, contrast, sharpness_score])


resize_med_colour = transforms.Compose(
    [
        transforms.Resize((MEDIUM_IMG_SIZE, MEDIUM_IMG_SIZE)),
        transforms.ToTensor(),
        _colour_normalize,
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
