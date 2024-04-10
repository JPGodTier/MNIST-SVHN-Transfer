from src.utils import *
import torch
from torchvision import transforms
import random
from PIL import Image


# -----------------------------------------------------------------------------
# colorize_digit_tensor
# -----------------------------------------------------------------------------
def apply_random_affine(image, degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)):
    """
    Applies a random affine transformation to the image.

    Parameters:
    - image: PIL Image to be transformed.
    - degrees: Range of degrees to select from for rotation.
    - translate: Tuple of maximum absolute fraction for horizontal and vertical translations.
    - scale: Scaling factor interval.
    """
    affine_transform = transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale)
    return affine_transform(image)


# -----------------------------------------------------------------------------
# colorize_digit_tensor
# -----------------------------------------------------------------------------
def apply_random_resized_crop(image, size=(32, 32), scale=(0.8, 1.0)):
    """
    Crop the given PIL Image to random size and aspect ratio.

    Parameters:
    - image: PIL Image to be cropped and resized.
    - size: Expected output size of each edge.
    - scale: Range of size of the origin size cropped.
    """
    crop_transform = transforms.RandomResizedCrop(size=size, scale=scale)
    return crop_transform(image)


# -----------------------------------------------------------------------------
# colorize_digit_tensor
# -----------------------------------------------------------------------------
def colorize_digit_tensor(image, color_variability=0.2):
    """
    Colorizes a tensor representing an MNIST digit image, introducing variability in color.
    Can selectively colorize the digit or the background.
    Assumes the image tensor is normalized to [-1, 1] and in CxHxW format.

    Parameters:
    - image: Tensor image of an MNIST digit, normalized to [-1, 1].
    - color_variability: Variability in the color intensity to apply.

    Returns:
    - A colorized version of the input image tensor.
    """
    # Copy original to avoid unwanted modification
    augmented_image = image.clone()

    # Random Color Genration
    base_color = torch.rand(3, device=augmented_image.device) * color_variability + (1 - color_variability)
    base_color = base_color.view(3, 1, 1)

    # Digit mask
    digit_mask = augmented_image.mean(dim=0, keepdim=True) > 0

    # Applying the base color to the digit
    augmented_image = torch.where(digit_mask, base_color, augmented_image)

    return augmented_image


def augment_image(image, affine_params={}, crop_params={}, colorize_params={}):
    """
    Randomly applies one or more augmentation methods to the input tensor image with optional parameters.

    Parameters:
    - image: Tensor image of an MNIST digit, normalized to [-1, 1].
    - affine_params: Optional parameters for the affine transformation as a dict.
    - crop_params: Optional parameters for the random resized crop as a dict.
    - colorize_params: Optional parameters for the colorize digit tensor as a dict.

    Returns:
    - The augmented image tensor.
    """
    # List of augmentation functions to choose from with their optional parameters
    augmentations = [
        lambda img: apply_random_affine(img, **affine_params),
        lambda img: apply_random_resized_crop(img, **crop_params),
        lambda img: colorize_digit_tensor(img, **colorize_params)
    ]

    # Convert tensor image to PIL Image for apply_random_affine and apply_random_resized_crop
    img_pil = transforms.ToPILImage()(image)

    # Randomly select augmentations (1 or more)
    chosen_augmentations = random.sample(augmentations, k=random.randint(1, len(augmentations)))

    for aug in chosen_augmentations:
        if aug == colorize_digit_tensor:
            image = aug(s)  # Apply augmentation for tensor
        else:
            img_pil = aug(img_pil)  # Apply augmentation for PIL image

    # If the last augmentation was on PIL, convert back to tensor
    if isinstance(img_pil, Image.Image):
        image = transforms.ToTensor()(img_pil)

    return image
