from src.utils import *


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



