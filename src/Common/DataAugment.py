from src.Common.utils import *
import torch
from torchvision import transforms
import random
from PIL import Image


# -----------------------------------------------------------------------------
# apply_random_affine
# -----------------------------------------------------------------------------
def apply_random_affine(image, degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)):
    """
    Applies a random affine transformation to the image.

    Parameters:
    - image: Tensor or PIL Image to be transformed.
    - degrees: Range of degrees to select from for rotation.
    - translate: Tuple of maximum absolute fraction for horizontal and vertical translations.
    - scale: Scaling factor interval.
    """
    affine_transform = transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=-1)
    return affine_transform(image)


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


# -----------------------------------------------------------------------------
# augment_image
# -----------------------------------------------------------------------------
def augment_image(image, affine_params={}, colorize_params={}):
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
        lambda img: colorize_digit_tensor(img, **colorize_params)
    ]    

    # Randomly select augmentations (1 or more)
    chosen_augmentations = random.sample(augmentations, k=random.randint(1, len(augmentations)))

    for aug in chosen_augmentations:
        image = aug(image)  # Apply augmentation

    return image


# -----------------------------------------------------------------------------
# augment_images
# -----------------------------------------------------------------------------
def augment_images(mini_batch, affine_params={}, colorize_params={}):
    """
    Randomly applies one or more augmentation methods to the input mini-batch of tensor images with optional parameters.

    Parameters:
    - mini-batch of images: Mini-batch of tensor image of an MNIST digit, normalized to [-1, 1].
    - affine_params: Optional parameters for the affine transformation as a dict.
    - crop_params: Optional parameters for the random resized crop as a dict.
    - colorize_params: Optional parameters for the colorize digit tensor as a dict.

    Returns:
    - The mini-batch ith augmented image tensors.
    """

    for i in range(len(mini_batch)):
        # Now correctly passing the optional parameters and updating the images in the mini_batch
        mini_batch[i] = augment_image(mini_batch[i], affine_params=affine_params, colorize_params=colorize_params)
    
    return mini_batch