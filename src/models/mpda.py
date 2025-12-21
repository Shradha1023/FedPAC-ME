"""
mpda.py

Multi-Perspective Data Augmentation (MPDA) for contrastive learning.
Generates multiple views of the same input:
- Perturbed Contrastive View (PCV)
- Spatially Shifted View (SSV)
- Adversarial View (GAV)
"""

import torch
import random

class MultiPerspectiveAugment:
    """
    Implements multi-perspective augmentations:
    1. Small random noise (PCV)
    2. Spatial pixel shifts (SSV)
    3. Adversarial perturbations (GAV)
    """
    def __init__(self, epsilon=0.01, sigma=0.02, shift_pixels=5):
        self.epsilon = epsilon  # For adversarial perturbation
        self.sigma = sigma
        self.shift_pixels = shift_pixels  # Pixel shift for spatial view

    def perturbed_contrastive_view(self, image):
        """
        Adds small random noise to simulate intensity variation.
        """
        noise = torch.randn_like(image) * self.sigma
        return image + noise

    def spatially_shifted_view(self, image):
        """
        Shifts the image spatially by a few pixels along height and width.
        """
        _, _, H, W = image.shape
        dx, dy = random.randint(-self.shift_pixels, self.shift_pixels), random.randint(-self.shift_pixels, self.shift_pixels)
        shifted_image = torch.roll(image, shifts=(dx, dy), dims=(2, 3))
        return shifted_image

    def adversarial_view(self, model, image):
        """
        Creates an adversarial example using FGSM (without labels).
        """
        image_adv = image.clone().detach().requires_grad_(True)
        output = model(image_adv)

        # Compute a fake loss using L2 norm instead of cross-entropy
        loss = output.norm(p=2)
        loss.backward()

        perturbation = self.epsilon * image.grad.sign()
        adversarial_image = image_adv + perturbation
        return adversarial_image.detach()

    def apply(self, model, image):
        """
        Applies all transformations and returns three augmented views.
        Returns:
            pcv, ssv, gav: tensors of augmented images
        """
        pcv = self.perturbed_contrastive_view(image)
        ssv = self.spatially_shifted_view(image)
        gav = self.adversarial_view(model, image)
        return pcv, ssv, gav
