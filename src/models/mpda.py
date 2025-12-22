"""
mpda.py

Multi-Perspective Data Augmentation (MPDA) for FedPAC-ME.
"""

import torch
import torch.nn.functional as F
import random


class MultiPerspectiveAugment:
    def __init__(self, epsilon=0.01, sigma=0.02, shift_pixels=5):
        self.epsilon = epsilon
        self.sigma = sigma
        self.shift_pixels = shift_pixels

    def perturbed_contrastive_view(self, image):
        noise = torch.randn_like(image) * self.sigma
        return image + noise

    def spatially_shifted_view(self, image):
        _, _, H, W = image.shape
        dx = random.randint(-self.shift_pixels, self.shift_pixels)
        dy = random.randint(-self.shift_pixels, self.shift_pixels)

        shifted = torch.roll(image, shifts=(dx, dy), dims=(2, 3))
        return shifted

    def adversarial_view(self, model, image):
        """
        FGSM-style adversarial view using feature-level loss.
        """
        image_adv = image.clone().detach().requires_grad_(True)

        # Forward pass
        features = model(image_adv)

        # Feature dispersion loss (contrastive-friendly)
        loss = -torch.mean(F.normalize(features, dim=1))

        # Backprop
        loss.backward()

        # FGSM perturbation
        perturbation = self.epsilon * image_adv.grad.sign()
        adv_image = image_adv + perturbation

        # Clamp to valid range
        adv_image = torch.clamp(adv_image, 0.0, 1.0)

        return adv_image.detach()

    def apply(self, model, image):
        pcv = self.perturbed_contrastive_view(image)
        ssv = self.spatially_shifted_view(image)
        gav = self.adversarial_view(model, image)
        return pcv, ssv, gav
