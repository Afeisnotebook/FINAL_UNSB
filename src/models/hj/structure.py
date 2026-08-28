"""Source-structure tangent used as the bridge direction in HJ-PatchNCE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicReflectionPad2d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        if len(padding) != 4:
            raise ValueError("padding must be an int or a 4-tuple")
        self.padding = tuple(int(value) for value in padding)

    def forward(self, value):
        left, right, top, bottom = self.padding
        height, width = value.shape[-2:]
        if left >= width or right >= width or top >= height or bottom >= height:
            raise ValueError("reflection padding must be smaller than the input dimension")
        horizontal = [value]
        if left:
            horizontal.insert(0, value[..., 1 : left + 1].flip(-1))
        if right:
            horizontal.append(value[..., -right - 1 : -1].flip(-1))
        value = torch.cat(horizontal, dim=-1)
        vertical = [value]
        if top:
            vertical.insert(0, value[..., 1 : top + 1, :].flip(-2))
        if bottom:
            vertical.append(value[..., -bottom - 1 : -1, :].flip(-2))
        return torch.cat(vertical, dim=-2)


def _sobel_magnitude(image):
    gray = (
        image[:, 0:1] * 0.2989
        + image[:, 1:2] * 0.5870
        + image[:, 2:3] * 0.1140
    )
    kernel_x = image.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(2, 3)
    gray = DeterministicReflectionPad2d(1)(gray)
    gradient_x = F.conv2d(gray, kernel_x)
    gradient_y = F.conv2d(gray, kernel_y)
    return torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-8)


def _parse_scales(scales):
    if isinstance(scales, str):
        scales = [int(item.strip()) for item in scales.split(",") if item.strip()]
    scales = [int(s) for s in scales]
    if not scales or any(s < 1 for s in scales):
        raise ValueError("scales must contain positive integers")
    return scales


def _resize_if_needed(target, source):
    if source.shape[-2:] != target.shape[-2:]:
        source = F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return source


def edge_gradient(target, source, scales="1,2,4"):
    """Gradient of a multi-scale edge-magnitude mismatch w.r.t. the target."""
    source = _resize_if_needed(target, source)
    losses = []
    for scale in _parse_scales(scales):
        if scale == 1:
            target_scale, source_scale = target, source
        else:
            target_scale = F.avg_pool2d(target, kernel_size=scale, stride=scale)
            source_scale = F.avg_pool2d(source, kernel_size=scale, stride=scale)
        difference = _sobel_magnitude(target_scale) - _sobel_magnitude(source_scale.detach())
        losses.append(torch.sqrt(difference.square() + 1e-6).mean())
    edge_loss = torch.stack(losses).mean()
    return torch.autograd.grad(
        edge_loss, target, retain_graph=True, create_graph=False, allow_unused=False
    )[0].detach()


def ssim_gradient(target, source, scales="1,2,4"):
    """Gradient of a multi-scale structural-similarity loss w.r.t. the target."""
    source = _resize_if_needed(target, source)
    losses = []
    for scale in _parse_scales(scales):
        if scale == 1:
            target_scale, source_scale = target, source
        else:
            target_scale = F.avg_pool2d(target, kernel_size=scale, stride=scale)
            source_scale = F.avg_pool2d(source, kernel_size=scale, stride=scale)
        kernel = min(7, target_scale.shape[-2], target_scale.shape[-1])
        if kernel % 2 == 0:
            kernel -= 1
        kernel = max(kernel, 1)
        pad = kernel // 2

        def local_mean(value):
            if pad:
                value = DeterministicReflectionPad2d(pad)(value)
            return F.avg_pool2d(value, kernel, stride=1)

        source_scale = source_scale.detach()
        mean_x = local_mean(target_scale)
        mean_y = local_mean(source_scale)
        variance_x = (local_mean(target_scale.square()) - mean_x.square()).clamp_min(0.0)
        variance_y = (local_mean(source_scale.square()) - mean_y.square()).clamp_min(0.0)
        covariance = local_mean(target_scale * source_scale) - mean_x * mean_y
        c1 = (0.01 * 2.0) ** 2
        c2 = (0.03 * 2.0) ** 2
        numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
        denominator = (mean_x.square() + mean_y.square() + c1) * (
            variance_x + variance_y + c2
        )
        losses.append((1.0 - numerator / denominator.clamp_min(1e-8)).mean() * 0.5)
    ssim_loss = torch.stack(losses).mean()
    return torch.autograd.grad(
        ssim_loss, target, retain_graph=True, create_graph=False, allow_unused=False
    )[0].detach()


def source_structure_direction(target, source, direction="joint", scales="1,2,4", eps=1e-6):
    """Return a detached unit-RMS structural tangent for the projection."""
    edge = edge_gradient(target, source, scales)
    if str(direction).lower() == "edge":
        return edge
    ssim = ssim_gradient(target, source, scales)

    def unit_rms(value):
        rms = value.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        return value / rms.clamp_min(eps)

    return unit_rms((unit_rms(edge) + unit_rms(ssim)) * 0.5).detach()
