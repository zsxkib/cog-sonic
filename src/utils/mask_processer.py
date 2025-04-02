
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import CONFIG_NAME, PIL_INTERPOLATION, deprecate
from diffusers.image_processor import VaeImageProcessor

class IPAdapterMaskProcessor(VaeImageProcessor):
    """
    Image processor for IP Adapter image masks.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `False`):
            Whether to normalize the image to [-1,1].
        do_binarize (`bool`, *optional*, defaults to `True`):
            Whether to binarize the image to 0/1.
        do_convert_grayscale (`bool`, *optional*, defaults to be `True`):
            Whether to convert the images to grayscale format.

    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = False,
        do_binarize: bool = True,
        do_convert_grayscale: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )

    @staticmethod
    def downsample(mask: torch.Tensor, batch_size: int, num_queries: int, value_embed_dim: int):
        """
        Downsamples the provided mask tensor to match the expected dimensions for scaled dot-product attention. If the
        aspect ratio of the mask does not match the aspect ratio of the output image, a warning is issued.

        Args:
            mask (`torch.Tensor`):
                The input mask tensor generated with `IPAdapterMaskProcessor.preprocess()`.
            batch_size (`int`):
                The batch size.
            num_queries (`int`):
                The number of queries.
            value_embed_dim (`int`):
                The dimensionality of the value embeddings.

        Returns:
            `torch.Tensor`:
                The downsampled mask tensor.

        """
        o_h = mask.shape[1]
        o_w = mask.shape[2]
        ratio = o_w / o_h
        mask_h = int(torch.sqrt(torch.FloatTensor([num_queries / ratio]))[0])
        mask_h = int(mask_h) + int((num_queries % int(mask_h)) != 0)
        mask_w = num_queries // mask_h

        mask_downsample = F.interpolate(mask.unsqueeze(0), size=(mask_h, mask_w), mode="bicubic").squeeze(0)

        # Repeat batch_size times
        if mask_downsample.shape[0] < batch_size:
            mask_downsample = mask_downsample.repeat(batch_size, 1, 1)

        mask_downsample = mask_downsample.view(mask_downsample.shape[0], -1)

        downsampled_area = mask_h * mask_w
        # If the output image and the mask do not have the same aspect ratio, tensor shapes will not match
        # Pad tensor if downsampled_mask.shape[1] is smaller than num_queries
        if downsampled_area < num_queries:
            warnings.warn(
                "The aspect ratio of the mask does not match the aspect ratio of the output image. "
                "Please update your masks or adjust the output size for optimal performance.",
                UserWarning,
            )
            mask_downsample = F.pad(mask_downsample, (0, num_queries - mask_downsample.shape[1]), value=0.0)
        # Discard last embeddings if downsampled_mask.shape[1] is bigger than num_queries
        if downsampled_area > num_queries:
            warnings.warn(
                "The aspect ratio of the mask does not match the aspect ratio of the output image. "
                "Please update your masks or adjust the output size for optimal performance.",
                UserWarning,
            )
            mask_downsample = mask_downsample[:, :num_queries]

        # Repeat last dimension to match SDPA output shape
        mask_downsample = mask_downsample.view(mask_downsample.shape[0], mask_downsample.shape[1], 1).repeat(
            1, 1, value_embed_dim
        )

        return mask_downsample