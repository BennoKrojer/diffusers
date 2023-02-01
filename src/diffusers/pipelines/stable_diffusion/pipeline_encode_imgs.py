import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import torch

import PIL
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKL, UNet2DConditionModel
from ...pipeline_utils import DiffusionPipeline
from ...schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from ...utils import deprecate, logging
# from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


class StableDiffusionImg2LatentPipeline(DiffusionPipeline):
    r"""
    Elinor is trying to return just the latent image and text

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
        )

    @torch.no_grad()
    def __call__(
        self,
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.

        Returns:
            Torch.tensor of the latent encoding of the init_image
        """

        if isinstance(init_image, PIL.Image.Image):
            init_image = preprocess(init_image)


        # encode the init image into latents and scale the latents
        latents_dtype = torch.bfloat16 # this changes based on the vae initialization
        init_image = init_image.to(device=self.device, dtype=latents_dtype)
        init_latent_dist = self.vae.encode(init_image).latent_dist
        init_latents = init_latent_dist.sample(generator=generator)
        init_latents = 0.18215 * init_latents

        return init_latents