from dataclasses import dataclass

import torch
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.training_utils import cast_training_params
from peft import PeftConfig
from transformers import CLIPTextModel, CLIPTokenizer

DTYPES = {
    'no': torch.float32,  # the name of the accelerator's mixed precision
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


@dataclass
class SDComponents:
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    vae: AutoencoderKL
    unet: UNet2DConditionModel
    noise_scheduler: DDPMScheduler


def load_sd(model_name: str, device: torch.device | str, dtype: str = 'fp32') -> SDComponents:
    if dtype not in DTYPES:
        raise ValueError(f"Unknown dtype '{dtype}', choose one of {list(DTYPES)}.")
    _dtype = DTYPES[dtype]
    sd = SDComponents(
        tokenizer=CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer'),
        text_encoder=CLIPTextModel.from_pretrained(
            model_name, subfolder='text_encoder', dtype=_dtype
        ),
        vae=AutoencoderKL.from_pretrained(model_name, subfolder='vae', dtype=_dtype),
        unet=UNet2DConditionModel.from_pretrained(model_name, subfolder='unet', dtype=_dtype),
        noise_scheduler=DDPMScheduler.from_pretrained(model_name, subfolder='scheduler'),  # ty:ignore[ invalid-argument-type]
    )
    for m in (sd.text_encoder, sd.vae, sd.unet):
        m.requires_grad_(False)  # ty:ignore[unresolved-attribute]
        m.to(device)  # ty:ignore[invalid-argument-type,unresolved-attribute]
    return sd


def attach_adapter(
    unet: UNet2DConditionModel,
    peft_config: PeftConfig,
    gradient_checkpointing: bool = True,
) -> None:
    unet.add_adapter(peft_config)  # ty:ignore[unresolved-attribute]
    cast_training_params(unet, dtype=torch.float32)  # ty:ignore[invalid-argument-type]
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()  # ty:ignore[unresolved-attribute]
