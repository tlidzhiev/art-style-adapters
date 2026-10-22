from pathlib import Path

from diffusers import UNet2DConditionModel
from peft import PeftConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file


def save_adapter(unet: UNet2DConditionModel, peft_config: PeftConfig, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    peft_config.save_pretrained(str(save_path))
    state_dict = {k: v.contiguous() for k, v in get_peft_model_state_dict(unet).items()}
    save_file(state_dict, save_path / 'adapter_model.safetensors')


def load_adapter_weights(unet: UNet2DConditionModel, pretrained_path: str | Path) -> None:
    set_peft_model_state_dict(unet, load_file(Path(pretrained_path) / 'adapter_model.safetensors'))


def load_adapter(unet: UNet2DConditionModel, pretrained_path: str | Path) -> UNet2DConditionModel:
    pretrained_path = Path(pretrained_path)
    unet.add_adapter(PeftConfig.from_pretrained(str(pretrained_path)))  # ty:ignore[unresolved-attribute]
    load_adapter_weights(unet, pretrained_path)
    return unet
