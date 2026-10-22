from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import Dataset
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from src.dataset.style import StyleDataset
from src.dataset.utils import save_holdout
from src.model.sd import SDComponents
from src.utils.io import write_json


def get_val_prompts(
    templates: list[str],
    holdout: Dataset,
    num_holdout: int,
    out_dir: Path,
) -> list[str]:
    """
    Prompts for the samples: the templates and the captions of the held-out paintings.

    A template with {trigger} is used once per artist (trigger of the artist), the others once.
    num_holdout paintings of every artist are taken. The held-out paintings and the prompts
    are saved to out_dir for comparison with the samples.
    """
    triggers = list(dict.fromkeys(holdout['trigger']))  # unique, in the order of appearance
    prompts = []
    for template in templates:
        if '{trigger}' in template:
            prompts.extend(template.format(trigger=trigger) for trigger in triggers)
        else:
            prompts.append(template)

    counts: dict[str, int] = {}
    indices = []
    for i, slug in enumerate(holdout['artist_slug']):
        if counts.get(slug, 0) < num_holdout:
            indices.append(i)
            counts[slug] = counts.get(slug, 0) + 1
    prompts += save_holdout(holdout.select(indices), out_dir / 'holdout')

    write_json(prompts, out_dir / 'prompts.json')
    return prompts


def build_pipeline(
    sd: SDComponents, unet: UNet2DConditionModel, device: torch.device
) -> StableDiffusionPipeline:
    """Pipeline that shares the weights with the unet that is trained."""
    pipe = StableDiffusionPipeline(
        vae=sd.vae,
        text_encoder=sd.text_encoder,
        tokenizer=sd.tokenizer,
        unet=unet,
        scheduler=DPMSolverMultistepScheduler.from_config(sd.noise_scheduler.config),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.to(device)  # no dtype: the LoRA weights stay in fp32
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def generate_samples(
    pipe: StableDiffusionPipeline,
    accelerator: Accelerator,
    prompts: list[str],
    out_dir: Path,
    step: int,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> None:
    """Generate images for the fixed prompts with the fixed noise: runs are comparable by step."""
    was_training = pipe.unet.training
    pipe.unet.eval()
    step_dir = out_dir / f'step_{step:04d}'
    step_dir.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        with accelerator.autocast():
            image = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator('cpu').manual_seed(seed),  # same noise on any device
            ).images[0]
        image.save(step_dir / f'{i:02d}.png')
    pipe.unet.train(was_training)


class ValMetrics:
    """
    Cheap validation metrics: no image generation, only passes of the unet.

    The loss is measured on held-out paintings of the same artists with the fixed noise and
    timesteps, so it is comparable between the steps. It is measured with the caption of the
    painting with the trigger and without it. The adapter is initialized to the identity, so
    the losses at the creation are those of the base model: the gains are measured from them.

    - val/loss, val/loss_no_trigger: MSE of the noise (or velocity) prediction.
    - val/gain, val/gain_no_trigger: the loss of the base model minus the loss now, the higher
      the better. If the gain without the trigger is about the gain with it, the style leaks to
      the prompts without the trigger.
    - adapter/delta_norm: the distance of the trainable weights from their initialization,
      it is 0 if the adapter does not train.
    """

    def __init__(
        self,
        sd: SDComponents,
        accelerator: Accelerator,
        path: str,
        artists: list[str] | None,
        resolution: int,
        images_per_artist: int = 4,
        num_timesteps: int = 5,
        batch_size: int = 8,
        seed: int = 42,
    ) -> None:
        self.sd, self.accelerator, self.batch_size = sd, accelerator, batch_size
        device = accelerator.device
        # the same preprocessing as in the training; the prompt with the trigger is tokenized there
        data = StyleDataset(path, artists, sd.tokenizer, 'test', resolution, True, False)
        counts: dict[str, int] = {}
        indices = []
        for i, slug in enumerate(data.data['artist_slug']):
            if counts.get(slug, 0) < images_per_artist:
                indices.append(i)
                counts[slug] = counts.get(slug, 0) + 1
        rows = [data[i] for i in indices]
        ids = {
            'trigger': torch.stack([r['input_ids'] for r in rows]),
            'plain': sd.tokenizer(
                [data.data[i]['caption'] for i in indices],
                max_length=sd.tokenizer.model_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            ).input_ids,
        }
        pixels = torch.stack([r['pixel_values'] for r in rows])

        # the vae and the text encoder are frozen: encode once, only the unet runs later
        with torch.no_grad(), accelerator.autocast():
            self.latents = (
                torch.cat(
                    [
                        sd.vae.encode(
                            pixels[i : i + batch_size].to(device, sd.vae.dtype)
                        ).latent_dist.mean.float()
                        for i in range(0, len(pixels), batch_size)
                    ]
                )
                * sd.vae.config.scaling_factor
            )
            self.embeds = {
                name: torch.cat(
                    [
                        sd.text_encoder(x[i : i + batch_size].to(device))[0]
                        for i in range(0, len(x), batch_size)
                    ]
                )
                for name, x in ids.items()
            }
        num_train = sd.noise_scheduler.config.num_train_timesteps
        self.timesteps = torch.linspace(0, num_train - 1, num_timesteps + 2)[1:-1].long().tolist()
        generator = torch.Generator('cpu').manual_seed(seed)
        noise_shape = (num_timesteps, len(rows), *self.latents.shape[1:])
        self.noise = torch.randn(noise_shape, generator=generator).to(device)

        self.params = [p for p in accelerator.unwrap_model(sd.unet).parameters() if p.requires_grad]
        self.init_params = [p.detach().float().clone() for p in self.params]
        self.base = self._losses()
        accelerator.print(
            f'Validation loss of the base model: {self.base["trigger"]:.4f} with the trigger, '
            f'{self.base["plain"]:.4f} without it ({len(rows)} paintings, '
            f'timesteps {self.timesteps})'
        )

    @torch.no_grad()
    def _losses(self) -> dict[str, float]:
        scheduler = self.sd.noise_scheduler
        unet = self.accelerator.unwrap_model(self.sd.unet)
        was_training = unet.training
        unet.eval()
        device = self.accelerator.device
        losses = {}
        for name, embeds in self.embeds.items():
            total = 0.0
            for k, t in enumerate(self.timesteps):
                for i in range(0, len(self.latents), self.batch_size):
                    latents = self.latents[i : i + self.batch_size]
                    noise = self.noise[k, i : i + self.batch_size]
                    timesteps = torch.full((len(latents),), t, device=device, dtype=torch.long)
                    noisy = scheduler.add_noise(latents, noise, timesteps)
                    with self.accelerator.autocast():
                        pred = unet(noisy, timesteps, embeds[i : i + self.batch_size]).sample
                    if scheduler.config.prediction_type == 'v_prediction':
                        target = scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise
                    total += F.mse_loss(pred.float(), target.float(), reduction='sum').item()
            losses[name] = total / (len(self.timesteps) * self.noise[0].numel())
        unet.train(was_training)
        return losses

    @torch.no_grad()
    def __call__(self) -> dict[str, float]:
        """Metrics of the current state of the adapter, to log with accelerator.log."""
        now = self._losses()
        delta = sum(
            ((p.detach().float() - p0) ** 2).sum().item()
            for p, p0 in zip(self.params, self.init_params)
        )
        return {
            'val/loss': now['trigger'],
            'val/loss_no_trigger': now['plain'],
            'val/gain': self.base['trigger'] - now['trigger'],
            'val/gain_no_trigger': self.base['plain'] - now['plain'],
            'adapter/delta_norm': delta**0.5,
        }
