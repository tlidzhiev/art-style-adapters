import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.model.sd import SDComponents


def get_mixed_precision(mixed_precision: str | None = None) -> str:
    """Mixed precision for the accelerator; None -> bf16 / fp16 on CUDA, no otherwise."""
    if mixed_precision is not None:
        return mixed_precision
    if not torch.cuda.is_available():
        return 'no'
    return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'


def get_steps_per_epoch(train_loader: DataLoader, grad_accum: int, num_processes: int = 1) -> int:
    """
    Optimizer steps in one epoch.

    The last incomplete accumulation cycle of an epoch is also a step (accelerate syncs the
    gradients at the end of the dataloader). Pass num_processes for a loader that is not prepared.
    """
    return math.ceil(math.ceil(len(train_loader) / num_processes) / grad_accum)  # ty:ignore[invalid-argument-type]


def train(
    accelerator: Accelerator,
    sd: SDComponents,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    train_loader: DataLoader,
    num_epochs: int,
    max_grad_norm: float = 1.0,
    log_step: int = 10,
    eval_every: int = 100,
    eval_fn: Callable[[int], None] | None = None,
    save_every: int = 500,
    save_fn: Callable[[int], None] | None = None,
) -> None:
    """
    Train the adapter of sd.unet for num_epochs epochs over the train loader.

    Parameters
    ----------
    sd : SDComponents
        Frozen vae, text encoder and noise scheduler, and the unet with the adapter: it must be
        the one after accelerator.prepare (sd.unet = accelerator.prepare(sd.unet)).
    train_loader : DataLoader
        Train loader after accelerator.prepare.
    eval_fn : Callable[[int], None], optional
        Called with the step number every eval_every steps and at the last step, on all processes.
    save_fn : Callable[[int], None], optional
        The same for saving the adapter: every save_every steps and at the last step.
    """
    device = accelerator.device
    unet = sd.unet
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    steps_per_epoch = get_steps_per_epoch(train_loader, accelerator.gradient_accumulation_steps)
    num_steps = num_epochs * steps_per_epoch
    step, loss_sum, grad_norm = 0, 0.0, None
    pbar = tqdm(
        total=num_steps,
        desc='train',
        disable=not accelerator.is_local_main_process,
    )
    for _ in range(num_epochs):
        for batch in train_loader:
            with accelerator.accumulate(unet):
                pixel_values = batch['pixel_values'].to(device, dtype=sd.vae.dtype)
                input_ids = batch['input_ids'].to(device)

                with torch.no_grad():
                    latents = sd.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * sd.vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, sd.noise_scheduler.config.num_train_timesteps, (latents.shape[0],)
                    ).to(device)
                    noisy_latents = sd.noise_scheduler.add_noise(latents, noise, timesteps)
                    with accelerator.autocast():
                        text_embeds = sd.text_encoder(input_ids)[0]

                if sd.noise_scheduler.config.prediction_type == 'v_prediction':
                    target = sd.noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                pred = unet(noisy_latents, timesteps, text_embeds).sample
                loss = F.mse_loss(pred.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.item() / accelerator.gradient_accumulation_steps
            if accelerator.sync_gradients:  # a full accumulation cycle is done
                step += 1
                pbar.update(1)
                if step % log_step == 0:
                    lr = lr_scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=loss_sum / log_step, lr=lr)
                    accelerator.print(f'step {step}: loss={loss_sum / log_step:.4f}, lr={lr:.2e}')
                    metrics = {'train/loss': loss_sum / log_step, 'train/lr': lr}
                    if grad_norm is not None:  # the norm before the clipping
                        metrics['train/grad_norm'] = float(grad_norm)
                    accelerator.log(metrics, step=step)
                    loss_sum = 0.0
                last = step == num_steps
                if eval_fn is not None and (step % eval_every == 0 or last):
                    eval_fn(step)
                    accelerator.wait_for_everyone()
                if save_fn is not None and (step % save_every == 0 or last):
                    save_fn(step)
                    accelerator.wait_for_everyone()
    pbar.close()
