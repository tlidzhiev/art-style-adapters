import hydra
from accelerate import Accelerator
from diffusers.training_utils import set_seed
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.dataset.utils import get_dataloaders, load_artist_split
from src.model.adapter import save_adapter
from src.model.sd import attach_adapter, load_sd
from src.training.eval import ValMetrics, build_pipeline, generate_samples, get_val_prompts
from src.training.train import get_mixed_precision, get_steps_per_epoch, train
from src.utils.init import flatten_config, get_tracker_args, setup_saving
from src.utils.io import get_root
from src.utils.params import get_parameter_summary


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    log_dir = get_root() / cfg.writer.log_dir
    ckpt_dir = get_root() / cfg.trainer.checkpoint_dir
    eval_dir = log_dir / 'eval'
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.train.grad_accum,
        mixed_precision=get_mixed_precision(cfg.train.mixed_precision),
        log_with=cfg.trainer.log_with,
        project_dir=log_dir.parent,  # tensorboard: project_dir/{run_name}
    )
    device = accelerator.device

    # run directory, config and git info: logs/art-style-adapters/{run_name}
    if accelerator.is_main_process:
        setup_saving(cfg)
    accelerator.wait_for_everyone()
    project_name, init_kwargs = get_tracker_args(cfg)
    accelerator.init_trackers(
        project_name=project_name, config=flatten_config(cfg), init_kwargs=init_kwargs
    )
    accelerator.print(f'device={device}, mixed_precision={accelerator.mixed_precision}')

    # model: frozen SD + trainable adapter on the UNet
    sd = load_sd(cfg.model.name, device, accelerator.mixed_precision)
    peft_config = instantiate(cfg.adapter, _convert_='all')
    attach_adapter(sd.unet, peft_config, cfg.train.gradient_checkpointing)
    accelerator.print(get_parameter_summary(sd.unet))

    params = [p for p in sd.unet.parameters() if p.requires_grad]
    optimizer = instantiate(cfg.optimizer, params=params)
    ds_cfg = cfg.dataset.train
    train_loader = get_dataloaders(cfg, sd.tokenizer)['train']
    steps_per_epoch = get_steps_per_epoch(
        train_loader, cfg.train.grad_accum, accelerator.num_processes
    )
    num_steps = cfg.train.num_epochs * steps_per_epoch
    accelerator.print(f'{steps_per_epoch} steps per epoch, {num_steps} steps')
    lr_scheduler = instantiate(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps * accelerator.num_processes,
        num_training_steps=num_steps * accelerator.num_processes,
    )

    sd.unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        sd.unet, optimizer, train_loader, lr_scheduler
    )

    # samples are generated on the main process only
    val = cfg.validation
    pipe, val_prompts, val_metrics = None, [], None
    if accelerator.is_main_process:
        holdout = load_artist_split(ds_cfg.path, ds_cfg.artists, 'test')
        pipe = build_pipeline(sd, accelerator.unwrap_model(sd.unet), device)
        val_metrics = ValMetrics(
            sd=sd,
            accelerator=accelerator,
            path=ds_cfg.path,
            artists=ds_cfg.artists,
            resolution=ds_cfg.resolution,
            images_per_artist=val.loss_images,
            num_timesteps=val.loss_timesteps,
            seed=cfg.seed,
        )
        val_prompts = get_val_prompts(
            templates=list(val.prompts),
            holdout=holdout,
            num_holdout=val.num_holdout,
            out_dir=eval_dir,
        )

    def eval_fn(step: int) -> None:
        if val_metrics is not None:
            metrics = val_metrics()
            accelerator.print(
                f'step {step}: val_loss={metrics["val/loss"]:.4f} '
                f'(gain {metrics["val/gain"]:+.4f}), without trigger '
                f'{metrics["val/loss_no_trigger"]:.4f} (gain {metrics["val/gain_no_trigger"]:+.4f}), '
                f'adapter delta {metrics["adapter/delta_norm"]:.3f}'
            )
            accelerator.log(metrics, step=step)
        if pipe is not None:
            generate_samples(
                pipe=pipe,
                accelerator=accelerator,
                prompts=val_prompts,
                out_dir=eval_dir,
                step=step,
                num_inference_steps=val.steps,
                guidance_scale=val.guidance_scale,
                seed=cfg.seed,
            )

    def save_fn(step: int) -> None:
        if accelerator.is_main_process:
            path = ckpt_dir / f'checkpoint-{step:04d}'
            save_adapter(accelerator.unwrap_model(sd.unet), peft_config, path)
            accelerator.print(f'Adapter saved to {path}')

    # the adapter is zero-initialized (LoRA B matrices), so step 0 is the base model
    eval_fn(0)
    accelerator.wait_for_everyone()

    train(
        accelerator=accelerator,
        sd=sd,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        num_epochs=cfg.train.num_epochs,
        max_grad_norm=cfg.train.max_grad_norm,
        log_step=cfg.train.log_step,
        eval_every=val.every,
        eval_fn=eval_fn,
        save_every=cfg.trainer.save_every,
        save_fn=save_fn,
    )
    accelerator.print(f'Done: adapters in {ckpt_dir}, samples in {eval_dir}')
    accelerator.end_training()


if __name__ == '__main__':
    main()
