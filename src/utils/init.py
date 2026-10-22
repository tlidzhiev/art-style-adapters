import os
import random
import secrets
import shutil
import string
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .io import get_root


def set_worker_seed(worker_id) -> None:
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for model training or inference.

    Parameters:
    seed : int
        Defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 run id for the experiment tracker.

    There are ~2.8T base-36 ids of length 8: generating 210k of them gives
    a ~1% chance of collision.

    See https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py.

    Parameters
    ----------
    length : int, optional
        Length of the id, by default 8.

    Returns
    -------
    str
        Random string of lowercase letters and digits.
    """
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def log_git_commit_and_patch(log_dir: Path) -> None:
    """
    Save the current git commit and uncommitted changes to the run directory.

    The code version of the run is restored with:
        git checkout <hash from git_commit.txt>
        git apply git_diff.patch

    New files get into the patch only if they are staged via 'git add'
    before the run.

    Parameters
    ----------
    log_dir : Path
        Run directory.
    """
    commands = {
        'git_commit.txt': ['git', 'rev-parse', 'HEAD'],
        'git_diff.patch': ['git', 'diff', 'HEAD'],
    }
    for filename, command in commands.items():
        # Bytes are kept as is, so the patch still applies to files with CRLF or non-UTF-8 text.
        result = subprocess.run(command, cwd=get_root(), capture_output=True, check=False)
        if result.returncode == 0:
            (log_dir / filename).write_bytes(result.stdout)


def saving_init(log_dir: Path, cfg: DictConfig) -> None:
    """
    Create the run directory and save the config into it.

    Parameters
    ----------
    log_dir : Path
        Run directory for everything: logs, checkpoints, config, etc.
    cfg : DictConfig
        Project config. 'writer.run_id' is filled in place.

    Raises
    ------
    ValueError
        If the run directory holds a run and override is not set.
    """
    # Hydra creates the run directory and its own log before the job starts,
    # so a previous run is recognized by its saved config.
    if (log_dir / 'config.yaml').exists():
        if not cfg.trainer.override:
            raise ValueError(
                f"Log directory '{log_dir}' exists. "
                'Change writer.run_name or set trainer.override=True.'
            )
        print(f"Overriding log directory '{log_dir}'...")
        shutil.rmtree(log_dir)

    log_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.update(cfg, 'writer.run_id', generate_id(cfg.writer.id_length))
    OmegaConf.save(cfg, log_dir / 'config.yaml', resolve=True)
    log_git_commit_and_patch(log_dir)


def setup_saving(cfg: DictConfig) -> None:
    """
    Initialize the run directory.

    The run directory is 'writer.log_dir' relative to the project root.
    The checkpoint directory 'trainer.checkpoint_dir' is created inside it.

    Parameters
    ----------
    cfg : DictConfig
        Config for the current experiment.
    """
    saving_init(get_root() / cfg.writer.log_dir, cfg)
    (get_root() / cfg.trainer.checkpoint_dir).mkdir(exist_ok=True, parents=True)


def flatten_config(cfg: DictConfig) -> dict[str, Any]:
    """
    Flat dict of the config for an experiment tracker: {'train.grad_accum': 4, ...}.

    Trackers accept only numbers, strings and booleans, other values (lists) become strings.
    """

    def flatten(node: Any, prefix: str) -> dict[str, Any]:
        if isinstance(node, dict):
            return {
                k: v
                for key, val in node.items()
                for k, v in flatten(val, f'{prefix}{key}.').items()
            }
        value = node if isinstance(node, (int, float, bool, str)) else str(node)
        return {prefix.removesuffix('.'): value}

    return flatten(OmegaConf.to_container(cfg, resolve=True), '')


def get_tracker_args(cfg: DictConfig) -> tuple[str, dict[str, dict[str, Any]]]:
    """
    Arguments of accelerator.init_trackers: one project, every run is a separate run in it.

    wandb and comet_ml: the project is 'writer.project', the run is 'writer.run_name'.
    tensorboard: the project is the directory logs/{writer.project} (the project_dir of the
    accelerator), the run is its subdirectory {writer.run_name}.

    Parameters
    ----------
    cfg : DictConfig
        Config of the experiment, 'writer.run_id' must be filled.

    Returns
    -------
    project_name : str
        For the accelerator.init_trackers.
    init_kwargs : dict[str, dict[str, Any]]
        Arguments of every tracker by its name, for the accelerator.init_trackers.
    """
    run_name = cfg.writer.run_name
    init_kwargs: dict[str, dict[str, Any]] = {
        'wandb': {'name': run_name, 'id': cfg.writer.run_id, 'resume': 'allow'}
    }
    if cfg.trainer.log_with == 'comet_ml':
        from comet_ml import ExperimentConfig

        init_kwargs['comet_ml'] = {'experiment_config': ExperimentConfig(name=run_name)}
    project_name = run_name if cfg.trainer.log_with == 'tensorboard' else cfg.writer.project
    return project_name, init_kwargs
