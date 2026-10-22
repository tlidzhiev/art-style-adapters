from collections.abc import Iterable
from pathlib import Path

from datasets import Dataset, load_from_disk
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from src.utils.io import get_root, write_json


def load_artist_split(
    path: str | Path,
    artists: Iterable[str] | None = None,
    split: str = 'train',
) -> Dataset:
    data = load_from_disk(get_root() / path)[split]
    if artists is not None:
        keep = set(artists)
        data = data.filter(lambda slug: slug in keep, input_columns='artist_slug')  # ty:ignore[unresolved-attribute]
        if missing := keep - set(data.unique('artist_slug')):
            raise ValueError(f"No paintings of {sorted(missing)} in the '{split}' split of {path}.")
    print(f'{len(data)} paintings of {sorted(artists) if artists else "all artists"} ({split})')
    return data  # ty:ignore[invalid-return-type]


def get_dataloaders(
    config: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, DataLoader]:
    dataloaders = {}
    for partition in config.dataset:
        if partition not in ['train', 'val', 'test'] or config.dataloader.get(partition) is None:
            continue

        dataset = instantiate(config.dataset[partition], tokenizer=tokenizer)
        batch_size = config.dataloader[partition].batch_size
        assert batch_size <= len(dataset), (
            f'The batch size ({batch_size}) cannot be larger than the dataset length '
            f'({len(dataset)})'
        )
        dataloaders[partition] = instantiate(config.dataloader[partition], dataset=dataset)
    return dataloaders


def save_holdout(data: Dataset, save_path: Path) -> list[str]:
    save_path.mkdir(parents=True, exist_ok=True)
    prompts = []
    for j, row in enumerate(data):
        row['image'].convert('RGB').save(save_path / f'{j:02d}.jpg', quality=95)
        prompts.append(row['prompt'])
    write_json(prompts, save_path / 'prompts.json')
    return prompts
