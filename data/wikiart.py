import hashlib
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

import hydra
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from omegaconf import DictConfig

from .captions import add_captions, resolve_vlm


def get_data_dir(
    root: str | Path,
    artists: Mapping[str, str],
    train_size: int,
    test_size: int,
    seed: int,
    target: str,
) -> Path:
    """Directory of the captioned dataset; the name encodes everything that defines its content."""
    # triggers are baked into the prompts, so they are part of the dataset identity
    key = ','.join(f'{slug}:{trigger}' for slug, trigger in sorted(artists.items()))
    artists_hash = hashlib.md5(key.encode()).hexdigest()[:6]
    name = f'wikiart_{artists_hash}_train{train_size}_test{test_size}_seed{seed}_{target}'
    return Path(root) / name


def build_subset(
    ds: Dataset,
    artists: Iterable[str],
    train_size: int = 50,
    test_size: int = 10,
    seed: int = 42,
) -> DatasetDict:
    names = ds.features['artist'].names
    artist_col = np.array(list(ds['artist']))  # labels only, images are not decoded

    rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng([seed, 1])
    train_indices, test_indices = [], []
    for slug in artists:
        if slug not in names:
            raise ValueError(f"Artist '{slug}' not found in the dataset.")
        idx = np.flatnonzero(artist_col == names.index(slug))
        k = min(train_size, len(idx))
        if k < train_size:
            print(f'[warn] {slug} has only {len(idx)} paintings, taking all of them for train')
        train_idx = rng.choice(idx, size=k, replace=False)
        rest = np.setdiff1d(idx, train_idx)
        k_test = min(test_size, len(rest))
        if k_test < test_size:
            print(f'[warn] {slug}: only {k_test} paintings left for test instead of {test_size}')
        test_idx = test_rng.choice(rest, size=k_test, replace=False)
        train_indices.extend(train_idx.tolist())
        test_indices.extend(test_idx.tolist())

    feats = ds.features

    def labels_to_str(batch: dict[str, list]) -> dict[str, list[str]]:
        out = {'artist_slug': [feats['artist'].int2str(x) for x in batch['artist']]}
        for col in ('genre', 'style'):
            if col in batch:
                out[f'{col}_name'] = [feats[col].int2str(x) for x in batch[col]]
        return out

    subset = DatasetDict(
        {
            'train': ds.select(sorted(train_indices)),
            'test': ds.select(sorted(test_indices)),
        }
    )
    return subset.map(labels_to_str, batched=True, desc='labels -> str')


def add_prompts(sub: DatasetDict, artists: Mapping[str, str]) -> DatasetDict:
    def fn(slugs: list[str], captions: list[str]) -> dict[str, list[str]]:
        triggers = [artists[s] for s in slugs]
        return {'trigger': triggers, 'prompt': [f'{t}, {c}' for t, c in zip(triggers, captions)]}

    return sub.map(
        fn,
        batched=True,
        input_columns=['artist_slug', 'caption'],
        desc='trigger -> prompt',
    )


def build_dataset(
    data_dir: str | Path,
    artists: Mapping[str, str],
    train_size: int,
    test_size: int,
    seed: int,
    model_name: str,
    batch_size: int,
    prompt: str,
    max_new_tokens: int,
    max_side: int,
    device: str = 'cuda',
) -> DatasetDict:
    data_dir = Path(data_dir)
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset('huggan/wikiart', split='train')
    sub = build_subset(ds, artists, train_size, test_size, seed)
    sub = add_captions(sub, device, model_name, batch_size, prompt, max_new_tokens, max_side)
    sub = add_prompts(sub, artists)

    tmp = data_dir.with_name(data_dir.name + '_tmp')
    if tmp.is_dir():
        shutil.rmtree(tmp)
    sub.save_to_disk(tmp)
    if data_dir.is_dir():
        shutil.rmtree(data_dir)
    tmp.rename(data_dir)

    data = load_from_disk(data_dir)
    assert isinstance(data, DatasetDict)
    for name, split in data.items():
        split.remove_columns('image').to_csv(data_dir / f'captions_{name}.csv')
    num_rev = sum(sum(split['needs_review']) for split in data.values())
    sizes = ', '.join(f'{len(split)} {name}' for name, split in data.items())
    print(f'Saved {sizes} examples to {data_dir}; flagged for review: {num_rev}')
    return data


@hydra.main(version_base='1.3', config_path='configs', config_name='wikiart')
def main(cfg: DictConfig) -> None:
    artists = {str(slug): str(trigger) for slug, trigger in cfg.artists.items()}
    sub, cap = cfg.subset, cfg.caption
    data_dir = get_data_dir(
        root=cfg.root,
        artists=artists,
        train_size=sub.train_size,
        test_size=sub.test_size,
        seed=sub.seed,
        target=cap.target,
    )

    if data_dir.is_dir() and not cap.recaption:
        print(f'Loading the ready dataset from Drive: {data_dir}')
        data = load_from_disk(data_dir)
        assert isinstance(data, DatasetDict)
    else:
        model_name, batch_size = resolve_vlm(
            model_name=cap.model,
            batch_size=cap.batch_size,
            large_model=cap.large_model,
            small_model=cap.small_model,
            large_vram_gb=cap.large_vram_gb,
            large_batch_size=cap.large_batch_size,
            small_batch_size=cap.small_batch_size,
        )
        data = build_dataset(
            data_dir=data_dir,
            artists=artists,
            train_size=sub.train_size,
            test_size=sub.test_size,
            seed=sub.seed,
            model_name=model_name,
            batch_size=batch_size,
            prompt=cap.prompts[cap.target],
            max_new_tokens=cap.max_new_tokens[cap.target],
            max_side=cap.max_side,
        )

    for row in data['train'].select(range(min(5, len(data['train'])))):
        print(f'[{row["artist_slug"]}] {row["prompt"]}')


if __name__ == '__main__':
    main()
