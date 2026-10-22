import torch
from torchvision.transforms import InterpolationMode, v2
from transformers import PreTrainedTokenizerBase

from .utils import load_artist_split


class StyleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        artists: list[str] | None,
        tokenizer: PreTrainedTokenizerBase,
        split: str = 'train',
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = False,
    ) -> None:
        self.data = load_artist_split(path, artists, split)
        self.tokenizer = tokenizer
        crop = v2.CenterCrop(resolution) if center_crop else v2.RandomCrop(resolution)
        self.transform = v2.Compose(
            [
                v2.Resize(resolution, interpolation=InterpolationMode.LANCZOS),
                crop,
                *([v2.RandomHorizontalFlip()] if random_flip else []),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5], [0.5]),
            ]
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.data[index]
        input_ids = self.tokenizer(
            row['prompt'],
            max_length=self.tokenizer.model_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        ).input_ids[0]
        pixel_values = self.transform(row['image'].convert('RGB'))
        return {'pixel_values': pixel_values, 'input_ids': input_ids}

    def __len__(self) -> int:
        return len(self.data)
