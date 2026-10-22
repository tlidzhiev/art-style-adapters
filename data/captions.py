import hashlib
import re

import torch
from datasets import DatasetDict
from transformers import AutoModelForImageTextToText, AutoProcessor

_LEADING = re.compile(
    r'^(the|this|in this)\s+(image|picture|painting|artwork|scene|illustration)\s*'
    r'(shows|depicts|features|portrays|presents|captures|is of|,)?\s*',
    re.IGNORECASE,
)
_LEAK = re.compile(
    r'\b(paint\w*|artwork|artist|style|brush\w*|oil|canvas|impression\w*|realis\w*|romantic\w*|'
    r'art nouveau|symbolis\w*|watercolou?r)\b',
    re.IGNORECASE,
)


def _clean_caption(text: str) -> str:
    c = text.strip().strip('"\'').strip()
    c = _LEADING.sub('', c)
    c = re.sub(r'\s+', ' ', c).strip()
    if len(c) > 1 and c[0].isupper() and not c[1].isupper():
        c = c[0].lower() + c[1:]
    return c.rstrip('. ')


def _needs_review(caption: str, artist_slug: str) -> bool:
    surname = artist_slug.split('-')[-1]
    return bool(_LEAK.search(caption)) or surname.lower() in caption.lower() or len(caption) < 10


def resolve_vlm(
    model_name: str | None,
    batch_size: int | None,
    large_model: str,
    small_model: str,
    large_vram_gb: float,
    large_batch_size: int,
    small_batch_size: int,
) -> tuple[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found. Please enable a GPU.')

    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    large = vram >= large_vram_gb
    model_name = model_name or (large_model if large else small_model)
    batch_size = batch_size or (large_batch_size if large else small_batch_size)
    print(
        f'GPU: {torch.cuda.get_device_name(0)}, {vram:.0f} GB -> {model_name}, batch {batch_size}'
    )
    return model_name, batch_size


def load_vlm(model_name: str, device: str) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_name)
    processor.tokenizer.padding_side = 'left'
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
    ).eval()
    return processor, model


def add_captions(
    ds: DatasetDict,
    device: str,
    model_name: str,
    batch_size: int,
    prompt: str,
    max_new_tokens: int,
    max_side: int,
) -> DatasetDict:
    processor, model = load_vlm(model_name, device)
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    chat_text = processor.apply_chat_template(  # ty:ignore[unresolved-attribute]
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    @torch.inference_mode()
    def caption_batch(batch: dict[str, list]) -> dict[str, list]:
        thumbs = []
        for im in batch['image']:
            im = im.convert('RGB')
            im.thumbnail((max_side, max_side))
            thumbs.append(im)
        inputs = processor(  # ty:ignore[call-non-callable]
            text=[chat_text] * len(thumbs),
            images=thumbs,
            padding=True,
            return_tensors='pt',
        ).to(device)
        out = model.generate(  # noqa: F821 # ty:ignore[unresolved-attribute]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        raw = processor.batch_decode(  # ty:ignore[unresolved-attribute]
            out[:, inputs['input_ids'].shape[1] :],
            skip_special_tokens=True,
        )
        caps = [_clean_caption(r) for r in raw]
        review = [_needs_review(c, s) for c, s in zip(caps, batch['artist_slug'])]
        return {'caption': caps, 'caption_raw': raw, 'needs_review': review}

    prompt_hash = hashlib.md5(f'{prompt}|{max_new_tokens}|{max_side}'.encode()).hexdigest()[:8]
    model_tag = model_name.replace('/', '_')
    result = DatasetDict(
        {
            name: split.map(
                caption_batch,
                batched=True,
                batch_size=batch_size,
                desc=f'captioning {name}',
                new_fingerprint=f'cap-{model_tag}-{prompt_hash}-{name}-{len(split)}',
            )
            for name, split in ds.items()
        }
    )
    del model
    torch.cuda.empty_cache()
    return result
