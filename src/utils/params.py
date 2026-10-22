from torch import nn


def get_parameter_summary(model: nn.Module) -> str:
    """Return formatted model parameter counts."""

    def format_number(num: int) -> str:
        if abs(num) >= 1_000_000:
            return f'{num / 1_000_000:.2f}M'
        elif abs(num) >= 1_000:
            return f'{num / 1_000:.2f}K'
        else:
            return str(num)

    all_parameters = sum([p.numel() for p in model.parameters()])
    trainable_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])

    return (
        f'All parameters: {format_number(all_parameters)}\n'
        f'Trainable parameters: {format_number(trainable_parameters)}'
    )
