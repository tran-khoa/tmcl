import math


def warmup_cosine(
    step: int,
    *,
    peak_lr: float,
    num_steps: int,
    warmup_steps: int,
    start_lr: float = 0.0,
    end_lr: float | None = None,
    debug: bool = False,
) -> float:
    if step < warmup_steps:
        lr = start_lr + (peak_lr - start_lr) * step / warmup_steps
    else:
        step -= warmup_steps
        num_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / num_steps))
        if end_lr is None:
            end_lr = peak_lr * 0.001
        lr = peak_lr * q + end_lr * (1 - q)
    if debug:
        print(
            f'step: {step}, peak_lr: {peak_lr}, num_steps: {num_steps}, warmup_steps: {warmup_steps}, lr: {lr}'
        )
    return lr
