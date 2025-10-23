from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    name: str = ''
    project_name: str = 'tmcl_cil'
    group: str | None = None

    num_epochs: int = 100
    batch_size: int = 128
    precision: str = 'bf16-mixed'

    accelerator: str = 'gpu'

    base_work_path: Path = Path('/p/scratch/jinm60/tran4/out_llfs/')
    data_path: Path = Path('/p/scratch/jinm60/tran4/data')

    eval_dataset: str = 'cifar100'
    continual_setup: str | None = None
    labeled_frac: float = 1.0
    timm_model: str = 'vit_tiny_12heads'
    torch_compile: bool = False

    resume: bool = False
    resume_id: str | None = None
    seed: int = 0  # also determines split...
    devel: bool = False
    deterministic: bool = True

    checkpoint_interval: int = 1  # in epochs
    log_interval: int = 50  # in steps
    eval_interval: int = 1

    # Optimizer setup
    optim_algo: Literal['sgd'] = 'sgd'
    lr: float = 0.1
    weight_decay: float = 0.0
    optim_momentum: float = 0.0

    lr_scheduler: Literal['step', 'cosine', 'none'] = 'step'
    lr_decay_steps: tuple[int, ...] = (60, 80)
    lr_decay_gamma: float = 0.1

    l2_normalize: bool = False
    train_augmentations: bool = True
    eval_last_n_layers: int | None = None

    # Checkpoing loading setup
    load_checkpoint_from: Path | None = None
    load_direct_r18: bool = False
    num_checkpoint_tasks: int = 101
    eval_modulation: int | None = 100
    use_bias_modulations: bool = True

    # Multi-GPU settings
    sync_batchnorm: bool = False
    gpus_per_node: int = 1
    n_nodes: int = 1
    n_data_workers: int = 8

    @property
    def device(self):
        return 'cuda' if self.accelerator == 'gpu' else 'cpu'

    @property
    def n_devices(self) -> int:
        return self.n_nodes * self.gpus_per_node

    @property
    def batch_size_per_gpu(self) -> int:
        return self.batch_size // self.n_devices

    @property
    def work_path(self) -> Path:
        return self.base_work_path / self.name

    def get_last_checkpoint(self, force: bool = False) -> Path | None:
        path = self.work_path / 'checkpoints' / 'last.ckpt'
        if not path.exists():
            if force:
                raise RuntimeError(f'No checkpoint found at {path}')
            return None
        return path
