from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    name: str = ''
    project_name: str = 'tmcl_cil'
    group: str | None = None

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

    seed: int = 0  # also determines split...
    devel: bool = False
    deterministic: bool = True

    # kNN setup
    knn_k: int = 20
    knn_temp: float = 0.07
    eval_last_n_layers: int | None = None
    l2_normalize: bool = True

    # Checkpoing loading setup
    load_checkpoint_from: Path | None = None
    num_checkpoint_tasks: int = 1
    eval_modulation: int | None = None
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
