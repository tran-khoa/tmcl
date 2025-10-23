from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

from tmcl.phase import Phase


@dataclass
class PastDistillationConfig:
    algo: Literal['pnr', 'cassle',  'none'] = 'none'
    distill_ssl: Literal['barlow', 'ghosh'] = 'barlow'
    distill_weight: float = 1.0
    barlow_lambda: float = 0.0051
    # https://github.com/DonkeyShot21/cassle/blob/main/cassle/distillers/decorrelative.py
    barlow_scale_loss: float = 0.1
    head_hidden_dim: int = 2048
    head_output_dim: int = 2048
    use_predictor: bool = True
    pnr_neg_lamb: float = 1.0
    pnr_weight: float = 1.0


@dataclass
class OptimConfig:
    optim_algo: str = 'adamw'
    tl_optim_algo: str | None = None
    cons_lr: float = 1.5e-3
    cons_min_lr: float = 1.5e-6
    cons_start_lr: float = 0.0

    tl_lr: float = 1.5e-2
    tl_min_lr: float = 1.5e-5
    tl_start_lr: float = 0.0
    joint_lr: float = 1.5e-3
    joint_min_lr: float = 1.5e-6

    backbone_weight_decay: tuple[float, float] = (0.04, 0.4)
    warmup_epochs: int = 10
    adam_betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float | None = 3.0
    sgd_momentum: float = 0.9
    lars_eta: float = 0.02
    lars_clip: bool = False
    lars_exclude_bias_n_norm: bool = True


@dataclass
class ClassLearnerConfig:
    method: Literal['sup', 'contrastive_opl'] = 'sup'
    layerwise_weight_decay: tuple[float, float] = (0.4, 0.04)
    augment: Literal['sup', 'ssl'] = 'sup'
    labeled_setup: Literal['onevsall', 'allvsall'] = 'onevsall'
    during_pretraining: bool = True

    # Contrastive-OPL
    opl_neg_weight: float = 1.0
    opl_square_loss: bool = False

    use_task_modulations: bool = True


@dataclass
class SupHeadConfig:
    enable_sup_head: bool = False
    sup_head_weight: float = 1.0
    head_hidden_dim: int = 2048


@dataclass
class SSLConfig:
    ssl_algo: str = 'mvbarlow'
    ssl_head: str = 'barlow'
    head_hidden_dim: int = 2048
    head_output_dim: int = 2048
    normalize_eps: float = 1e-5
    barlow_lambda: float = 0.0051
    barlow_scale_loss: float = 0.024
    ssl_weight: float = 1.0
    disable_ssl: bool = False
    sup_weight: float = None
    reset_projector: bool = False


@dataclass
class TMCLConfig:
    tmcl_algo: str = 'mvbarlow'
    tmcl_weight: float = 1.0
    barlow_lambda: float = 5e-3
    barlow_scale_loss: float = 0.024
    head_layers: int = 3
    head_hidden_dim: int = 2048
    head_output_dim: int = 2048
    head_batch_norm: bool = True
    unmod_first_view: bool = True
    use_predictor: bool = True
    pred_hidden_dim: int = 2048
    stop_grad_tms: bool = True

    disable_tmcl: bool = False

    pretraining: bool = False
    pretraining_after_n_epochs: int = -1
    reset_projector: bool = True

    # ablations
    random_modulations: bool = False
    single_augment: bool = False


@dataclass
class EvalConfig:
    disable_eval: bool = False
    eval_before_training: bool = False
    eval_interval: int = 1  # in epochs
    knn_per_session: bool = True
    eval_valid: bool = False
    linear_eval: bool = False
    linear_last_n_layers: int = 4
    linear_resume: bool = False


@dataclass
class SupConConfig:
    enable_supcon: bool = False
    supcon_weight: float = 1.0
    head_hidden_dim: int = 2048
    head_output_dim: int = 128
    temperature: float = 0.1


@dataclass
class Config:
    name: str = ''

    project_name: str = 'tmcl'
    group: str | None = None

    dataset: str = 'imagenet100'
    setup: str = 's5'

    batch_size: int = 512
    precision: str = 'bf16-mixed'

    accelerator: Literal['gpu', 'cpu'] = 'gpu'
    gpus_per_task: int = 1
    n_nodes: int = 1
    n_data_workers: int = 8

    base_work_path: Path = Path('/p/scratch/jinm60/tran4/out_llfs/')
    data_path: Path = Path('/p/project1/neuroml/tran4/data')
    unsup_replacement: bool = True
    sup_replacement: bool = True
    last_batch: Literal['keep', 'drop', 'pad'] = 'keep'

    timm_model: str = 'vit_tiny_12heads'
    torch_compile: bool = False

    reshuffle_class_order: bool = False
    labeled_frac: float = 1.0

    resume: bool = False
    resume_id: str | None = None
    resume_from: Path | None = None
    seed: int = 0
    devel: bool = False
    deterministic: bool = False

    checkpoint_interval: int = 1  # in epochs
    log_interval: int = 50  # in steps

    load_checkpoint_from: Path | None = None
    load_checkpoint_heads: bool = True
    ignore_checkpoint_tmcl_heads: bool = False

    pretrain_epochs: tuple[int, int] = (100, 100)
    incremental_epochs: tuple[int, int] = (100, 100)
    joint_epochs: int = 200
    end_after_session: int | None = None

    cons_num_views: int = 4
    bias_modulations: bool = True
    resume_strict: bool = True

    class_learner: ClassLearnerConfig = field(default_factory=ClassLearnerConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    tmcl: TMCLConfig = field(default_factory=TMCLConfig)
    supcon: SupConConfig = field(default_factory=SupConConfig)
    suphead: SupHeadConfig = field(default_factory=SupHeadConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    distill: PastDistillationConfig = field(default_factory=PastDistillationConfig)

    @property
    def is_imagenet(self) -> bool:
        # Check if the setup is for ImageNet
        # e.g. turns on DALI data loading, disables validation, copies data to local storage, ...
        return self.setup.startswith('imagenet')

    @property
    def device(self):
        return 'cuda' if self.accelerator == 'gpu' else 'cpu'

    @property
    def n_devices(self) -> int:
        return self.n_nodes * self.gpus_per_task

    @property
    def batch_size_per_gpu(self) -> int:
        return self.batch_size // self.n_devices

    @property
    def work_path(self) -> Path:
        return self.base_work_path / self.name

    @cached_property
    def session_epochs(self) -> list[Sequence[tuple[Phase, int]]]:
        match (self.dataset, self.setup):
            case ('imagenet100', sessions):
                if sessions == 's5':
                    return [
                        (
                            (Phase.PRETRAIN, self.pretrain_epochs[0]),
                            (Phase.TASK_LEARNING, self.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, self.pretrain_epochs[1]),
                        )
                    ] + [
                        (
                            (Phase.TASK_LEARNING, self.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, self.incremental_epochs[1]),
                        )
                    ] * 4
                elif sessions == 's10':
                    return [
                        (
                            (Phase.PRETRAIN, self.pretrain_epochs[0]),
                            (Phase.TASK_LEARNING, self.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, self.pretrain_epochs[1]),
                        )
                    ] + [
                        (
                            (Phase.TASK_LEARNING, self.incremental_epochs[0]),
                            (Phase.CONSOLIDATION, self.incremental_epochs[1]),
                        )
                    ] * 9
                elif sessions == 'full':
                    return [((Phase.PRETRAIN, self.pretrain_epochs[0]),)]
                else:
                    raise NotImplementedError(f'Unknown setup: {self.setup}')
            case _:
                raise NotImplementedError(f'Unsupported dataset setup: {self.dataset}@{self.setup}')

    @property
    def num_epochs(self) -> int:
        return sum(epochs for session in self.session_epochs for _, epochs in session)

    def get_last_checkpoint(self, force: bool = False) -> Path | None:
        curr_version = -1
        path = None
        for ckpt in (self.work_path / 'checkpoints').glob('last-v*.ckpt'):
            v = int(ckpt.stem.split('-')[1].removeprefix('v'))
            if v > curr_version:
                curr_version = v
                path = ckpt
        if path is None and ((self.work_path / 'checkpoints') / 'last.ckpt').exists(
            follow_symlinks=True
        ):
            path = (self.work_path / 'checkpoints') / 'last.ckpt'

        if path is None:
            if force:
                raise RuntimeError('Could not load checkpoint.')
            print("Can't resume training, no checkpoint found.")
        return path

    def verify(self):
        if not self.class_learner.use_task_modulations and not self.tmcl.disable_tmcl:
            raise ValueError('TMCL requires task modulations to be enabled.')

        if self.supcon.enable_supcon and not self.class_learner.labeled_setup == 'allvsall':
            raise ValueError('SupCon requires allvsall labeled setup.')
