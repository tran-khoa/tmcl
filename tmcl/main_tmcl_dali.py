import copy
import dataclasses
import itertools
import logging
import random
import re
from collections.abc import Iterable, Mapping
from functools import partial
from typing import Any, Final, override

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import simple_parsing
import torch
import torch.nn.functional as F
import wandb
from jaxtyping import Bool, Float, Integer
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.utils import (
    activate_requires_grad,
    deactivate_requires_grad,
    get_weight_decay_parameters,
)
from lightly.utils.scheduler import cosine_schedule
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split
from timm.models import ConvNeXt, VisionTransformer
from torch import Tensor, nn
from torch.nn import Identity
from torch.optim import SGD, Optimizer
from torchvision.models import ResNet
from tqdm import tqdm

from tmcl.config_tmcl_dali import Config
from tmcl.data import SessionBatch
from tmcl.datasets.dali_imagenet import (
    ContinualMeta,
    Dataset,
    ImageNet100Dataset,
    IncrementalDALISessionData,
)
from tmcl.eval_dali import EvalModule
from tmcl.hints import (
    BatchId,
    FeatureBatch,
    ImageBatch,
)
from tmcl.main_linear_dali import _run_linear
from tmcl.nn.convit import ConVit
from tmcl.nn.heads import TaskSpecificReadout
from tmcl.nn.losses.barlow import BarlowTwinsLoss
from tmcl.nn.losses.contrastive_opl import contrastive_opl
from tmcl.nn.losses.mv_barlow import GhoshBarlowLoss, MultiViewBarlowTwinsLoss
from tmcl.nn.losses.pnr import simsiam_loss_func
from tmcl.nn.losses.simclr import manual_simclr_loss_func, simclr_loss_func
from tmcl.nn.losses.supcon import supcon_positive_mask
from tmcl.nn.modulations import TaskModulationWrapper, build_tm_model
from tmcl.optim.lars import LARS
from tmcl.optim.warmup_cosine import warmup_cosine
from tmcl.phase import Phase
from tmcl.utils import (
    disable_optimizer_step_increment,
    enable_optimizer_step_increment,
)


class TMCLDALIModule(pl.LightningModule):
    cfg: Config
    wandb_logger: WandbLogger
    py_logger: logging.Logger

    backbone: TaskModulationWrapper

    def __init__(
        self,
        cfg: Config,
        *,
        wandb_logger: WandbLogger,
        py_logger: logging.Logger | None = None,
    ):
        cfg.verify()
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg

        match (self.cfg.dataset, self.cfg.setup):
            case ('imagenet100', sessions):
                dataset_classes: Final[int] = 100
                image_size: Final[int] = 224
                patch_size: Final[int] = 16

                rng = random.Random(cfg.seed)
                class_order = list(range(100))
                rng.shuffle(class_order)

                if sessions == 's5':
                    session_classes = [
                        class_order[session * 20 : (session + 1) * 20] for session in range(5)
                    ]
                elif sessions == 's10':
                    session_classes = [
                        class_order[session * 10 : (session + 1) * 10] for session in range(10)
                    ]
                elif sessions == 'full':
                    session_classes = [class_order]
                else:
                    raise NotImplementedError(f'Unknown setup: {self.cfg.setup}')
            case _:
                raise NotImplementedError(f'Unsupported dataset setup: {self.cfg.setup}')
        self.wandb_logger = wandb_logger
        self.py_logger = py_logger if py_logger is not None else logging.getLogger(__name__)

        self.session_classes = session_classes
        self.session_epochs = self.cfg.session_epochs

        match self.cfg.class_learner.labeled_setup:
            case 'onevsall':
                self.num_tasks = dataset_classes
                self.num_classes = 1
                self.py_logger.info(
                    f'Class-incremental learning setup, num_tasks={self.num_tasks}, num_classes={self.num_classes}'
                )
            case 'allvsall':
                self.num_tasks = 1
                self.num_classes = dataset_classes
                self.py_logger.info(
                    f'All-vs-all learning setup, num_tasks=1, num_classes={self.num_classes}'
                )
            case _:
                raise ValueError(f'Unknown labeled_setup {cfg.class_learner.labeled_setup}')

        self.backbone = build_tm_model(
            cfg.timm_model,
            num_tasks=self.num_tasks if cfg.class_learner.use_task_modulations else 0,
            image_size=image_size,
            patch_size=patch_size,
            has_bias=cfg.bias_modulations,
            pretrained=False,
        )

        match self.cfg.ssl.ssl_head:
            case 'barlow':
                self.ssl_head = BarlowTwinsProjectionHead(
                    input_dim=self.backbone.output_dim,
                    hidden_dim=cfg.ssl.head_hidden_dim,
                    output_dim=cfg.ssl.head_output_dim,
                )
            case _:
                raise ValueError(f'Unknown ssl_head {cfg.ssl.ssl_head}')

        match cfg.ssl.ssl_algo:
            case 'mvbarlow':
                self.ssl_loss = MultiViewBarlowTwinsLoss(
                    sync_normalize=self.cfg.n_devices > 1,
                    gather_distributed=self.cfg.n_devices > 1,
                    normalize_eps=cfg.ssl.normalize_eps,
                    lambda_param=cfg.ssl.barlow_lambda,
                    scale_loss=cfg.ssl.barlow_scale_loss,
                )
            case 'ghosh':
                self.ssl_loss = GhoshBarlowLoss(
                    sync_normalize=self.cfg.n_devices > 1,
                    gather_distributed=self.cfg.n_devices > 1,
                    normalize_eps=cfg.ssl.normalize_eps,
                    lambda_param=cfg.ssl.barlow_lambda,
                    scale_loss=cfg.ssl.barlow_scale_loss,
                )
            case 'barlow':
                if cfg.cons_num_views != 2:
                    raise ValueError(f'Barlow Twins requires 2 views, but got {cfg.cons_num_views}')
                self.ssl_loss = BarlowTwinsLoss(
                    sync_normalize=self.cfg.n_devices > 1,
                    gather_distributed=self.cfg.n_devices > 1,
                    normalize_eps=cfg.ssl.normalize_eps,
                    lambda_param=cfg.ssl.barlow_lambda,
                    scale_loss=cfg.ssl.barlow_scale_loss,
                    norm_strategy='per-branch',
                )
            case _:
                raise NotImplementedError(f'{cfg.ssl.ssl_algo} is unknown')

        self.tmcl_head = BarlowTwinsProjectionHead(
            input_dim=self.backbone.output_dim,
            hidden_dim=cfg.tmcl.head_hidden_dim,
            output_dim=cfg.tmcl.head_output_dim,
        )
        self.tmcl_pred = nn.Identity()
        if self.cfg.tmcl.use_predictor:
            self.tmcl_pred = nn.Sequential(
                nn.Linear(cfg.tmcl.head_output_dim, cfg.tmcl.pred_hidden_dim),
                nn.BatchNorm1d(cfg.tmcl.pred_hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.tmcl.pred_hidden_dim, cfg.tmcl.head_output_dim),
            )

        self.distill_backbone = None
        self.distill_head = None
        self.distill_pred = nn.Identity()
        self.distill_loss = nn.Identity()
        if self.cfg.distill.algo != 'none':
            # if self.cfg.resume:
            self.distill_backbone = copy.deepcopy(self.backbone)
            if cfg.torch_compile:
                self.distill_backbone = torch.compile(self.distill_backbone)
            self.distill_backbone.set_feedforward_grads(False)
            self.distill_backbone.set_task(None, update_grads=True)
            self.distill_backbone.eval()

            self.distill_head = copy.deepcopy(self.ssl_head)
            if cfg.torch_compile:
                self.distill_head = torch.compile(self.distill_head)
            deactivate_requires_grad(self.distill_head)
            self.distill_head.eval()

            if self.cfg.distill.use_predictor:
                input_dim = cfg.ssl.head_output_dim
                self.distill_pred = nn.Sequential(
                    nn.Linear(input_dim, cfg.distill.head_hidden_dim),
                    nn.BatchNorm1d(cfg.distill.head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.distill.head_hidden_dim, input_dim),
                )

            if self.cfg.distill.distill_ssl == 'barlow':
                self.distill_loss = BarlowTwinsLoss(
                    lambda_param=cfg.distill.barlow_lambda,
                    scale_loss=cfg.distill.barlow_scale_loss,
                    norm_strategy='per-branch',
                    gather_distributed=cfg.n_devices > 1,
                    sync_normalize=cfg.n_devices > 1,
                )
            elif self.cfg.distill.distill_ssl == 'ghosh':
                self.distill_loss = GhoshBarlowLoss(
                    lambda_param=cfg.distill.barlow_lambda,
                    scale_loss=cfg.distill.barlow_scale_loss,
                    gather_distributed=cfg.n_devices > 1,
                    sync_normalize=cfg.n_devices > 1,
                )
            else:
                raise ValueError()

        self.suphead_session_heads = nn.Identity()
        if self.cfg.suphead.enable_sup_head:
            # https://github.com/danielm1405/sl-vs-ssl-cl/blob/master/cassle/methods/supervised.py
            # (MLPP) https://github.com/danielm1405/sl-vs-ssl-cl/blob/master/cassle/utils/projectors.py
            self.suphead_session_heads = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(self.backbone.output_dim, cfg.suphead.head_hidden_dim),
                    nn.BatchNorm1d(cfg.suphead.head_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(cfg.suphead.head_hidden_dim, self.backbone.output_dim),
                    nn.Linear(self.backbone.output_dim, len(s)),
                )
                for s in session_classes
            )

        self.supcon_head = nn.Identity()
        if self.cfg.supcon.enable_supcon:
            self.supcon_head = nn.Sequential(
                nn.Linear(self.backbone.output_dim, cfg.supcon.head_hidden_dim),
                nn.BatchNorm1d(cfg.supcon.head_hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.supcon.head_hidden_dim, cfg.supcon.head_output_dim),
            )

        if cfg.torch_compile:
            self.backbone = torch.compile(self.backbone)
            self.ssl_head = torch.compile(self.ssl_head)
            self.tmcl_head = torch.compile(self.tmcl_head)
            self.distill_pred = torch.compile(self.distill_pred)
            self.supcon_head = torch.compile(self.supcon_head)

        match cfg.tmcl.tmcl_algo:
            case 'mvbarlow':
                self.tmcl_loss = MultiViewBarlowTwinsLoss(
                    lambda_param=cfg.tmcl.barlow_lambda,
                    scale_loss=cfg.tmcl.barlow_scale_loss,
                )
            case 'ghosh':
                self.tmcl_loss = GhoshBarlowLoss(
                    sync_normalize=self.cfg.n_devices > 1,
                    gather_distributed=self.cfg.n_devices > 1,
                    normalize_eps=cfg.ssl.normalize_eps,
                    lambda_param=cfg.ssl.barlow_lambda,
                    scale_loss=cfg.ssl.barlow_scale_loss,
                )
            case 'barlow':
                if cfg.cons_num_views != 2:
                    raise ValueError(f'Barlow Twins requires 2 views, but got {cfg.cons_num_views}')
                self.tmcl_loss = BarlowTwinsLoss(
                    normalize_eps=cfg.ssl.normalize_eps,
                    lambda_param=cfg.tmcl.barlow_lambda,
                    scale_loss=cfg.tmcl.barlow_scale_loss,
                    norm_strategy='per-branch',
                )
            case _:
                raise NotImplementedError(f'{cfg.tmcl.tmcl_algo} is unknown')

        match cfg.class_learner.method:
            case 'sup' | 'til_ce':
                self.class_heads = TaskSpecificReadout(
                    in_features=self.backbone.output_dim,
                    num_classes=self.num_classes,
                    num_tasks=self.num_tasks,
                )
            case 'contrastive_opl':
                self.class_heads = Identity()
            case _:
                raise NotImplementedError(f'{cfg.class_learner.method} is unknown')

        if cfg.load_checkpoint_from is not None:
            py_logger.info(f'Loading checkpoint from {cfg.load_checkpoint_from}')
            checkpoint_pkl = torch.load(
                cfg.load_checkpoint_from, map_location=self.device, weights_only=False
            )

            backbone_sd = {
                k.removeprefix('backbone.'): v
                for k, v in checkpoint_pkl['state_dict'].items()
                if k.startswith('backbone.')
            }

            task_id_re = re.compile(r'.*_task(\d+)')
            checkpoint_tasks = set(
                int(match.group(1)) for k in backbone_sd if (match := task_id_re.match(k))
            )
            missing, unexpected = self.backbone.load_state_dict(
                backbone_sd,
                strict=False,
            )
            # we expect keys for new tasks to be missing
            missing = [
                k
                for k in missing
                if (not (m := task_id_re.match(k))) or int(m.group(1)) in checkpoint_tasks
            ]

            py_logger.info(f'Missing keys: {missing}')
            py_logger.info(f'Unexpected keys: {unexpected}')

            if cfg.load_checkpoint_heads:
                py_logger.info(f'Loading heads from {cfg.load_checkpoint_from}')
                missing, unexpected = self.ssl_head.load_state_dict(
                    {
                        k.removeprefix('ssl_head.'): v
                        for k, v in checkpoint_pkl['state_dict'].items()
                        if k.startswith('ssl_head.')
                    }
                )
                py_logger.info(f'Missing keys: {missing}')
                py_logger.info(f'Unexpected keys: {unexpected}')
                if not cfg.ignore_checkpoint_tmcl_heads:
                    missing, unexpected = self.tmcl_head.load_state_dict(
                        {
                            k.removeprefix('tmcl_head.'): v
                            for k, v in checkpoint_pkl['state_dict'].items()
                            if k.startswith('tmcl_head.')
                        }
                    )
                    py_logger.info(f'Missing keys: {missing}')
                    py_logger.info(f'Unexpected keys: {unexpected}')

        # noinspection PyTypeChecker
        self.save_hyperparameters(dataclasses.asdict(cfg))

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        if all(not k.startswith('distill_backbone') for k in state_dict):
            # if distill is not in state_dict, we can skip loading it
            self.distill_backbone = None
            self.distill_head = None

        strict = self.cfg.resume_strict
        issues = super().load_state_dict(state_dict, strict, assign)
        if issues.missing_keys:
            self.py_logger.warning(
                f'Missing keys in state_dict: {issues.missing_keys}, strict={strict}'
            )
        if issues.unexpected_keys:
            self.py_logger.warning(
                f'Unexpected keys in state_dict: {issues.unexpected_keys}, strict={strict}'
            )
        return issues

    def setup(self, stage: str) -> None:
        if stage != 'fit':
            raise NotImplementedError

        if self.cfg.dataset == 'imagenet100':
            train_dataset = ImageNet100Dataset(
                tar_root=self.cfg.data_path / 'imagenet100',
                split='train',
                n_procs=self.cfg.n_data_workers,
                supervised_frac=self.cfg.labeled_frac,
                seed=self.cfg.seed,
            )
            if self.cfg.eval.eval_valid:
                train_files, eval_files, train_labels, eval_labels, train_supervised, _ = (
                    train_test_split(
                        train_dataset.files,
                        train_dataset.labels,
                        train_dataset.supervised,
                        train_size=0.85,
                        random_state=self.cfg.seed,
                        shuffle=True,
                        stratify=train_dataset.labels,
                    )
                )
                train_dataset = Dataset(
                    files=train_files,
                    labels=train_labels,
                    supervised=train_supervised,
                )
                eval_dataset = Dataset(
                    files=eval_files,
                    labels=eval_labels,
                    supervised=[True] * len(eval_files),
                )
            else:
                eval_dataset = ImageNet100Dataset(
                    tar_root=self.cfg.data_path / 'imagenet100',
                    split='val',
                    n_procs=self.cfg.n_data_workers,
                )

            self.session_data = IncrementalDALISessionData(
                dataset=train_dataset,
                session_classes=self.session_classes,
                batch_size_per_gpu=self.cfg.batch_size_per_gpu,
                workers=self.cfg.n_data_workers,
                seed=self.cfg.seed,
                num_views=self.cfg.cons_num_views,
                device=self.cfg.accelerator,
                shard_id=self.trainer.global_rank,
                num_shards=self.trainer.world_size,
                contrastive_allvsall=self.cfg.class_learner.labeled_setup == 'allvsall',
            )

            self.continual_meta = ContinualMeta(
                dataset=train_dataset,
                session_classes=self.session_classes,
                session_epochs=self.session_epochs,
                total_batch_size=self.cfg.batch_size,
            )

            self.eval_module = EvalModule(
                model=self.backbone,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                shard_id=self.global_rank,
                num_shards=self.trainer.world_size,
                batch_size_per_device=self.cfg.batch_size_per_gpu,
                device=self.cfg.accelerator,
                num_workers=self.cfg.n_data_workers,
            )

        else:
            raise ValueError(f'Unknown dataset {self.cfg.dataset}')

    @override
    def on_train_start(self) -> None:
        # Basically lightning's sanity check
        if self.cfg.eval.eval_before_training:
            self.custom_evaluation_loop(eval_before_training=True)

    @override
    def train_dataloader(self) -> Iterable[SessionBatch]:
        self.continual_meta.set_current_step(self.global_step, self.current_epoch)
        session = self.continual_meta.session
        phase = self.continual_meta.phase
        return self.session_data.train_dataloader(session, phase=phase)

    @override
    def configure_optimizers(self):
        self.py_logger.info('Configuring optimizers...')

        def get_optim_builder(algo: str):
            match algo:
                case 'adamw':
                    _optim_fn = partial(
                        torch.optim.AdamW,
                        betas=self.cfg.optim.adam_betas,
                        lr=0.0,
                    )
                case 'lars':
                    _optim_fn = partial(
                        LARS,
                        lr=0.0,
                        momentum=self.cfg.optim.sgd_momentum,
                        eta=self.cfg.optim.lars_eta,
                        clip_lr=self.cfg.optim.lars_clip,
                        exclude_bias_n_norm=self.cfg.optim.lars_exclude_bias_n_norm,
                    )
                case 'sgd':
                    _optim_fn = partial(SGD, lr=0.0, momentum=self.cfg.optim.sgd_momentum)
                case _:
                    raise ValueError(f'Unknown optimizer {self.cfg.optim.optim_algo}')
            return _optim_fn

        optim_fn = get_optim_builder(self.cfg.optim.optim_algo)
        tl_optim_fn = (
            get_optim_builder(self.cfg.optim.tl_optim_algo)
            if self.cfg.optim.tl_optim_algo
            else optim_fn
        )

        if isinstance(self.backbone.module, VisionTransformer | ConVit):
            num_blocks: int = len(self.backbone.module.blocks)
        elif isinstance(self.backbone.module, ResNet):
            num_blocks: int = 5  # hardcoded actually, disregarding first conv1
        else:
            raise ValueError(f'Unknown module {type(self.backbone.module)}')
        tl_param_groups = {
            'class_heads': {
                'name': 'class_heads',
                'params': list(self.class_heads.parameters()),
            },
            'task_modulations': {'name': 'task_modulations', 'params': []},
            **{
                f'block{idx}/task_modulations': {
                    'name': f'block{idx}/task_modulations',
                    'params': [],
                    'weight_decay': cosine_schedule(
                        idx,
                        num_blocks,
                        start_value=self.cfg.class_learner.layerwise_weight_decay[0],
                        end_value=self.cfg.class_learner.layerwise_weight_decay[1],
                    ),
                }
                for idx in range(num_blocks)
            },
        }
        cons_param_groups = {
            'ssl_proj': {'name': 'ssl_proj', 'params': []},
            'ssl_proj/nodecay': {'name': 'ssl_proj/nodecay', 'params': []},
            'tmcl_proj': {'name': 'tmcl_proj', 'params': []},
            'tmcl_proj/nodecay': {'name': 'tmcl_proj/nodecay', 'params': []},
            'feedforward': {'name': 'feedforward', 'params': []},
            'feedforward/nodecay': {'name': 'feedforward/nodecay', 'params': []},
            **{
                f'block{idx}/feedforward': {'name': f'block{idx}/feedforward', 'params': []}
                for idx in range(num_blocks)
            },
            **{
                f'block{idx}/feedforward/nodecay': {
                    'name': f'block{idx}/feedforward/nodecay',
                    'params': [],
                }
                for idx in range(num_blocks)
            },
            'cons_modulation': {'name': 'cons_modulation', 'params': []},
            **{
                f'block{idx}/cons_modulation': {
                    'name': f'block{idx}/cons_modulation',
                    'params': [],
                    'weight_decay': cosine_schedule(
                        idx,
                        num_blocks,
                        start_value=self.cfg.class_learner.layerwise_weight_decay[0],
                        end_value=self.cfg.class_learner.layerwise_weight_decay[1],
                    ),
                }
                for idx in range(num_blocks)
            },
        }

        for name, module in self.backbone.modulations:
            if isinstance(self.backbone.module, VisionTransformer | ConVit) and 'blocks.' in name:
                block_index = int(name.split('blocks.')[1].split('.')[0])
                pg_prefix = f'block{block_index}/'
            elif isinstance(self.backbone.module, ResNet):
                block_index = int(name.split('layer')[1].split('.')[0]) if 'layer' in name else 0
                pg_prefix = f'block{block_index}/'
            elif isinstance(self.backbone.module, ConvNeXt):
                if name.startswith('stem.'):
                    pg_prefix = 'block0/'
                elif name.startswith('stages.'):
                    block_index = int(name.split('stages.')[1].split('.')[0])
                    pg_prefix = f'block{block_index}/'
                else:
                    raise ValueError(f'Unknown ConvNeXt module name {name}!')
            else:
                pg_prefix = ''
            for gain, bias in itertools.zip_longest(module.gains, module.biases):
                tl_param_groups[f'{pg_prefix}task_modulations']['params'] += [gain]
                if bias is not None:
                    tl_param_groups[f'{pg_prefix}task_modulations']['params'] += [bias]

        _, p_feedforward_nodecay = get_weight_decay_parameters([self.backbone])
        p_feedforward_nodecay_ids = {id(p) for p in p_feedforward_nodecay}

        for name, p in self.backbone.feedforward_parameters.items():
            if isinstance(self.backbone.module, VisionTransformer | ConVit) and 'blocks.' in name:
                block_index = int(name.split('blocks.')[1].split('.')[0])
                pg_prefix = f'block{block_index}/'
            elif isinstance(self.backbone.module, ResNet) and 'layer' in name:
                block_index = int(name.split('layer')[1].split('.')[0])
                pg_prefix = f'block{block_index}/'
            elif isinstance(self.backbone.module, ConvNeXt):
                if name.startswith('stem.'):
                    pg_prefix = 'block0/'
                elif name.startswith('stages.'):
                    block_index = int(name.split('stages.')[1].split('.')[0])
                    pg_prefix = f'block{block_index}/'
                elif name.startswith('norm_pre.'):
                    pg_prefix = ''
                else:
                    raise ValueError(f'Unknown ConvNeXt module name {name}!')
            else:
                pg_prefix = ''
            if id(p) in p_feedforward_nodecay_ids:
                cons_param_groups[f'{pg_prefix}feedforward/nodecay']['params'].append(p)
            else:
                cons_param_groups[f'{pg_prefix}feedforward']['params'].append(p)

        p_ssl_proj, p_ssl_proj_nodecay = get_weight_decay_parameters([self.ssl_head])
        cons_param_groups['ssl_proj']['params'] = p_ssl_proj
        cons_param_groups['ssl_proj/nodecay']['params'] = p_ssl_proj_nodecay

        p_tmcl_proj, p_tmcl_proj_nodecay = get_weight_decay_parameters([self.tmcl_head])
        p_tmcl_pred, p_tmcl_pred_nodecay = get_weight_decay_parameters([self.tmcl_pred])

        cons_param_groups['tmcl_proj']['params'] = p_tmcl_proj + p_tmcl_pred
        cons_param_groups['tmcl_proj/nodecay']['params'] = p_tmcl_proj_nodecay + p_tmcl_pred_nodecay

        if not isinstance(self.distill_pred, nn.Identity):
            p_distill_head, p_distill_head_nodecay = get_weight_decay_parameters(
                [self.distill_pred]
            )
            cons_param_groups['distill_head'] = {'name': 'distill_head', 'params': p_distill_head}
            cons_param_groups['distill_head/nodecay'] = {
                'name': 'distill_head/nodecay',
                'params': p_distill_head_nodecay,
            }
        if not isinstance(self.supcon_head, nn.Identity):
            p_supcon_head, p_supcon_head_nodecay = get_weight_decay_parameters([self.supcon_head])
            cons_param_groups['supcon_head'] = {'name': 'supcon_head', 'params': p_supcon_head}
            cons_param_groups['supcon_head/nodecay'] = {
                'name': 'supcon_head/nodecay',
                'params': p_supcon_head_nodecay,
            }
        if not isinstance(self.suphead_session_heads, nn.Identity):
            p_suphead_session_heads, p_suphead_session_heads_nodecay = get_weight_decay_parameters(
                [self.suphead_session_heads]
            )
            cons_param_groups['suphead_session_heads'] = {
                'name': 'suphead_session_heads',
                'params': p_suphead_session_heads,
            }
            cons_param_groups['suphead_session_heads/nodecay'] = {
                'name': 'suphead_session_heads/nodecay',
                'params': p_suphead_session_heads_nodecay,
            }

        def get_pg(pg):
            # creates a dictionary copy, otherwise pg['lr'] affects all pg's across optimizers...
            return [{k: v for k, v in _pg.items()} for _pg in pg.values()]

        tl_optim = tl_optim_fn(get_pg(tl_param_groups))
        cons_optim = optim_fn(get_pg(cons_param_groups))

        return tl_optim, cons_optim

    @override
    def on_train_epoch_start(self) -> None:
        self.continual_meta.set_current_step(global_step=self.global_step, epoch=self.current_epoch)

        self.log(
            'progress/current_session',
            self.continual_meta.session,
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            'progress/current_phase',
            self.continual_meta.phase.value,
            prog_bar=True,
            sync_dist=False,
        )

    @torch.no_grad()
    def update_weight_decay(self, consolidation_optim: Optimizer) -> None:
        weight_decay = cosine_schedule(
            self.trainer.global_step,
            self.continual_meta.total_steps,
            start_value=self.cfg.optim.backbone_weight_decay[0],
            end_value=self.cfg.optim.backbone_weight_decay[1],
        )

        for param_group in consolidation_optim.param_groups:
            path = param_group['name'].split('/')
            if 'nodecay' not in path and 'cons_modulation' not in path:
                param_group['weight_decay'] = weight_decay

        self.log('progress/backbone_weight_decay', weight_decay, sync_dist=False)

    @torch.no_grad()
    def update_lrs(self, tl_optim: Optimizer, consolidation_optim: Optimizer) -> None:
        self.log(
            'progress/session_step',
            self.continual_meta.session_step,
            prog_bar=False,
            sync_dist=False,
        )
        session_step = self.continual_meta.session_step
        current_session = self.continual_meta.session
        current_phase = self.continual_meta.phase

        phase_epochs = self.continual_meta.num_phase_epochs
        phase_steps = self.continual_meta.num_phase_steps
        for param_group in tl_optim.param_groups:
            if self.continual_meta.phase == Phase.TASK_LEARNING or (
                self.continual_meta.phase == Phase.PRETRAIN
                and self.cfg.class_learner.during_pretraining
            ):
                previous_steps = 0
                for session, phase, steps in reversed(
                    self.continual_meta.timeline_steps[: self.continual_meta.current_idx]
                ):
                    if session != current_session:
                        break
                    if (
                        phase == Phase.PRETRAIN
                        and not self.cfg.class_learner.during_pretraining
                        or phase == Phase.CONSOLIDATION
                    ):
                        previous_steps += steps

                param_group['lr'] = warmup_cosine(
                    session_step + 1 - previous_steps,
                    peak_lr=self.cfg.optim.tl_lr * (self.cfg.batch_size / 256),
                    num_steps=phase_steps,
                    warmup_steps=int((self.cfg.optim.warmup_epochs / phase_epochs) * phase_steps),
                    end_lr=self.cfg.optim.tl_min_lr,
                    start_lr=self.cfg.optim.tl_start_lr,
                )
            else:
                param_group['lr'] = 0.0
        # noinspection PyUnboundLocalVariable
        self.log('progress/lr_tl', param_group['lr'], prog_bar=False, sync_dist=False)

        for param_group in consolidation_optim.param_groups:
            previous_steps = 0
            for session, phase, steps in reversed(
                self.continual_meta.timeline_steps[: self.continual_meta.current_idx]
            ):
                if session != current_session:
                    break
                if phase not in (Phase.PRETRAIN, Phase.CONSOLIDATION):
                    previous_steps += steps

            # relative to the number of cons./pret. steps
            current_step = (
                session_step + 1 - previous_steps
                if current_phase in (Phase.PRETRAIN, Phase.CONSOLIDATION)
                else 0
            )
            num_steps = sum(
                steps
                for session, phase, steps in self.continual_meta.timeline_steps
                if session == current_session and phase in (Phase.PRETRAIN, Phase.CONSOLIDATION)
            )
            ssl_epochs = sum(
                epochs
                for session, phase, epochs in self.continual_meta.timeline_epochs
                if session == current_session and phase in (Phase.PRETRAIN, Phase.CONSOLIDATION)
            )
            warmup_steps = int((self.cfg.optim.warmup_epochs / ssl_epochs) * num_steps)
            param_group['lr'] = warmup_cosine(
                current_step,
                peak_lr=self.cfg.optim.cons_lr * (self.cfg.batch_size / 256),
                num_steps=num_steps,
                warmup_steps=warmup_steps,
                end_lr=self.cfg.optim.cons_min_lr,
                start_lr=self.cfg.optim.cons_start_lr,
            )

        self.log('progress/lr_cons', param_group['lr'], prog_bar=False, sync_dist=False)

    def class_learning_task(self, task_idx: int, session_idx: int) -> int:
        """
        Which task (e.g. modulation/readout) to assume for task-specific computations.
        """

        match self.cfg.class_learner.labeled_setup:
            case 'onevsall':
                return task_idx
            case 'allvsall':
                return 0
            case _:
                raise ValueError(f'Unknown labeled_setup {self.cfg.class_learner.labeled_setup}')

    def sample_tasks(
        self,
        batch_size: int,
        num_views: int | None = None,
    ) -> Integer[Tensor, 'num_views * batch']:
        if num_views is None:
            num_views = self.cfg.cons_num_views

        seen_classes_tensor = torch.tensor(
            list(self.continual_meta.seen_classes),
            device=self.device,
            dtype=torch.long,
        )
        tmcl_tasks: Integer[Tensor, 'num_views * batch'] = seen_classes_tensor[
            torch.randint(
                len(seen_classes_tensor),
                size=(num_views * batch_size,),
                device=self.device,
                dtype=torch.long,
            )
        ]

        return tmcl_tasks

    @override
    def training_step(
        self,
        batch: SessionBatch,
        batch_index: BatchId,
    ) -> None:
        current_phase = self.continual_meta.phase
        current_session = self.continual_meta.session

        tl_optim, consolidation_optim = self.optimizers()
        self.update_weight_decay(consolidation_optim)
        self.update_lrs(tl_optim, consolidation_optim)

        if current_phase in (Phase.PRETRAIN,):
            disable_optimizer_step_increment(tl_optim)
        else:
            enable_optimizer_step_increment(tl_optim)

        if current_phase == Phase.TASK_LEARNING or (
            current_phase == Phase.PRETRAIN and self.cfg.class_learner.during_pretraining
        ):
            class_learning_task = self.class_learning_task(batch.task.task_idx, current_session)
            tl_optim.zero_grad()
            self.backbone.set_feedforward_grads(False)
            self.backbone.eval()  # also freeze BNs

            match self.cfg.class_learner.method:
                case 'sup' | 'til_ce':
                    # class-incremental binary cross entropy on 1-vs-all tasks
                    self.backbone.set_task(
                        torch.full(
                            (batch.task.batch_size,),
                            fill_value=class_learning_task,
                            device=self.device,
                        )
                        if self.cfg.class_learner.use_task_modulations
                        else None,
                        update_grads=True,
                    )
                    self.class_heads.set_task(class_learning_task, update_grads=True)

                    sup_images: ImageBatch = batch.task.images
                    sup_embeds: FeatureBatch = self.backbone(sup_images)
                    logits: Float[Tensor, 'batch'] = self.class_heads(sup_embeds).squeeze()
                    if self.cfg.class_learner.labeled_setup == 'onevsall':
                        tl_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, batch.task.labels.float()
                        )
                    else:
                        tl_loss = torch.nn.functional.cross_entropy(logits, batch.task.labels)

                case 'contrastive_opl':
                    self.backbone.set_task(
                        torch.full(
                            (batch.task.batch_size,),
                            fill_value=class_learning_task,
                            device=self.device,
                        )
                        if self.cfg.class_learner.use_task_modulations
                        else None,
                        update_grads=True,
                    )

                    sup_images: ImageBatch = batch.task.images
                    sup_embeds: FeatureBatch = self.backbone(sup_images)
                    tl_loss, tl_stats = contrastive_opl(
                        sup_embeds,
                        batch.task.labels,
                        neg_weight=self.cfg.class_learner.opl_neg_weight,
                        square_loss=self.cfg.class_learner.opl_square_loss,
                    )
                    self.log_dict({f'train_tl/{k}': v for k, v in tl_stats.items()}, prog_bar=False)
                case _:
                    raise NotImplementedError(f'{self.cfg.class_learner.method} is unknown')

            self.log(
                f'train_tl/{class_learning_task}/sup_loss',
                tl_loss,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch.task.batch_size,
            )
            self.manual_backward(tl_loss)
            if self.cfg.optim.clip_grad is not None:
                self.clip_gradients(
                    tl_optim,
                    gradient_clip_val=self.cfg.optim.clip_grad,
                    gradient_clip_algorithm='norm',
                )
            tl_optim.step()
            tl_optim.zero_grad()
            self.backbone.train()

        if current_phase in (Phase.PRETRAIN, Phase.CONSOLIDATION):
            consolidation_optim.zero_grad()
            self.backbone.set_feedforward_grads(True)
            self.backbone.train()

            do_task_inv: Final[bool] = (
                current_phase == Phase.CONSOLIDATION and not self.cfg.tmcl.disable_tmcl
            ) or (
                current_phase == Phase.PRETRAIN
                and self.cfg.tmcl.pretraining
                and self.current_epoch > self.cfg.tmcl.pretraining_after_n_epochs
            )
            do_distill: Final[bool] = (
                self.cfg.distill.algo != 'none'
                and current_phase == Phase.CONSOLIDATION
                and (current_session > 0)
            )
            do_ssl: Final[bool] = (
                not self.cfg.ssl.disable_ssl
                or (  # run if TMCL is enabled and in pretraining phase
                    not self.cfg.tmcl.disable_tmcl and current_phase == Phase.PRETRAIN
                )
            )

            grad_modulations = []
            self.backbone.set_task_grads(grad_modulations)

            cons_images = torch.cat(batch.cons.images, dim=0)

            self.backbone.set_task(None, update_grads=False)
            ssl_loss = 0.0
            distill_loss = 0.0
            if do_ssl:
                ssl_embeds = self.backbone(cons_images)
                ssl_projs = self.ssl_head(ssl_embeds)
                ssl_loss, bt_stats = self.ssl_loss(*ssl_projs.split(batch.cons.batch_size))
                for k, v in bt_stats.items():
                    self.log(
                        f'train_cons/ssl_{k}',
                        v,
                        prog_bar=False,
                        sync_dist=True,
                        batch_size=batch.cons.batch_size,
                    )

                self.log(
                    'train_cons/ssl_loss',
                    ssl_loss,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch.cons.batch_size,
                )

                if do_distill and self.cfg.distill.algo in ('pnr', 'cassle'):
                    # do with two views only
                    # distill_images = cons_images[: batch.cons.batch_size * 2, :, :, :]
                    distill_images = cons_images
                    with torch.no_grad():
                        distill_embeds = self.distill_backbone(distill_images)
                        distill_projs = self.distill_head(distill_embeds)

                    # curr_preds = self.distill_pred(ssl_projs[: batch.cons.batch_size])
                    curr_preds = self.distill_pred(ssl_projs)

                    frozens = distill_projs.split(batch.cons.batch_size)
                    ps = curr_preds.split(batch.cons.batch_size)

                    if self.cfg.distill.distill_ssl == 'barlow':
                        self.distill_loss: BarlowTwinsLoss
                        distill_loss = (
                            self.distill_loss(ps[0], frozens[0])[0]
                            + self.distill_loss(ps[1], frozens[1])[0]
                        ) / 2
                    elif self.cfg.distill.distill_ssl == 'ghosh':
                        self.distill_loss: GhoshBarlowLoss
                        distill_loss = sum(
                            self.distill_loss(p, frozen_z.detach())[0]
                            for p, frozen_z in zip(ps, frozens, strict=True)
                        ) / len(frozens)
                    else:
                        raise ValueError(f'Unknown distill_ssl {self.cfg.distill.distill_ssl}!')

                    self.log(
                        'train_cons/distill_loss',
                        distill_loss,
                        sync_dist=True,
                        prog_bar=True,
                    )
                    if self.cfg.distill.distill_ssl == 'barlow' and self.cfg.distill.algo == 'pnr':
                        pnr_loss = (
                            -self.cfg.distill.pnr_neg_lamb
                            * (
                                simsiam_loss_func(ps[0], frozens[1])
                                + simsiam_loss_func(ps[1], frozens[0])
                            )
                            / 2
                        )
                        self.log(
                            'train_cons/distill_pnr',
                            pnr_loss,
                            sync_dist=True,
                        )

                    elif self.cfg.distill.distill_ssl == 'ghosh' and self.cfg.distill.algo == 'pnr':
                        assert self.cfg.cons_num_views >= 4, "can't be bothered to generalize this"
                        pnr_loss = (
                            -self.cfg.distill.pnr_neg_lamb
                            * (
                                simsiam_loss_func(ps[0], frozens[1])
                                + simsiam_loss_func(ps[1], frozens[0])
                                + simsiam_loss_func(ps[2], frozens[3])
                                + simsiam_loss_func(ps[3], frozens[2])
                            )
                            / 4
                        )
                        self.log(
                            'train_cons/distill_pnr',
                            pnr_loss,
                            sync_dist=True,
                        )
                        distill_loss = distill_loss + self.cfg.distill.pnr_weight * pnr_loss

            if do_task_inv:
                tmcl_images = cons_images
                if self.cfg.tmcl.single_augment:
                    tmcl_images = cons_images[: batch.cons.batch_size, :, :, :].repeat(
                        self.cfg.cons_num_views, 1, 1, 1
                    )
                tmcl_tasks: Integer[Tensor, 'num_views * batch'] = self.sample_tasks(
                    batch_size=batch.cons.batch_size
                )

                if self.cfg.tmcl.random_modulations:
                    self.backbone.reset_modulations()

                elif self.cfg.tmcl.unmod_first_view:
                    tmcl_tasks[: batch.cons.batch_size] = self.backbone.no_mod_idx

                self.tmcl_head.train()

                if self.cfg.tmcl.stop_grad_tms:
                    # fast track computation
                    self.backbone.set_task(tmcl_tasks[: batch.cons.batch_size], update_grads=False)
                    tmcl_embeds_0 = self.backbone(tmcl_images[: batch.cons.batch_size, :, :, :])
                    tmcl_projs_0 = self.tmcl_head(tmcl_embeds_0)
                    with torch.no_grad():
                        self.backbone.set_task(
                            tmcl_tasks[batch.cons.batch_size :], update_grads=False
                        )
                        tmcl_embeds_rest = self.backbone(
                            tmcl_images[batch.cons.batch_size :, :, :, :]
                        )
                        tmcl_projs_rest = self.tmcl_head(tmcl_embeds_rest)
                    tmcl_projs = [tmcl_projs_0, *tmcl_projs_rest.split(batch.cons.batch_size)]
                else:
                    self.backbone.set_task(tmcl_tasks, update_grads=False)
                    tmcl_embeds = self.backbone(tmcl_images)
                    tmcl_projs = self.tmcl_head(tmcl_embeds).split(batch.cons.batch_size)

                if self.cfg.tmcl.use_predictor:
                    tmcl_preds = self.tmcl_pred(tmcl_projs[0])
                else:
                    tmcl_preds = tmcl_projs[0]

                tmcl_projs = tmcl_projs[1:]
                tmcl_loss, tmcl_stats = self.tmcl_loss(tmcl_preds, *tmcl_projs)
                self.log_dict(
                    {f'train_cons/tmcl_{k}': v for k, v in tmcl_stats.items()},
                    batch_size=batch.cons.batch_size,
                )
                self.log(
                    'train_cons/tmcl_loss',
                    tmcl_loss,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch.cons.batch_size,
                )
            else:
                tmcl_loss = 0.0

            sup_head_loss = 0.0
            if self.cfg.suphead.enable_sup_head:
                for s in range(self.continual_meta.num_sessions):
                    if s == current_session:
                        activate_requires_grad(self.suphead_session_heads[s])
                    else:
                        deactivate_requires_grad(self.suphead_session_heads[s])

                suphead_images = torch.cat(batch.task.images, dim=0)
                self.backbone.set_task(None, update_grads=False)
                suphead_embeds = self.backbone(suphead_images)
                suphead_logits = self.suphead_session_heads[current_session](suphead_embeds)
                suphead_labels = batch.task.labels.repeat(self.cfg.cons_num_views)
                # remap to 0..(|C|-1)
                current_classes = self.session_classes[current_session]
                comparison: Bool[Tensor, 'batch current_classes'] = suphead_labels.unsqueeze(
                    1
                ) == torch.tensor(current_classes, device=self.device).unsqueeze(0)
                suphead_labels = torch.argmax(comparison.float(), dim=1)

                sup_head_loss = F.cross_entropy(suphead_logits, suphead_labels)

                # compute labels
                self.log(
                    'train_cons/suphead_loss',
                    sup_head_loss,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch.task.batch_size,
                )

            supcon_loss = 0.0
            if self.cfg.supcon.enable_supcon:
                pos_mask = supcon_positive_mask(batch.task.labels, self.cfg.cons_num_views)

                supcon_images = torch.cat(batch.task.images, dim=0)
                self.backbone.set_task(None, update_grads=False)
                supcon_embeds = self.backbone(supcon_images)
                supcon_projs = self.supcon_head(supcon_embeds)

                if self.cfg.cons_num_views == 2:
                    z1, z2 = supcon_projs.split(batch.task.batch_size)
                    supcon_loss = simclr_loss_func(
                        z1, z2, extra_pos_mask=pos_mask, temperature=self.cfg.supcon.temperature
                    )
                else:
                    neg_mask = (~pos_mask).fill_diagonal_(False)
                    supcon_loss = manual_simclr_loss_func(
                        supcon_projs,
                        pos_mask=pos_mask,
                        neg_mask=neg_mask,
                        temperature=self.cfg.supcon.temperature,
                    )
                self.log(
                    'train_cons/supcon_loss',
                    supcon_loss,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch.task.batch_size,
                )

            if current_phase == Phase.PRETRAIN:
                consolidation_loss = (
                    ssl_loss
                    + self.cfg.suphead.sup_head_weight * sup_head_loss
                    + self.cfg.supcon.supcon_weight * supcon_loss
                )  # no TMCL loss, no ssl_loss reweighting(!)
            else:
                consolidation_loss = (
                    self.cfg.tmcl.tmcl_weight * tmcl_loss
                    + self.cfg.ssl.ssl_weight * ssl_loss
                    + self.cfg.distill.distill_weight * distill_loss
                    + self.cfg.suphead.sup_head_weight * sup_head_loss
                    + self.cfg.supcon.supcon_weight * supcon_loss
                )

            self.manual_backward(consolidation_loss)
            if self.cfg.optim.clip_grad is not None:
                self.clip_gradients(
                    consolidation_optim,
                    gradient_clip_val=self.cfg.optim.clip_grad,
                    gradient_clip_algorithm='norm',
                )
            consolidation_optim.step()
            consolidation_optim.zero_grad()

    @override
    def on_train_epoch_end(self) -> None:
        session = self.continual_meta.session
        is_not_last_epoch = self.current_epoch < self.trainer.max_epochs - 1

        if self.continual_meta.is_last_session_epoch and is_not_last_epoch:
            if self.cfg.distill.algo != 'none':
                self.py_logger.info(f'End of session {session}, updating distillation backbone.')
                self.distill_backbone = copy.deepcopy(self.backbone)
                self.distill_backbone.set_feedforward_grads(False)
                self.distill_backbone.set_task(None, update_grads=True)

                self.distill_backbone.eval()

                self.distill_head = copy.deepcopy(self.ssl_head)
                deactivate_requires_grad(self.distill_head)
                self.distill_head.eval()

            # reset projectors for next session
            if self.cfg.ssl.reset_projector:
                self.py_logger.info('Resetting SSL head')
                did_smth = False
                for module in self.ssl_head.modules():
                    if isinstance(module, nn.Linear):
                        did_smth = True
                        module.reset_parameters()
                    if isinstance(module, nn.BatchNorm1d):
                        module.reset_parameters()
                assert did_smth
            if self.cfg.tmcl.reset_projector:
                self.py_logger.info('Resetting TMCL head')
                did_smth = False
                for module in self.tmcl_head.modules():
                    if isinstance(module, nn.Linear):
                        did_smth = True
                        module.reset_parameters()
                    if isinstance(module, nn.BatchNorm1d):
                        module.reset_parameters()
                assert did_smth

        self.custom_evaluation_loop()

        if (
            self.continual_meta.is_last_session_epoch
            and self.cfg.end_after_session is not None
            and self.cfg.end_after_session == session
        ):
            self.py_logger.info(f'End of session {session}, stopping training.')
            self.trainer.should_stop = True

    @torch.inference_mode()
    def custom_evaluation_loop(self, eval_before_training: bool = False):
        if self.cfg.eval.disable_eval:
            return

        def log_fn(name: str, value: float | Tensor, **kwargs) -> None:
            if eval_before_training:
                if self.global_rank != 0:
                    # no-op
                    return
                self.wandb_logger.experiment.log({name: value, 'trainer/global_step': -1})
            else:
                self.log(name, value, **kwargs)

        is_eval_epoch = ((self.current_epoch + 1) % self.cfg.eval.eval_interval == 0) or (
            eval_before_training and self.cfg.eval.eval_before_training
        )
        if not any((self.continual_meta.is_last_session_epoch, is_eval_epoch)):
            return

        self.backbone.set_task(None)

        try:
            results = self.eval_module.evaluate()
            knn_correct = results['knn_correct'].float()
            labels = results['labels']
            rankme = torch.stack(results['rankme']).mean()

            # kNN results
            log_fn('eval/knn_all_top1', knn_correct.mean(), prog_bar=True)

            if self.cfg.eval.knn_per_session:
                for session_idx, session_classes in enumerate(self.session_classes):
                    session_classes_tensor = torch.tensor(session_classes)
                    feature_mask = torch.isin(labels, session_classes_tensor)
                    log_fn(
                        f'eval/knn_session{session_idx}_top1_acc',
                        knn_correct[feature_mask].mean(),
                    )

            # rankme
            log_fn('eval/test_rankme', rankme)
        except UnicodeDecodeError:
            self.py_logger.warning(' ⚠ Evaluation failed!')
            import traceback

            traceback.print_exc()

            log_fn('eval/knn_all_top1', torch.nan, prog_bar=True)
            if self.cfg.eval.knn_per_session:
                for session_idx, _ in enumerate(self.session_classes):
                    log_fn(
                        f'eval/knn_session{session_idx}_top1_acc',
                        torch.nan,
                    )
            log_fn('eval/test_rankme', torch.nan)
            return

        self.py_logger.info(' ✔ Evaluation completed.')


def upload_csv_to_wandb(cfg: Config) -> None:
    if cfg.resume_id is None:
        raise ValueError('Resume ID must be provided to upload metrics to WandB.')

    wandb_project = cfg.project_name
    if cfg.devel:
        wandb_project += '-devel'

    api = wandb.Api()
    run = api.run(f'llfs/{wandb_project}/{cfg.resume_id}')
    print(f'Uploading metrics to WandB for run llfs/{wandb_project}/{cfg.resume_id}.')

    last_logged_step = run.summary.get('_step', -1)
    print('Last logged step in WandB:', last_logged_step)

    wandb_logger = WandbLogger(
        name=cfg.name[:50],
        save_dir=cfg.work_path,
        id=cfg.resume_id,
        project=wandb_project,
        group=cfg.group or cfg.name,
        config=dataclasses.asdict(cfg),
        resume='must',
    )
    for version in (cfg.work_path / 'logs').glob('version_*'):
        print(f'Uploading metrics from version_{version}.')
        csv_path = cfg.work_path / 'logs' / version / 'metrics.csv'
        metrics_df = pd.read_csv(csv_path)
        metrics_df = metrics_df[metrics_df['step'] >= last_logged_step]

        for _, row in tqdm(metrics_df.iterrows()):
            wandb_logger.experiment.log(
                {k: v for k, v in row.to_dict().items() if not np.isnan(v)}, step=int(row['step'])
            )

    wandb_logger.experiment.finish()
    print(f'Updated https://wandb.ai/llfs/{wandb_project}/{cfg.resume_id}')


def run_tmcl(cfg: Config):
    logging.basicConfig(
        level='INFO',
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(cfg, extra={'markup': True})

    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision('high')

    (cfg.work_path / 'logs').mkdir(parents=True, exist_ok=True)

    wandb_project = cfg.project_name
    if cfg.devel:
        wandb_project += '-devel'
    # noinspection PyTypeChecker

    wandb_logger = WandbLogger(
        name=cfg.name[:50],
        save_dir=cfg.work_path,
        id=cfg.resume_id,
        project=wandb_project,
        group=cfg.group or cfg.name,
        config=dataclasses.asdict(cfg),
        resume='must' if cfg.resume_id is not None else 'never',
    )
    module = TMCLDALIModule(cfg, wandb_logger=wandb_logger, py_logger=logger)
    logger.info('Module initialized, starting trainer.')
    trainer = pl.Trainer(
        sync_batchnorm=True,
        reload_dataloaders_every_n_epochs=1,
        benchmark=not cfg.deterministic,
        deterministic=cfg.deterministic,
        max_epochs=cfg.num_epochs,
        devices=cfg.gpus_per_task,
        num_nodes=cfg.n_nodes,
        accelerator=cfg.accelerator,
        strategy='ddp_find_unused_parameters_true' if cfg.n_devices > 1 else 'auto',
        precision=cfg.precision,
        logger=[
            wandb_logger,
            CSVLogger(cfg.work_path / 'logs', name=''),
        ],
        default_root_dir=cfg.work_path,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=cfg.work_path / 'checkpoints',
                save_last=True,
                auto_insert_metric_name=False,
                every_n_epochs=cfg.checkpoint_interval,
                save_top_k=-1,
            ),
            pl.callbacks.ModelSummary(depth=3),
        ],
        # num_sanity_val_steps=2 if hparams.sanity_check else 0,
        log_every_n_steps=cfg.log_interval,
        limit_train_batches=2 if cfg.devel else 1.0,
    )
    ckpt_path = None
    if cfg.resume_from is not None:
        ckpt_path = cfg.resume_from
    if cfg.resume:
        ckpt_path = cfg.get_last_checkpoint(force=True)
    if ckpt_path:
        logger.info(f'Resuming from checkpoint {ckpt_path}.')

    trainer.fit(
        model=module,
        ckpt_path=ckpt_path,
    )

    if cfg.eval.linear_eval:
        from tmcl.config_linear import Config as LinearConfig

        linear_cfg = LinearConfig(
            data_path=cfg.data_path,
            eval_last_n_layers=cfg.eval.linear_last_n_layers,
            lr=0.1 * 1024 / 256,
            batch_size=1024,
            project_name=cfg.project_name,
            name=f'linear@{cfg.name}',
            group=cfg.group,
            continual_setup=cfg.setup,
            devel=cfg.devel,
            n_data_workers=cfg.n_data_workers,
            checkpoint_interval=50,
            eval_interval=10,
            num_epochs=100,
            eval_dataset=cfg.dataset,
            timm_model=cfg.timm_model,
            weight_decay=0.0,
            optim_momentum=0.9,
            lr_scheduler='cosine',
            l2_normalize=False,
            load_checkpoint_from=cfg.get_last_checkpoint(force=True),
            num_checkpoint_tasks=module.num_tasks,
            eval_modulation=None,
            use_bias_modulations=cfg.bias_modulations,
            seed=cfg.seed,
            deterministic=True,
            torch_compile=True,
            resume=cfg.eval.linear_resume,
        )
        for ckpt in (linear_cfg.work_path / 'checkpoints').glob('*.ckpt'):
            if 'last' in ckpt.stem:
                continue
            epoch = int(ckpt.stem.split('-')[0])
            if epoch >= 100 - 1:
                logger.info('Final linear evaluation checkpoint found, done!')
                return
        _run_linear(linear_cfg)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    run_tmcl(simple_parsing.parse(Config))
