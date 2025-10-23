import bisect
import copy
import dataclasses
import itertools
import logging
import re
import statistics
from collections.abc import Iterable, Mapping
from functools import cached_property, partial
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
from timm.models import ConvNeXt, VisionTransformer
from torch import Tensor, nn
from torch.nn import Identity
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet
from tqdm import tqdm

from represent.knn import knn_predict
from represent.neural_collapse import class_distance_normalized_variance
from represent.rankme import rankme
from represent.utils import to_per_class_list
from tmcl.config_tmcl import Config
from tmcl.data import IncrementalSessionData, SessionBatch
from tmcl.data_setups import (
    Data,
    prepare_data,
)
from tmcl.datasets.kornia_ssl import (
    build_ssl_transforms,
)
from tmcl.datasets.kornia_supervised import (
    build_eval_transform,
    build_supervised_transform,
)
from tmcl.hints import (
    BatchId,
    ClassBatch,
    FeatureBatch,
    ImageBatch,
)
from tmcl.main_linear import run_linear
from tmcl.nn.convit import ConVit
from tmcl.nn.heads import TaskSpecificReadout
from tmcl.nn.losses.barlow import BarlowTwinsLoss
from tmcl.nn.losses.contrastive_opl import contrastive_opl
from tmcl.nn.losses.mv_barlow import GhoshBarlowLoss, MultiViewBarlowTwinsLoss
from tmcl.nn.losses.pnr import simsiam_loss_func
from tmcl.nn.losses.simclr import manual_simclr_loss_func, simclr_loss_func
from tmcl.nn.losses.supcon import supcon_positive_mask
from tmcl.nn.modulations import build_tm_model
from tmcl.optim.lars import LARS
from tmcl.optim.warmup_cosine import warmup_cosine
from tmcl.phase import Phase
from tmcl.utils import (
    disable_optimizer_step_increment,
    enable_optimizer_step_increment,
    get_slurm_stdout_path,
)


class TMCLModule(pl.LightningModule):
    cfg: Config
    data: Data
    wandb_logger: WandbLogger
    py_logger: logging.Logger

    def __init__(
        self,
        cfg: Config,
        data: Data,
        *,
        wandb_logger: WandbLogger,
        py_logger: logging.Logger | None = None,
    ):
        cfg.verify()
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.data = data
        self.wandb_logger = wandb_logger
        self.py_logger = py_logger if py_logger is not None else logging.getLogger(__name__)

        self.session_data = IncrementalSessionData(
            sessions=data.sessions,
            labeled_setup=cfg.class_learner.labeled_setup,
            batch_size_per_gpu=cfg.batch_size_per_gpu,
            workers=cfg.n_data_workers,
            unsup_replacement=cfg.unsup_replacement,
            sup_replacement=cfg.sup_replacement,
            last_batch=cfg.last_batch,
        )

        match self.cfg.class_learner.labeled_setup:
            case 'onevsall':
                self.num_tasks = sum(data.classes_per_dataset.values())
                self.num_classes = 1
                self.py_logger.info(
                    f'Class-incremental learning setup, num_tasks={self.num_tasks}, num_classes={self.num_classes}'
                )
            case 'allvsall':
                self.num_tasks = 1
                self.num_classes = data.num_classes
                self.py_logger.info(
                    f'All-vs-all learning setup, num_tasks=1, num_classes={self.num_classes}'
                )
            case _:
                raise ValueError(f'Unknown labeled_setup {cfg.class_learner.labeled_setup}')

        self.backbone = build_tm_model(
            cfg.timm_model,
            num_tasks=self.num_tasks if cfg.class_learner.use_task_modulations else 0,
            image_size=data.image_size,
            patch_size=data.patch_size,
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
                    normalize_eps=cfg.ssl.normalize_eps,
                    lambda_param=cfg.ssl.barlow_lambda,
                    scale_loss=cfg.ssl.barlow_scale_loss,
                )
            case 'ghosh':
                self.ssl_loss = GhoshBarlowLoss(
                    normalize_eps=cfg.ssl.normalize_eps,
                    lambda_param=cfg.ssl.barlow_lambda,
                    scale_loss=cfg.ssl.barlow_scale_loss,
                )
            case 'barlow':
                if cfg.cons_num_views != 2:
                    raise ValueError(f'Barlow Twins requires 2 views, but got {cfg.cons_num_views}')
                self.ssl_loss = BarlowTwinsLoss(
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
            if self.cfg.resume:
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
                    nn.Linear(self.backbone.output_dim, len(s.current_classes)),
                )
                for s in data.sessions
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

        self.ssl_transform = build_ssl_transforms(
            data.dataset,
            cfg.cons_augments,
        )
        self.sup_transform = (
            build_supervised_transform(data.dataset)
            if self.cfg.class_learner.augment in ('sup', 'sup_fixed')
            else (
                self.ssl_transform
                if not isinstance(self.ssl_transform, tuple)
                else self.ssl_transform[0]
            )
        )
        self.eval_transform = build_eval_transform(data.dataset)

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

    @override
    def forward(self, x: ImageBatch) -> FeatureBatch:
        return self.backbone(x)

    def session_idx(self, epoch: int | None = None) -> int:
        epoch = epoch if epoch is not None else self.current_epoch
        return bisect.bisect_right(
            list(itertools.accumulate(s.num_epochs for s in self.data.sessions)), epoch
        )

    @property
    def session_step(self) -> int:
        """
        Steps executed in the current session.
        """
        return self.global_step - sum(self.num_session_batches[: self.session_idx()])

    @override
    def on_train_start(self) -> None:
        # Basically lightning's sanity check
        if self.cfg.eval.eval_before_training:
            self.custom_evaluation_loop(eval_before_training=True)

    @override
    def train_dataloader(self) -> Iterable[SessionBatch]:
        return self.session_data.train_dataloader(self.session_idx(), phase=self.phase())

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

        tl_optim = optim_fn(get_pg(tl_param_groups))
        cons_optim = optim_fn(get_pg(cons_param_groups))

        return tl_optim, cons_optim

    @override
    def on_train_epoch_start(self) -> None:
        self.log(
            'progress/current_session',
            self.session_idx(),
            prog_bar=True,
            sync_dist=False,
        )
        self.log('progress/current_phase', self.phase().value, prog_bar=True, sync_dist=False)

    @torch.no_grad()
    def update_weight_decay(self, consolidation_optim: Optimizer) -> None:
        weight_decay = cosine_schedule(
            self.trainer.global_step,
            self.num_total_batches,
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
        self.log('progress/session_step', self.session_step, prog_bar=False, sync_dist=False)
        current_phase: Final[Phase] = self.phase()
        session_idx = self.session_idx()
        session = self.session_data.sessions[session_idx]
        phase_epochs = session.epochs[self.phase_idx()][1]
        for param_group in tl_optim.param_groups:
            if current_phase == Phase.TASK_LEARNING or (
                current_phase == Phase.PRETRAIN and self.cfg.class_learner.during_pretraining
            ):
                previous_steps = 0
                for phase, _ in session.epochs:
                    if phase == current_phase:
                        break
                    if (
                        phase == Phase.PRETRAIN
                        and not self.cfg.class_learner.during_pretraining
                        or phase == Phase.CONSOLIDATION
                    ):
                        previous_steps += self.num_session_phase_batches[session_idx][phase]

                param_group['lr'] = warmup_cosine(
                    self.session_step + 1 - previous_steps,
                    peak_lr=self.cfg.optim.tl_lr * (self.cfg.batch_size / 256),
                    num_steps=self.num_session_phase_batches[session_idx][current_phase],
                    warmup_steps=int(
                        (self.cfg.optim.warmup_epochs / phase_epochs)
                        * self.num_session_phase_batches[session_idx][current_phase]
                    ),
                    end_lr=self.cfg.optim.tl_min_lr,
                    start_lr=self.cfg.optim.tl_start_lr,
                )
            else:
                param_group['lr'] = 0.0
        # noinspection PyUnboundLocalVariable
        self.log('progress/lr_tl', param_group['lr'], prog_bar=False, sync_dist=False)

        for param_group in consolidation_optim.param_groups:
            if self.cfg.optim.cons_lr_reset:
                previous_steps = 0
                for phase, _ in session.epochs:
                    if phase == current_phase:
                        break
                    if phase not in (Phase.PRETRAIN, Phase.CONSOLIDATION):
                        previous_steps += self.num_session_phase_batches[session_idx][phase]

                # relative to the number of cons./pret. steps
                current_step = (
                    self.session_step + 1 - previous_steps
                    if current_phase in (Phase.PRETRAIN, Phase.CONSOLIDATION)
                    else 0
                )
                num_steps = sum(
                    steps
                    for phase, steps in self.num_session_phase_batches[session_idx].items()
                    if phase in (Phase.PRETRAIN, Phase.CONSOLIDATION)
                )
                ssl_epochs = sum(
                    epochs
                    for phase, epochs in self.data.sessions[session_idx].epochs
                    if phase in (Phase.PRETRAIN, Phase.CONSOLIDATION)
                )
                warmup_steps = int((self.cfg.optim.warmup_epochs / ssl_epochs) * num_steps)
            else:
                current_step = self.global_step + 1
                num_steps = self.num_total_batches
                warmup_steps = int(
                    (self.cfg.optim.warmup_epochs / self.trainer.max_epochs)
                    * self.num_total_batches
                )

            param_group['lr'] = warmup_cosine(
                current_step,
                peak_lr=self.cfg.optim.cons_lr * (self.cfg.batch_size / 256),
                num_steps=num_steps,
                warmup_steps=warmup_steps,
                end_lr=self.cfg.optim.cons_min_lr,
                start_lr=self.cfg.optim.cons_start_lr,
            )
        self.log('progress/lr_cons', param_group['lr'], prog_bar=False, sync_dist=False)

    def phase(self, epoch: int | None = None) -> Phase:
        return self.data.sessions[self.session_idx(epoch)].epochs[self.phase_idx(epoch)][0]

    def phase_idx(self, epoch: int | None = None) -> int:
        epoch = epoch if epoch is not None else self.current_epoch
        session_idx = self.session_idx(epoch)
        prev_epochs = sum(s.num_epochs for s in self.data.sessions[:session_idx])

        e = epoch - prev_epochs
        for idx, (_, epochs) in enumerate(self.data.sessions[session_idx].epochs):
            if e < epochs:
                return idx
            e -= epochs
        raise ValueError(
            f'Epoch {epoch} is out of bounds for {self.data.sessions[session_idx].epochs}'
        )

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
        current_session: int,
        num_views: int | None = None,
    ) -> Integer[Tensor, 'num_views * batch']:
        if num_views is None:
            num_views = self.cfg.cons_num_views

        seen_classes_tensor = torch.tensor(
            self.session_data.sessions[current_session].seen_classes,
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

    @torch.no_grad()
    def apply_ssl_transform(
        self, images: Float[Tensor, 'batch channels height width'], num_views: int
    ) -> Float[Tensor, 'batch*num_views channels height width']:
        if isinstance(self.ssl_transform, tuple):
            aug_a, aug_b = self.ssl_transform
            a_views = num_views // 2
            b_views = num_views - a_views
            cons_images_a = aug_a(images.repeat(a_views, 1, 1, 1))
            cons_images_b = aug_b(images.repeat(b_views, 1, 1, 1))
            return torch.cat([cons_images_a, cons_images_b], dim=0)
        else:
            return self.ssl_transform(images.repeat(num_views, 1, 1, 1))

    @override
    def training_step(
        self,
        batch: SessionBatch,
        batch_index: BatchId,
    ) -> None:
        current_phase = self.phase()
        current_session = self.session_idx()

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

                    sup_images: ImageBatch = self.sup_transform(batch.task.images)
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

                    sup_images: ImageBatch = self.sup_transform(batch.task.images)
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

            cons_images = self.apply_ssl_transform(batch.cons.images, self.cfg.cons_num_views)

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

                # self.log(
                #     'train_cons/ssl_loss',
                #     ssl_loss,
                #     prog_bar=True,
                #     sync_dist=True,
                #     batch_size=batch.cons.batch_size,
                # )

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
                        distill_loss = distill_loss + pnr_loss

            if do_task_inv:
                tmcl_images = cons_images
                if self.cfg.tmcl.single_augment:
                    tmcl_images = cons_images[: batch.cons.batch_size, :, :, :].repeat(
                        self.cfg.cons_num_views, 1, 1, 1
                    )
                tmcl_tasks: Integer[Tensor, 'num_views * batch'] = self.sample_tasks(
                    batch_size=batch.cons.batch_size, current_session=current_session
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
            else:
                tmcl_loss = 0.0

            sup_head_loss = 0.0
            if self.cfg.suphead.enable_sup_head:
                for s, _ in enumerate(self.data.sessions):
                    if s == current_session:
                        activate_requires_grad(self.suphead_session_heads[s])
                    else:
                        deactivate_requires_grad(self.suphead_session_heads[s])

                suphead_images = self.apply_ssl_transform(
                    batch.task.images, num_views=self.cfg.cons_num_views
                )
                self.backbone.set_task(None, update_grads=False)
                suphead_embeds = self.backbone(suphead_images)
                suphead_logits = self.suphead_session_heads[current_session](suphead_embeds)
                suphead_labels = batch.task.labels.repeat(self.cfg.cons_num_views)
                # remap to 0..(|C|-1)
                current_classes = self.data.sessions[current_session].current_classes
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

                supcon_images = self.apply_ssl_transform(
                    batch.task.images, num_views=self.cfg.cons_num_views
                )
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

    def is_last_session_epoch(self, epoch: int | None = None) -> bool:
        epoch = epoch if epoch is not None else self.current_epoch
        return epoch == sum(s.num_epochs for s in self.data.sessions[: self.session_idx() + 1]) - 1

    @override
    def on_train_epoch_end(self) -> None:
        is_not_last_epoch = self.current_epoch < self.trainer.max_epochs - 1
        is_last_session_epoch = self.is_last_session_epoch()

        if is_last_session_epoch and is_not_last_epoch:
            if self.cfg.distill.algo != 'none':
                self.py_logger.info(
                    f'End of session {self.session_idx()}, updating distillation backbone.'
                )
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
            is_last_session_epoch
            and self.cfg.end_after_session is not None
            and self.cfg.end_after_session == self.session_idx()
        ):
            self.py_logger.info(f'End of session {self.session_idx()}, stopping training.')
            self.trainer.should_stop = True

    @torch.inference_mode()
    def custom_evaluation_loop(self, eval_before_training: bool = False):
        if self.cfg.eval.disable_eval:
            return

        if self.global_rank != 0:
            return

        def log_fn(name: str, value: float | Tensor, **kwargs) -> None:
            if eval_before_training:
                self.wandb_logger.experiment.log({name: value, 'trainer/global_step': -1})
            else:
                self.log(name, value, **kwargs)

        is_eval_epoch = ((self.current_epoch + 1) % self.cfg.eval.eval_interval == 0) or (
            eval_before_training and self.cfg.eval.eval_before_training
        )
        if not any((self.is_last_session_epoch(), is_eval_epoch)):
            return

        def get_samples(ds: Dataset) -> tuple[ImageBatch, ClassBatch]:
            dl = DataLoader(
                ds,
                batch_size=self.cfg.batch_size_per_gpu,
                shuffle=False,
                num_workers=self.cfg.n_data_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=False,
            )
            _images, _classes = [], []
            for image_batch, class_batch in dl:
                _images.append(image_batch)
                _classes.append(class_batch)

            _images = torch.cat(_images)
            _classes = torch.cat(_classes).squeeze()
            return _images, _classes

        self.backbone.eval()

        for dataset in self.data.eval_datasets:
            if self.cfg.eval.eval_valid:
                splits: tuple[str, ...] = ('train', 'valid', 'test')
            else:
                splits: tuple[str, ...] = ('train', 'test')
            images: dict[str, ImageBatch] = {}
            classes: dict[str, ClassBatch] = {}

            ################
            # Load samples
            ################
            for split in splits:
                ds_by_split = {
                    'train': self.data.eval_datasets[dataset]['train_full']
                    if self.cfg.eval.eval_on_full_train
                    else self.data.eval_datasets[dataset]['train'],
                    'valid': self.data.eval_datasets[dataset]['valid'],
                    'test': self.data.eval_datasets[dataset]['test'],
                }
                images[split], classes[split] = get_samples(ds_by_split[split])

            images = {k: v.to(self.device) for k, v in images.items()}
            classes = {k: v.to(self.device) for k, v in classes.items()}

            def forward(
                images: ImageBatch, modulation: int | None, *, transform=self.eval_transform
            ) -> FeatureBatch:
                _features = []
                for image_batch in images.split(self.cfg.batch_size_per_gpu):
                    self.backbone.set_task(
                        torch.full((len(image_batch),), fill_value=modulation, device=self.device)
                        if modulation is not None
                        else None,
                        update_grads=False,
                    )
                    _features.append(self.backbone(transform(image_batch)))
                return torch.cat(_features)

            if self.data.incrementality == 'task':
                # task-incremental learning metrics
                self.py_logger.info('Evaluating task-incremental learning metrics...')
                for session in tqdm(self.data.sessions):
                    for split in splits:
                        if split == 'train':
                            continue
                        session_classes_tensor = torch.tensor(
                            session.current_classes, device=self.device
                        )
                        relabeling = {c: i for i, c in enumerate(session.current_classes)}

                        def relabel(label_tensor):
                            new_label_tensor = torch.empty_like(label_tensor)
                            for original, new in relabeling.items():  # noqa: B023
                                new_label_tensor[label_tensor == original] = new
                            return new_label_tensor

                        feature_mask = torch.isin(classes[split], session_classes_tensor)
                        support_mask = torch.isin(classes['train'], session_classes_tensor)
                        query_labels = relabel(classes[split][feature_mask])
                        support_labels = relabel(classes['train'][support_mask])

                        if self.cfg.class_learner.use_task_modulations:
                            mod_session_features = forward(images[split][feature_mask], session.idx)
                            mod_session_support = forward(
                                images['train'][support_mask], session.idx
                            )
                            mod_knn_preds = knn_predict(
                                feature=mod_session_features,
                                feature_bank=mod_session_support,
                                feature_labels=support_labels,
                                num_classes=len(session_classes_tensor),
                                knn_k=20,
                                knn_t=0.07,
                                normalize_feature_bank=True,
                            )
                            aware_preds_top1 = mod_knn_preds[:, 0].squeeze()
                            aware_top1_acc = aware_preds_top1.eq(query_labels).float().mean()
                            log_fn(
                                f'eval_{dataset}_{split}/mod_session{session.idx}_top1_acc',
                                aware_top1_acc,
                            )

                        unmod_session_features = forward(images[split][feature_mask], None)
                        unmod_session_support = forward(images['train'][support_mask], None)
                        unmod_knn_preds = knn_predict(
                            feature=unmod_session_features,
                            feature_bank=unmod_session_support,
                            feature_labels=support_labels,
                            num_classes=len(session_classes_tensor),
                            knn_k=20,
                            knn_t=0.07,
                            normalize_feature_bank=True,
                        )
                        unmod_session_top1_preds = unmod_knn_preds[:, 0].squeeze()
                        unmod_session_top1_acc = (
                            unmod_session_top1_preds.eq(query_labels).float().mean()
                        )
                        log_fn(
                            f'eval_{dataset}_{split}/unmod_session{session.idx}_top1_acc',
                            unmod_session_top1_acc,
                        )
            else:
                # class-incremental learning metrics
                self.py_logger.info('Evaluating class-incremental learning metrics.')

                evaluated_modulations = (None,) + self.cfg.eval.evaluated_tasks

                ################
                # Compute features
                ################
                features = {}
                for split, modulation in (
                    pbar := tqdm(list(itertools.product(splits, evaluated_modulations)))
                ):
                    pbar.set_description(f'{split}@mod{modulation}')
                    features[(split, modulation)] = forward(images[split], modulation)

                ################
                # Compute metrics
                ################

                for (split, modulation), hs in features.items():
                    log_fn(f'eval_{dataset}_{split}/{modulation}/rankme', rankme(hs).item())

                for (split, modulation), xs in features.items():
                    cdnv = class_distance_normalized_variance(
                        features=to_per_class_list(
                            xs,
                            classes[split],
                        ),
                    )
                    log_fn(
                        f'eval_{dataset}_{split}/{modulation}/cdnv',
                        cdnv.item(),
                    )

                # nearest neighbor estimation based on training samples
                for (split, modulation), hs in tqdm(
                    features.items(), total=sum(1 for s, _ in features if s != 'train')
                ):
                    if split == 'train':
                        continue
                    if modulation not in (None, 0):
                        # No need to evaluate full kNN on 1-vs-all modulations
                        continue

                    knn_preds_full = knn_predict(
                        feature=hs,
                        feature_bank=features[('train', modulation)],
                        feature_labels=classes['train'],
                        num_classes=sum(self.data.classes_per_dataset.values()),
                        knn_k=20,
                        knn_t=0.07,
                        normalize_feature_bank=True,
                    )
                    top1_preds = knn_preds_full[:, 0].squeeze()
                    all_top1_acc = top1_preds.eq(classes[split]).float().mean()
                    log_fn(f'eval_{dataset}_{split}/{modulation}/knn_all_top1_acc', all_top1_acc)

                    top5_preds = knn_preds_full[:, :5]
                    all_top5_acc = (
                        top5_preds.eq(classes[split].unsqueeze(1)).any(dim=1).float().mean()
                    )
                    log_fn(f'eval_{dataset}_{split}/{modulation}/knn_all_top5_acc', all_top5_acc)

                    if self.cfg.eval.knn_per_session:
                        eval_sessions = self.data.eval_sessions
                        session_agnostic_accs = []
                        for session_idx, session_classes in enumerate(eval_sessions):
                            session_classes_tensor = torch.tensor(
                                session_classes, device=self.device
                            )
                            feature_mask = torch.isin(classes[split], session_classes_tensor)

                            # task-agnostic kNN
                            agnostic_preds = top1_preds[feature_mask]
                            agnostic_top1_acc = (
                                agnostic_preds.eq(classes[split][feature_mask]).float().mean()
                            )
                            session_agnostic_accs.append(agnostic_top1_acc.item())
                            log_fn(
                                f'eval_{dataset}_{split}/{modulation}/knn_session{session_idx}_agnostic_top1_acc',
                                agnostic_top1_acc,
                            )

                        log_fn(
                            f'eval_{dataset}_{split}/{modulation}/knn_average_agnostic_top1_acc',
                            statistics.mean(session_agnostic_accs),
                        )
                # compute one-vs-all statistics
                if self.cfg.eval.eval_onevsall:
                    for m in self.cfg.eval.eval_onevsall:
                        m_features = forward(images['test'], m)
                        opl_loss, opl_stats = contrastive_opl(
                            m_features, (classes['test'] == m).long()
                        )
                        log_fn(f'eval_{dataset}_test/{m}/onevsall', opl_loss)
                        for k, v in opl_stats.items():
                            log_fn(f'eval_{dataset}_test/{m}/onevsall_{k}', v)

        self.py_logger.info(' ✔ Evaluation completed.')

    @cached_property
    def num_session_phase_batches(self) -> tuple[dict[Phase, int], ...]:
        session_batches = []
        for sdef in self.data.sessions:
            sbatches = {}
            for phase, epochs in sdef.epochs:
                sbatches[phase] = (
                    len(self.session_data.build_train_dataloader(sdef.idx, phase)) * epochs
                )
            session_batches.append(sbatches)
        self.py_logger.info(f'Sessions: {self.data.sessions}')
        self.py_logger.info(f'Session batches: {session_batches}')
        return tuple(session_batches)

    @cached_property
    def num_session_batches(self) -> tuple[int, ...]:
        return tuple(sum(sbatches.values()) for sbatches in self.num_session_phase_batches)

    @property
    def num_total_batches(self):
        return sum(self.num_session_batches)


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

    data = prepare_data(cfg)

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
    if (slurm_path := get_slurm_stdout_path()) is not None:
        wandb_logger.experiment.config.update(
            {'slurm_stdout': str(slurm_path)}, allow_val_change=True
        )
    try:
        module = TMCLModule(cfg, data, wandb_logger=wandb_logger, py_logger=logger)
        logger.info('Module initialized, starting trainer.')
        trainer = pl.Trainer(
            reload_dataloaders_every_n_epochs=1,
            benchmark=not cfg.deterministic,
            deterministic=cfg.deterministic,
            max_epochs=sum(s.num_epochs for s in data.sessions),
            devices=cfg.gpus_per_task,
            num_nodes=cfg.n_nodes,
            accelerator=cfg.accelerator,
            strategy='ddp' if cfg.n_devices > 1 else 'auto',
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
        trainer.fit(
            model=module,
            ckpt_path=cfg.get_last_checkpoint(force=True) if cfg.resume else None,
        )

        if cfg.setup.startswith('c100'):
            dataset = 'cifar100'
        else:
            dataset = cfg.setup.split('/')[0]

        # finished training, let's do linear eval
        if cfg.eval.linear_eval:
            from tmcl.config_linear import Config as LinearConfig

            linear_cfg = LinearConfig(
                eval_last_n_layers=cfg.eval.linear_last_n_layers,
                lr=cfg.eval.linear_lr * 1024 / 256,
                batch_size=cfg.eval.linear_batch_size,
                project_name=cfg.project_name,
                name=f'linear@{cfg.name}',
                group=cfg.group,
                continual_setup=cfg.setup if cfg.setup.split('/')[1] != 'offline' else None,  # (!)
                devel=cfg.devel,
                n_data_workers=cfg.n_data_workers,
                checkpoint_interval=50,
                eval_interval=10,
                num_epochs=100,
                eval_dataset=dataset,
                timm_model=cfg.timm_model,
                train_augmentations=False,
                weight_decay=0.0,
                optim_momentum=0.9,
                lr_scheduler=cfg.eval.linear_lr_scheduler,
                l2_normalize=False,
                load_checkpoint_from=cfg.get_last_checkpoint(force=True),
                num_checkpoint_tasks=module.num_tasks,
                eval_modulation=None,
                use_bias_modulations=cfg.bias_modulations,
                seed=cfg.seed,
                deterministic=True,
                torch_compile=True,
                resume=cfg.eval.linear_resume,  # temporary
            )
            for ckpt in (linear_cfg.work_path / 'checkpoints').glob('*.ckpt'):
                if 'last' in ckpt.stem:
                    continue
                epoch = int(ckpt.stem.split('-')[0])
                if epoch >= 100 - 1:
                    logger.info('Final linear evaluation checkpoint found, done!')
                    return
            run_linear(linear_cfg)
    finally:
        if not cfg.devel:
            wandb_logger.experiment.finish(quiet=False)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    run_tmcl(simple_parsing.parse(Config))
