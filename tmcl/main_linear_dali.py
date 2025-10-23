import dataclasses
import json
import logging
import os
import random
import time
import typing

import pytorch_lightning as pl
import simple_parsing
import torch
import torch.nn.functional as F
from jaxtyping import Float
from nvidia.dali import Pipeline
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIRaggedIterator
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from rich.logging import RichHandler
from torch import Tensor, nn
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import Accuracy, MetricCollection

from tmcl.config_linear import Config
from tmcl.datasets.dali_imagenet import (
    ImageNet100Dataset,
    eval_imagenet_pipeline,
    sup_imagenet_pipeline,
)
from tmcl.nn.modulations import build_tm_model


class LinearModule(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.console_logger = logging.getLogger(__name__)

        match cfg.eval_dataset:
            case 'imagenet100':
                image_size = 224
                patch_size = 16
                num_classes = 100
            case _:
                raise ValueError(f'Unknown dataset {cfg.eval_dataset}')

        match self.cfg.continual_setup:
            case None:
                self.sessions = None
            case 's5':
                rng = random.Random(cfg.seed)
                class_order = list(range(100))
                rng.shuffle(class_order)

                self.sessions = [
                    class_order[session * 20 : (session + 1) * 20] for session in range(5)
                ]
            case _:
                raise ValueError(f'Unknown continual setup {cfg.continual_setup}')

        self.backbone = build_tm_model(
            cfg.timm_model,
            num_tasks=cfg.num_checkpoint_tasks,
            image_size=image_size,
            patch_size=patch_size,
            has_bias=cfg.use_bias_modulations,
            pretrained=False,
        )
        feature_dim = self.backbone.output_dim
        if self.cfg.eval_last_n_layers is not None:
            feature_dim *= self.cfg.eval_last_n_layers

        self.backbone.set_feedforward_grads(False)
        self.backbone.set_task(cfg.eval_modulation, update_grads=True)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

        if self.cfg.load_checkpoint_from is not None:
            _ckpt = torch.load(self.cfg.load_checkpoint_from, weights_only=False)
            self.backbone.load_state_dict(
                {
                    k.removeprefix('backbone.').removeprefix('_orig_mod.'): v
                    for k, v in _ckpt['state_dict'].items()
                    if k.startswith('backbone.')
                }
            )
            self.console_logger.info(f'Loaded checkpoint from {self.cfg.load_checkpoint_from}.')

        if cfg.torch_compile:
            self.backbone = torch.compile(self.backbone)
            self.classifier = torch.compile(self.classifier)

        self.train_metrics = MetricCollection(
            {
                f'train_{self.cfg.eval_dataset}_top1': Accuracy(
                    task='multiclass', num_classes=num_classes, top_k=1
                ),
                f'train_{self.cfg.eval_dataset}_top5': Accuracy(
                    task='multiclass', num_classes=num_classes, top_k=5
                ),
            }
        )
        self.eval_metrics = MetricCollection(
            {
                f'eval_{self.cfg.eval_dataset}_top1': Accuracy(
                    task='multiclass', num_classes=num_classes, top_k=1, sync_on_compute=False
                ),
                f'eval_{self.cfg.eval_dataset}_top5': Accuracy(
                    task='multiclass', num_classes=num_classes, top_k=5, sync_on_compute=False
                ),
            }
        )
        self.session_eval_metrics = []
        if self.sessions is not None:
            self.session_eval_metrics = nn.ModuleList(
                [
                    Accuracy(
                        task='multiclass',
                        num_classes=num_classes,
                        top_k=1,
                        average='micro',
                        sync_on_compute=False,
                    )
                    for _ in self.sessions
                ]
            )

        # noinspection PyTypeChecker
        self.save_hyperparameters(dataclasses.asdict(cfg))

    def setup(self, stage: str) -> None:
        if stage in ('test', 'predict'):
            self.eval_dataset = ImageNet100Dataset(
                tar_root=self.cfg.data_path / 'imagenet100',
                split='val',
                n_procs=self.cfg.n_data_workers,
                supervised_frac=1.0,
                copy=False,
            )
            self.idx2labels = torch.tensor(
                self.eval_dataset.labels, device=self.device, dtype=torch.int64
            )
            return

        assert stage == 'fit', f'Unknown stage {stage}'
        match self.cfg.eval_dataset:
            case 'imagenet100':
                # # Check if SLURM environment and not local_rank 0 via
                # if os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_LOCALID']) != 0:
                #     while True:
                #         time.sleep(60)
                #         self.train_dataset = ImageNet100Dataset(
                #             tar_root=self.cfg.data_path / 'imagenet100',
                #             split='train',
                #             n_procs=self.cfg.n_data_workers,
                #             supervised_frac=self.cfg.labeled_frac,
                #             seed=self.cfg.seed,
                #             copy=False,
                #         )
                #         self.eval_dataset = ImageNet100Dataset(
                #             tar_root=self.cfg.data_path / 'imagenet100',
                #             split='val',
                #             n_procs=self.cfg.n_data_workers,
                #             supervised_frac=1.0,
                #             copy=False,
                #         )
                #         if len(self.train_dataset) >= 130_000 and len(self.eval_dataset) >= 5000:
                #             break
                #         else:
                #             self.console_logger.warning(
                #                 f'Missing samples {len(self.train_dataset)=}, {len(self.eval_dataset)=}, retrying...'
                #             )

                self.train_dataset = ImageNet100Dataset(
                    tar_root=self.cfg.data_path / 'imagenet100',
                    split='train',
                    n_procs=self.cfg.n_data_workers,
                    supervised_frac=self.cfg.labeled_frac,
                    seed=self.cfg.seed,
                )
                self.eval_dataset = ImageNet100Dataset(
                    tar_root=self.cfg.data_path / 'imagenet100',
                    split='val',
                    n_procs=self.cfg.n_data_workers,
                    supervised_frac=1.0,
                )
            case _:
                raise ValueError('Dataset not found.')

    def train_dataloader(self):
        pipeline = sup_imagenet_pipeline(
            files=self.train_dataset.files,
            labels=self.train_dataset.labels,
            device=self.cfg.accelerator,
            batch_size=self.cfg.batch_size_per_gpu,
            num_threads=self.cfg.n_data_workers,
            shard_id=self.trainer.global_rank,
            num_shards=self.trainer.world_size,
            seed=self.cfg.seed,  # epoch reset via fn.readers.file
        )
        pipeline = typing.cast(Pipeline, pipeline)
        pipeline.build()

        return DALIGenericIterator(
            pipeline,
            output_map=['images', 'labels'],
            reader_name='Reader',
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def test_dataloader(self):
        pipeline = eval_imagenet_pipeline(
            files=self.eval_dataset.files,
            labels=self.eval_dataset.labels,
            device=self.cfg.accelerator,
            batch_size=self.cfg.batch_size_per_gpu,
            num_threads=self.cfg.n_data_workers,
            shard_id=0,
            num_shards=1,
            seed=self.cfg.seed,  # epoch reset via fn.readers.file
        )
        pipeline = typing.cast(Pipeline, pipeline)
        pipeline.build()

        return DALIRaggedIterator(
            pipeline,
            output_map=['images', 'idxs'],
            output_types=[
                DALIRaggedIterator.DENSE_TAG,
                DALIRaggedIterator.DENSE_TAG,
            ],
            reader_name='Reader',
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        match self.cfg.optim_algo:
            case 'sgd':
                optimizer = torch.optim.SGD
            case _:
                raise ValueError(f'Unknown optimizer {self.cfg.optim_algo}')

        optimizer = optimizer(
            self.classifier.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.optim_momentum,
        )
        match self.cfg.lr_scheduler:
            case 'none':
                return optimizer
            case 'step':
                scheduler = MultiStepLR(
                    optimizer, self.cfg.lr_decay_steps, gamma=self.cfg.lr_decay_gamma
                )
            case 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.cfg.num_epochs, eta_min=0.0
                )
            case _:
                raise ValueError(f'Unknown scheduler {self.cfg.lr_scheduler}')
        return [optimizer], [scheduler]

    def forward(self, images: Float[Tensor, 'batch channel height width']):
        with torch.no_grad():
            if self.cfg.eval_last_n_layers is not None:
                _, intermediates = self.backbone(images, output_layers=True)
                features = torch.cat(intermediates[-self.cfg.eval_last_n_layers :], dim=-1)
            else:
                features = self.backbone(images)
            if self.cfg.l2_normalize:
                features = F.normalize(features, dim=-1, p=2)
        logits = self.classifier(features)
        return {'logits': logits, 'features': features}

    def shared_step(self, images, targets):
        logits = self(images)['logits']
        loss = F.cross_entropy(logits, targets)
        return loss, logits

    def training_step(self, batch, batch_idx):
        self.backbone.eval()
        images, targets = batch[0]['images'], batch[0]['labels'].squeeze()
        loss, logits = self.shared_step(images, targets)
        self.log('train_loss', loss, on_epoch=False, sync_dist=True)
        train_metrics = self.train_metrics(logits, targets)
        self.log_dict(train_metrics)
        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, indices = batch[0]['images'], batch[0]['idxs'].squeeze()
        targets = self.idx2labels[indices]
        loss, logits = self.shared_step(images, targets)
        self.eval_metrics.update(logits, targets)

        for session, session_m in enumerate(self.session_eval_metrics):
            session_class_ids = self.sessions[session]

            mask = torch.isin(targets, torch.tensor(session_class_ids, device=targets.device))
            if not mask.any():
                continue

            session_m.update(logits[mask], targets[mask])

        self.log(f'test_{self.cfg.eval_dataset}_loss', loss, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.eval_metrics.compute(), prog_bar=True)
        if self.session_eval_metrics:
            self.log_dict(
                {
                    f'eval_{self.cfg.eval_dataset}_session_{i}_top1': m.compute()
                    for i, m in enumerate(self.session_eval_metrics)
                }
            )
            for m in self.session_eval_metrics:
                m.reset()
        self.eval_metrics.reset()


def _run_linear(cfg: Config):
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
        resume='must' if cfg.resume else 'never',
    )

    try:
        module = LinearModule(cfg)
        logger.info('Module initialized, starting trainer.')
        trainer = pl.Trainer(
            benchmark=not cfg.deterministic,
            deterministic=cfg.deterministic,
            max_epochs=cfg.num_epochs if not cfg.devel else 1,
            devices=cfg.n_devices,
            num_nodes=cfg.n_nodes,
            accelerator=cfg.accelerator,
            strategy='ddp_find_unused_parameters_true' if cfg.n_devices > 1 else 'auto',
            sync_batchnorm=cfg.sync_batchnorm,
            precision=cfg.precision,
            logger=[
                CSVLogger(cfg.work_path / 'logs', name=''),
            ],
            default_root_dir=cfg.work_path,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=cfg.work_path / 'checkpoints',
                    save_last=True,
                    auto_insert_metric_name=False,
                    every_n_epochs=10,
                    save_top_k=0,
                ),
                pl.callbacks.ModelSummary(depth=3),
            ],
            num_sanity_val_steps=0,
            log_every_n_steps=cfg.log_interval,
            # limit_train_batches=2 if cfg.devel else 1.0,
        )
        trainer.fit(
            model=module,
            ckpt_path=cfg.get_last_checkpoint(force=True) if cfg.resume else None,
        )

        del trainer

        if os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_LOCALID']) != 0:
            logger.info('Skipping test (single GPU).')
            return

        logger.info('Starting test phase.')
        module.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        module.to(device)

        test_trainer = pl.Trainer(
            benchmark=not cfg.deterministic,
            deterministic=cfg.deterministic,
            devices=1,
            num_nodes=1,
            strategy=SingleDeviceStrategy(device),
            accelerator=cfg.accelerator,
            precision=cfg.precision,
            logger=[
                wandb_logger,
                CSVLogger(cfg.work_path / 'logs', name='test'),
            ],
            default_root_dir=cfg.work_path,
            callbacks=[
                pl.callbacks.ModelSummary(depth=3),
            ],
            num_sanity_val_steps=0,
            log_every_n_steps=cfg.log_interval,
        )
        results = test_trainer.test(
            model=module,
            verbose=True,
        )

        with open(cfg.work_path / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=4)
    finally:
        if not cfg.devel:
            wandb_logger.experiment.finish(quiet=False)


def run_linear(cfg: Config):
    try:
        return _run_linear(cfg)
    finally:
        time.sleep(60)  # Give WandB some time to finish logging


if __name__ == '__main__':
    # noinspection PyTypeChecker
    run_linear(simple_parsing.parse(Config))
