import dataclasses
import logging

import kornia.augmentation as K
import pytorch_lightning as pl
import simple_parsing
import torch
import torch.nn.functional as F
from jaxtyping import Float
from kornia.augmentation import AugmentationSequential
from kornia.constants import Resample
from lightly.models.utils import deactivate_requires_grad
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from rich.logging import RichHandler
from torch import Tensor, nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MetricCollection
from torchvision.datasets import CIFAR10, DTD, GTSRB, SVHN
from torchvision.transforms import v2

from tmcl.config_linear import Config
from tmcl.data_setups import STANDARD_TRANSFORM, downscale_transform
from tmcl.datasets.aircraft import Aircraft
from tmcl.datasets.cu_birds import CUBirds
from tmcl.datasets.eurosat import EuroSAT
from tmcl.datasets.labels import get_dataset_labels
from tmcl.datasets.transforms import CIFAR_STATS, STL10_STATS
from tmcl.datasets.vgg_flower import VGGFlower
from tmcl.nn.modulations import build_tm_model


class LinearModule(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.console_logger = logging.getLogger(__name__)

        match cfg.eval_dataset:
            case (
                'cifar100'
                | 'cifar10'
                | 'gtsrb'
                | 'eurosat'
                | 'svhn'
                | 'dtd'
                | 'cubirds'
                | 'vggflower'
                | 'aircraft'
            ):
                image_size = 32
                patch_size = 4
                dataset = self.train_dataloader().dataset
                num_classes = len(set(get_dataset_labels(dataset)))
                stats = CIFAR_STATS
            case 'stl10':
                image_size = 32
                patch_size = 4
                num_classes = 10
                stats = STL10_STATS
            case _:
                raise ValueError(f'Unknown dataset {cfg.eval_dataset}')

        match self.cfg.continual_setup:
            case None:
                self.sessions = None
            case 'stl10/class/s5':
                rng = torch.Generator().manual_seed(cfg.seed)
                self.sessions = [x.tolist() for x in torch.randperm(10, generator=rng).split(2)]
            case 'c100/class/s10@pretrain_tl':
                rng = torch.Generator().manual_seed(cfg.seed)
                self.sessions = [x.tolist() for x in torch.randperm(100, generator=rng).split(10)]
            case (
                'c100/class/cassle_s5@cassle_s5'
                | 'c100/class/cassle_s5@thal'
                | 'c100/class/cassle_s5@pretrain_tl'
            ):
                rng = torch.Generator().manual_seed(5)
                self.sessions = [x.tolist() for x in torch.randperm(100, generator=rng).split(20)]
            case _:
                raise ValueError(f'Unknown continual setup {cfg.continual_setup}')

        if self.cfg.load_direct_r18:
            self.backbone = torch.load(
                self.cfg.load_checkpoint_from, map_location=cfg.device, weights_only=False
            )
            self.backbone.classifier = nn.Identity()
            self.backbone.linear = nn.Identity()
            deactivate_requires_grad(self.backbone)
            self.classifier = nn.Linear(512, num_classes)
        else:
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

        if self.cfg.train_augmentations:
            self.sup_transform = AugmentationSequential(
                K.RandomResizedCrop(
                    size=(32, 32),
                    scale=(0.08, 1.0),
                    resample=Resample.BICUBIC,
                ),
                K.RandomHorizontalFlip(p=0.5),
                K.Normalize(
                    std=stats.std,
                    mean=stats.mean,
                ),
            )
        else:
            self.sup_transform = AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.Normalize(
                    std=stats.std,
                    mean=stats.mean,
                ),
            )

        self.eval_transform = AugmentationSequential(
            K.Normalize(
                std=stats.std,
                mean=stats.mean,
            ),
            same_on_batch=True,
        )

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
                    task='multiclass', num_classes=num_classes, top_k=1
                ),
                f'eval_{self.cfg.eval_dataset}_top5': Accuracy(
                    task='multiclass', num_classes=num_classes, top_k=5
                ),
            }
        )
        self.session_eval_metrics = []
        if self.sessions is not None:
            self.session_eval_metrics = nn.ModuleList(
                [
                    Accuracy(task='multiclass', num_classes=num_classes, top_k=1, average='micro')
                    for _ in self.sessions
                ]
            )

        # noinspection PyTypeChecker
        self.save_hyperparameters(dataclasses.asdict(cfg))

    def train_dataloader(self):
        match self.cfg.eval_dataset:
            case 'cifar100':
                from torchvision.datasets import CIFAR100

                dataset = CIFAR100(
                    root=self.cfg.data_path,
                    train=True,
                    download=True,
                    transform=v2.ToTensor(),
                )
            case 'cifar10':
                dataset = CIFAR10(
                    root=self.cfg.data_path,
                    train=True,
                    download=True,
                    transform=v2.ToTensor(),
                )
            case 'eurosat':
                dataset = EuroSAT(
                    self.cfg.data_path,
                    split='train',
                    transform=downscale_transform(32),
                )
            case 'gtsrb':
                dataset = GTSRB(
                    self.cfg.data_path,
                    split='train',
                    transform=downscale_transform(32),
                )
            case 'svhn':
                dataset = SVHN(
                    self.cfg.data_path,
                    split='train',
                    download=True,
                    transform=STANDARD_TRANSFORM,
                )
            case 'dtd':
                dataset = DTD(
                    self.cfg.data_path,
                    split='train',
                    download=True,
                    transform=downscale_transform(32),
                )
            case 'cubirds':
                dataset = CUBirds(
                    self.cfg.data_path,
                    train=True,
                    transform=downscale_transform(32),
                )
            case 'vggflower':
                dataset = VGGFlower(
                    self.cfg.data_path,
                    train=True,
                    transform=downscale_transform(32),
                )
            case 'aircraft':
                dataset = Aircraft(
                    self.cfg.data_path,
                    train=True,
                    transform=downscale_transform(32),
                )
            case 'stl10':
                from torchvision.datasets import STL10

                dataset = STL10(
                    root=self.cfg.data_path,
                    split='train',
                    download=True,
                    transform=downscale_transform(32),
                )
            case _:
                raise ValueError(f'Unknown dataset {self.cfg.eval_dataset}')
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size_per_gpu,
            num_workers=self.cfg.n_data_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.cfg.n_data_workers > 0),
        )

    def val_dataloader(self):
        match self.cfg.eval_dataset:
            case 'cifar100':
                from torchvision.datasets import CIFAR100

                test_dataset = CIFAR100(
                    root=self.cfg.data_path,
                    train=False,
                    download=True,
                    transform=v2.ToTensor(),
                )
            case 'cifar10':
                test_dataset = CIFAR10(
                    root=self.cfg.data_path,
                    train=False,
                    download=True,
                    transform=v2.ToTensor(),
                )
            case 'eurosat':
                test_dataset = EuroSAT(
                    self.cfg.data_path,
                    split='test',
                    transform=downscale_transform(32),
                )
            case 'gtsrb':
                test_dataset = GTSRB(
                    self.cfg.data_path,
                    split='test',
                    transform=downscale_transform(32),
                )
            case 'svhn':
                test_dataset = SVHN(
                    self.cfg.data_path,
                    split='test',
                    download=True,
                    transform=STANDARD_TRANSFORM,
                )
            case 'dtd':
                test_dataset = DTD(
                    self.cfg.data_path,
                    split='test',
                    download=True,
                    transform=downscale_transform(32),
                )
            case 'cubirds':
                test_dataset = CUBirds(
                    self.cfg.data_path,
                    train=False,
                    transform=downscale_transform(32),
                )
            case 'vggflower':
                test_dataset = VGGFlower(
                    self.cfg.data_path,
                    train=False,
                    transform=downscale_transform(32),
                )
            case 'aircraft':
                test_dataset = Aircraft(
                    self.cfg.data_path,
                    train=False,
                    transform=downscale_transform(32),
                )
            case 'stl10':
                from torchvision.datasets import STL10

                test_dataset = STL10(
                    root=self.cfg.data_path,
                    split='test',
                    download=True,
                    transform=downscale_transform(32),
                )
            case _:
                raise ValueError(f'Unknown dataset {self.cfg.eval_dataset}')
        return DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_per_gpu,
            num_workers=self.cfg.n_data_workers,
            shuffle=False,
            pin_memory=False,
            persistent_workers=False,
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
        images, targets = batch
        images = self.sup_transform(images)
        loss, logits = self.shared_step(images, targets)
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        train_metrics = self.train_metrics(logits, targets)
        self.log_dict(train_metrics)
        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = self.eval_transform(images)
        loss, logits = self.shared_step(images, targets)
        self.eval_metrics.update(logits, targets)

        for session, session_m in enumerate(self.session_eval_metrics):
            session_class_ids = self.sessions[session]

            mask = torch.isin(targets, torch.tensor(session_class_ids, device=targets.device))
            if not mask.any():
                continue

            session_m.update(logits[mask], targets[mask])

        self.log(f'val_{self.cfg.eval_dataset}_loss', loss, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.eval_metrics.compute())
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


def run_linear(cfg: Config):
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
            max_epochs=cfg.num_epochs,
            devices=cfg.n_devices,
            num_nodes=cfg.n_nodes,
            accelerator=cfg.accelerator,
            strategy='ddp_find_unused_parameters_true' if cfg.n_devices > 1 else 'auto',
            sync_batchnorm=cfg.sync_batchnorm,
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
                    every_n_epochs=10,
                    save_top_k=0,
                ),
                pl.callbacks.ModelSummary(depth=3),
            ],
            num_sanity_val_steps=2,
            log_every_n_steps=cfg.log_interval,
            limit_train_batches=2 if cfg.devel else 1.0,
            check_val_every_n_epoch=cfg.eval_interval,
        )
        trainer.fit(
            model=module,
            ckpt_path=cfg.get_last_checkpoint(force=True) if cfg.resume else None,
        )
    finally:
        if not cfg.devel:
            wandb_logger.experiment.finish(quiet=False)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    run_linear(simple_parsing.parse(Config))
