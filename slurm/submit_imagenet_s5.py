import itertools
import math
import socket
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

import rich
import simple_parsing
import submitit
import wandb

from slurm.utils import SingleConfigNode
from tmcl.config_tmcl_dali import (
    ClassLearnerConfig,
    Config,
    EvalConfig,
    OptimConfig,
    PastDistillationConfig,
    SSLConfig,
    SupConConfig,
    SupHeadConfig,
    TMCLConfig,
)
from tmcl.main_tmcl_dali import (
    run_tmcl,
)

SLURM_LOG_DIR = '/p/scratch/neuroml/tran4/slurm_logs'


@dataclass
class Args:
    name: str = 'imagenet_s5_iclr'
    project_name: str = 'imagenet_s5_iclr'
    group: str = 'imagenet_main'
    devel: bool = False
    resume: bool = False
    local: bool = False
    skip_until: str | None = None

    # SLURM settings
    gpus_per_task: int = 1
    cpus_per_task: int = 16
    account: str | None = 'neuroml'
    partition: str = 'booster'
    time: int = 1440
    project: str | None = None
    nodelist: str | None = None

    def __post_init__(self):
        if self.project is None:
            if 'juwels' in socket.gethostname():
                rich.print('[bold blue]Assuming `neuroml`![/bold blue]')
                self.project = 'neuroml'
            elif 'jureca' in socket.gethostname():
                rich.print('[bold blue]Assuming `jinm60`![/bold blue]')
                self.project = 'jinm60'
            elif 'pgi15' in socket.gethostname():
                rich.print('[bold blue]Assuming `pgi15`![/bold blue]')
                self.project = 'pgi15'
            else:
                rich.print('[bold red]Unknown host![/bold red]')
        match self.project:
            case 'neuroml':
                self.account = 'neuroml'
                self.partition = 'booster'
                self.cpus_per_task = 6
            case 'jinm60':
                self.account = 'jinm60'
                self.partition = 'dc-gpu'
                self.cpus_per_task = 128 // 4  # 128 cores per node
            case 'pgi15':
                self.account = None
                self.partition = 'pgi15'
                self.cpus_per_task = 24 // 4  # 24-32 real cores per node depending on the node
                self.nodelist = 'pgi15-gpu[14-20]'  # only consider nodes with 40GB+ per GPU
            case 'westai':
                self.account = 'westai0068'
                self.partition = 'dc-hwai'
                self.cpus_per_task = 64 // 4
            case 'janwestai':
                self.account = 'westai0082'
                self.partition = 'dc-hwai'
                self.cpus_per_task = 64 // 4
            case None:
                ...
            case _:
                raise ValueError(f'Unknown project: {self.project}')
        if self.devel:
            match self.project:
                case 'neuroml':
                    self.partition = 'develbooster'
                    self.time = 60
                case 'jinm60':
                    self.partition = 'dc-gpu-devel'
                    self.time = 60
                case 'pgi15':
                    ...  # no devel partition
                case _:
                    raise ValueError(f'Unknown project: {self.project}')


def build_configs(args: Args) -> list[Config]:
    configs = []
    for labeled_frac, methods, seed in itertools.product(
        [1.0, 0.1],
        [
            'vi',
            'vi_mi',
            'vi_pnr',
            'vi_pfr',
            'vi_supcon',
            'vi_suphead',
            'vi_pnr_mi',
            'vi_pnr_supcon',
            'vi_pnr_suphead',
            'supcon',
            'supcon_pnr',
            'suphead',
            'vi_si',
        ],
        [0, 1, 2],
    ):
        method_flags = methods.split('_')

        if labeled_frac != 1.0 and not any(m in method_flags for m in ['supcon', 'suphead', 'mi']):
            # skip labeled_frac ablation for unsupervised configs
            continue

        actual_labeled_frac = (
            labeled_frac if methods not in ['vi', 'vi_si', 'vi_pnr', 'vi_pfr'] else 1.0
        )

        distill_algo = 'none'
        distill_proj_hidden_dim = 2048
        if 'si' in method_flags:
            distill_algo = 'cassle'
        elif 'pnr' in method_flags:
            distill_algo = 'pnr'
        elif 'pfr' in method_flags:
            distill_algo = 'pfr'
            distill_proj_hidden_dim = 256

        if 'supcon' not in method_flags and 'suphead' not in method_flags:
            cl = ClassLearnerConfig(
                method='contrastive_opl',
                labeled_setup='onevsall',
                layerwise_weight_decay=(0.4, 0.04),
                augment='sup',
                during_pretraining=False,
                opl_square_loss=False,
                opl_neg_weight=1.0,
            )
        else:
            cl = ClassLearnerConfig(
                method='sup',
                labeled_setup='allvsall',
                layerwise_weight_decay=(0.4, 0.04),
                augment='ssl',
                during_pretraining=False,
            )
        hparams = Config(
            seed=seed,
            setup='s5',
            cons_num_views=4,
            labeled_frac=labeled_frac,
            project_name=args.project_name,
            deterministic=True,
            name=f'{args.name}@{labeled_frac}@{methods}@{seed}',
            group=f'{actual_labeled_frac} {methods}',
            devel=args.devel,
            gpus_per_task=4,
            n_nodes=1,
            n_data_workers=args.cpus_per_task,
            timm_model='convit_small_dytox',
            batch_size=256,
            optim=OptimConfig(
                optim_algo='adamw',
                cons_lr=1e-3,
                cons_min_lr=1e-3 / 1000,
                tl_lr=0.01,
                tl_min_lr=0.00001,
                warmup_epochs=10,
                backbone_weight_decay=(1e-4, 1e-4),
            ),
            class_learner=cl,
            ssl=SSLConfig(
                disable_ssl='vi' not in method_flags,
                ssl_algo='ghosh',
                barlow_lambda=0.005,
                head_hidden_dim=2048,
                head_output_dim=2048,
                barlow_scale_loss=0.024,
                ssl_weight=1.0,
                reset_projector=False,
            ),
            tmcl=TMCLConfig(
                disable_tmcl='mi' not in method_flags,
                tmcl_algo='ghosh',
                head_hidden_dim=2048,
                head_output_dim=2048,
                barlow_lambda=0.005,
                barlow_scale_loss=0.024,
                tmcl_weight=1.0,
                use_predictor=True,
                stop_grad_tms=True,
                unmod_first_view=True,
                reset_projector=False,
            ),
            eval=EvalConfig(
                eval_before_training=(methods == 'vi'),
                eval_interval=25,
                knn_per_session=True,
            ),
            checkpoint_interval=100,
            pretrain_epochs=(200, 200),
            incremental_epochs=(50, 200) if 'mi' in method_flags else (0, 200),
            torch_compile=True,
            # last_batch='pad',
            # unsup_replacement=True,
            # sup_replacement=True,
            distill=PastDistillationConfig(
                algo=distill_algo,
                barlow_lambda=0.005,
                barlow_scale_loss=0.024,
                head_hidden_dim=distill_proj_hidden_dim,
                head_output_dim=2048,
                distill_ssl='ghosh',
                pfr_lambda=25.0,
            ),
            supcon=SupConConfig(
                enable_supcon=('supcon' in method_flags),
                supcon_weight=1.0,
                head_hidden_dim=2048,
                head_output_dim=128,
                temperature=0.1,
            ),
            suphead=SupHeadConfig(
                enable_sup_head=('suphead' in method_flags),
                head_hidden_dim=2048,
            ),
        )
        if args.devel:
            hparams.pretrain_epochs = (1, 1)
            hparams.incremental_epochs = (int(hparams.incremental_epochs[0] > 1), 1)
        if args.project == 'pgi15':
            hparams.data_path = Path('/Users/vtran/data')
            hparams.base_work_path = Path('/Users/vtran/runs/tmcl-cil')
        configs.append(hparams)

    if args.resume:
        api = wandb.Api()
        entity = 'llfs'
        project = args.project_name
        for c in configs:
            latest_run = api.runs(
                f'{entity}/{project}',
                filters={'config.name': c.name},  # grid makes config.name unique per config
                order='-created_at',
                per_page=1,
            )
            if not latest_run:
                raise ValueError(f'No runs found for config {c.name}!')

            c.resume = True
            c.resume_id = latest_run[0].id
            rich.print(f'Found latest run for {c.name}: {c.resume_id}')
    return configs


def submit(args: Args):
    configs = build_configs(args)

    if args.skip_until:
        while configs and configs[0].name != args.skip_until:
            rich.print(f'Skipping {configs[0].name}')
            configs.pop(0)
        if not configs:
            raise ValueError(f'No config found with name {args.skip_until}')
        rich.print(f'Not skipping {configs[0].name} and following.')

    if args.devel:
        rich.print('[bold red]Running in development mode![/bold red]')
        rich.print('Will submit first 4 jobs over single node.')
    else:
        rich.print(
            f'Will submit {len(configs)} jobs over {int(math.ceil(len(configs) / 4))} nodes.'
        )

    if args.local:
        rich.print('[bold red]Running in local mode![/bold red]')
        for cfg in configs:
            pprint(cfg)
            run_tmcl(cfg)

            if args.devel:
                break
        return

    prefix = Path(__file__).stem
    if args.project == 'pgi15':
        slurm_log_dir = Path('/Users/vtran/slurm_logs') / prefix
    else:
        slurm_log_dir = Path('/p/scratch/neuroml/tran4/slurm_logs') / prefix
    if args.devel:
        slurm_log_dir /= 'devel'
        prefix = f'DEVEL__{prefix}'
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    for node_cfg in configs:
        executor = submitit.AutoExecutor(
            folder=str(slurm_log_dir / node_cfg.name), slurm_max_num_timeout=30
        )
        sbatch_kwargs = {}
        if args.nodelist:
            sbatch_kwargs['nodelist'] = args.nodelist
        if args.project == 'pgi15':
            sbatch_kwargs['exclusive'] = True
        executor.update_parameters(
            name=node_cfg.name,
            mem_gb=0,
            gpus_per_node=4,
            tasks_per_node=4,
            cpus_per_task=args.cpus_per_task,
            nodes=1,
            timeout_min=args.time,
            slurm_account=args.account,
            slurm_partition=args.partition,
            slurm_signal_delay_s=120,
            slurm_srun_args=[
                f'--cpus-per-task={args.cpus_per_task}',
                '--cpu-bind=threads',
                '--distribution=block:cyclic:cyclic',
            ],
            slurm_additional_parameters={'threads_per_core': 1, **sbatch_kwargs},
        )
        job = executor.submit(SingleConfigNode(node_cfg, run_fn=run_tmcl))
        print(f'Submitted job_id: {job.job_id}')
        print(f'Logs and checkpoints will be saved at: {SLURM_LOG_DIR}')

        if args.devel:
            break


if __name__ == '__main__':
    # noinspection PyTypeChecker
    submit(simple_parsing.parse(Args))
