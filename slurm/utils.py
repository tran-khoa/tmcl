import importlib
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable, Sequence
from io import TextIOWrapper
from pathlib import Path
from pprint import pprint
from typing import TextIO

import rich
import submitit


class SingleConfigNode[T]:
    def __init__(
        self,
        config: T,
        run_fn: Callable[[T], None],
        torch_distributed: bool = True,
        override_env: dict[str, str] | None = None,
    ):
        """
        Assumes one node per config.
        """
        self.config = config
        self.run_fn = run_fn
        self.torch_distributed = torch_distributed
        self.override_env = override_env or {}

    @property
    def environment(self) -> dict[str, str]:
        return {
            'PYTHONFAULTHANDLER': '1',
            'CUDA_VISIBLE_DEVICES': '0,1,2,3',
            **self.override_env,
        }

    def __call__(self):
        if self.torch_distributed:
            submitit.helpers.TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
        os.environ.update(self.environment)
        pprint(self.environment)

        self.run_fn(self.config)

        if os.environ.get('SLURM_ARRAY_JOB_ID'):
            slurm_job_id = f'{os.environ["SLURM_ARRAY_JOB_ID"]}_{os.environ["SLURM_ARRAY_TASK_ID"]}'
        else:
            slurm_job_id = os.environ['SLURM_JOB_ID']
        os.system(f'scancel {slurm_job_id}')


class MultiConfigNode[T]:
    def __init__(
        self,
        configs: Sequence[T],
        run_fn: Callable[[T], None],
        env_override: dict[str, str] | None = None,
    ):
        """
        Assumes one GPU per config (4 configs per node).
        """
        self.configs = configs
        self.run_fn = run_fn
        self.env_override = env_override or {}

    @property
    def environment(self) -> dict[str, str]:
        return {'PYTHONFAULTHANDLER': '1', **self.env_override}

    def __call__(self):
        job_env = submitit.JobEnvironment()
        global_rank = job_env.global_rank
        local_rank = job_env.local_rank

        if global_rank >= len(self.configs):
            print('Nothing to do.')
            return

        # One GPU per task
        os.environ.update(self.environment)
        os.environ.update({'CUDA_VISIBLE_DEVICES': str(local_rank)})

        # https://github.com/Lightning-AI/pytorch-lightning/issues/5225#issuecomment-750032030
        # Unset SLURM keys to avoid PyTorch Lightning SLURM detection and pretend we are on a single rank.
        for slurm_key in (
            'SLURM_NTASKS',
            'SLURM_JOB_NAME',
            'RANK',
            'LOCAL_RANK',
            'SLURM_PROCID',
            'JSM_NAMESPACE_RANK',
        ):
            if slurm_key in os.environ:
                print(f'Unsetting {slurm_key}(={os.environ[slurm_key]}) from environment.')
                del os.environ[slurm_key]
        # Reload PyTorch Lightning to avoid SLURM detection.
        import lightning_fabric
        import lightning_utilities

        importlib.reload(lightning_fabric)
        importlib.reload(lightning_utilities)
        from lightning_fabric.utilities import rank_zero_only

        rank_zero_only.rank = 0

        cfg = self.configs[global_rank]
        rich.print(cfg)

        self.run_fn(cfg)


class Tee:
    def __init__(self, *streams: TextIOWrapper | TextIO):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


class MultiConfigQueueNode[T]:
    def __init__(
        self,
        configs: Sequence[T],
        run_fn: Callable[[T], None],
        log_path: Path,
        *,
        num_gpus: int = 4,
        env_override: dict[str, str] | None = None,
    ):
        """
        Assumes one GPU per config (4 configs per node).
        """
        self.configs = configs
        self.run_fn = run_fn
        self.log_path = log_path
        self.num_gpus = num_gpus
        self.env_override = env_override or {}

    def worker(self, worker_id: int, config_queue: mp.Queue, progress_counter: mp.Value):
        print(f'Worker {worker_id} logs: {self.log_path / f"worker_{worker_id}.log"}')
        with open(self.log_path / f'worker_{worker_id}.log', 'w', buffering=1) as f:
            # os.dup2(f.fileno(), sys.stdout.fileno())
            # os.dup2(f.fileno(), sys.stderr.fileno())
            sys.stdout = Tee(sys.__stdout__, f)
            sys.stderr = Tee(sys.__stderr__, f)

            os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_id)
            while True:
                task = config_queue.get()
                if task is None:
                    print(f'RunDef: No tasks left, worker {worker_id} exiting.')
                    break
                print(f'RunDef: Worker {worker_id} processing task {task}.')

                self.run_fn(task)
                with progress_counter.get_lock():
                    progress_counter.value += 1

    @property
    def environment(self) -> dict[str, str]:
        return {'PYTHONFAULTHANDLER': '1', **self.env_override}

    def __call__(self):
        job_env = submitit.JobEnvironment()
        assert job_env.num_tasks == 1
        os.environ.update(self.environment)

        ctx = mp.get_context('spawn')
        task_queue = ctx.Queue()
        progress_counter = ctx.Value('i', 0)

        workers = [
            ctx.Process(target=self.worker, args=(i, task_queue, progress_counter))
            for i in range(self.num_gpus)
        ]
        for w in workers:
            w.start()
        for config in self.configs:
            task_queue.put(config)
        for _ in range(self.num_gpus):
            task_queue.put(None)
        print(f'RunDef: Queued {len(self.configs)} tasks for {self.num_gpus} workers.')

        counter = 0
        while counter < len(self.configs):
            with progress_counter.get_lock():
                if progress_counter.value > counter:
                    counter = progress_counter.value
                    print(f'RunDef: {counter}/{len(self.configs)} tasks completed.')
            time.sleep(10)

        for p in workers:
            p.join()
