import os
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, download_url


class EuroSAT(ImageFolder):
    splits: ClassVar[tuple[str, str, str]] = ('train', 'val', 'test')
    split_filenames: ClassVar[dict[str, str]] = {
        'train': 'eurosat-train.txt',
        'val': 'eurosat-val.txt',
        'test': 'eurosat-test.txt',
    }
    split_md5s: ClassVar[dict[str, str]] = {
        'train': '908f142e73d6acdf3f482c5e80d851b1',
        'val': '95de90f2aa998f70a3b2416bfe0687b4',
        'test': '7ae5ab94471417b6e315763121e67c5f',
    }

    def __init__(
        self,
        root: str | Path,
        split: Literal['train', 'val', 'test'] = 'train',
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, 'eurosat')
        self._data_folder = os.path.join(self._base_folder, '2750')

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        with open(os.path.join(self._base_folder, self.split_filenames[split])) as f:
            self.samples: tuple[str, ...] = tuple([line.strip().lower() for line in f])

        def filter_fn(path: str) -> bool:
            return path.lower().endswith(self.samples)

        super().__init__(
            self._data_folder,
            is_valid_file=filter_fn,
            transform=transform,
            target_transform=target_transform,
        )
        self.root = os.path.expanduser(root)

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder)

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self._base_folder, exist_ok=True)
        for split in self.splits:
            download_url(
                f'https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c/{self.split_filenames[split]}',
                root=self._base_folder,
                filename=self.split_filenames[split],
                md5=self.split_md5s[split],
            )
        download_and_extract_archive(
            'https://huggingface.co/datasets/torchgeo/eurosat/resolve/c877bcd43f099cd0196738f714544e355477f3fd/EuroSAT.zip',
            download_root=self._base_folder,
            md5='c8fa014336c82ac7804f0398fcb19387',
        )
