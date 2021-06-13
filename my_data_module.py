import pytorch_lightning as pl
from typing import Optional
import pandas as pd
from pathlib import Path
from utils import Mydataset
from torchvision import transforms
import torch
import argparse

PATH_DATASETS = "D:/xxd/classify-leaves"
BATCH_SIZE = 32
NUM_WORKERS = 0


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.data_dir = PATH_DATASETS
        self.args = vars(args) if args is not None else {}
        self.data_dir = Path(self.args.get("path_datasets", PATH_DATASETS))
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)
        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        }

    def setup(self, stage: Optional[str] = None):
        leaves_data = pd.read_csv(self.data_dir / "train.csv")
        leave_classes = leaves_data["label"].unique()
        leave_classes = dict((k, v) for v, k in enumerate(leave_classes))
        leaves_data["label"] = leaves_data["label"].map(leave_classes)
        leaves_data = leaves_data.sample(frac=1.0, random_state=1)

        train_num = int(0.9 * len(leaves_data))

        train_data = leaves_data.iloc[:train_num]
        train_data = train_data.reset_index(drop=True)

        val_data = leaves_data.iloc[train_num:]
        val_data = val_data.reset_index(drop=True)

        self.train_dataset = Mydataset(
            train_data, self.data_dir, transform=self.data_transforms["train"]
        )
        self.val_dataset = Mydataset(
            val_data, self.data_dir, transform=self.data_transforms["val"]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=NUM_WORKERS,
            help="Number of additional processes to load data.",
        )
        parser.add_argument(
            "--path_datasets", type=str, default=PATH_DATASETS, help="Path of datasets."
        )
        return parser
