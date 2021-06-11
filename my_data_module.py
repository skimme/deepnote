import pytorch_lightning as pl
from typing import Optional
import pandas as pd
from pathlib import Path
from utils import Mydataset
from torchvision import transforms
import torch

PATH_DATASETS = Path("D:/xxd/classify-leaves")
BATCH_SIZE = 48

class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = PATH_DATASETS, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_transforms = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224), 
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

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

        self.train_dataset = Mydataset(train_data, self.data_dir, transform=self.data_transforms["train"])
        self.val_dataset = Mydataset(val_data, self.data_dir, transform=self.data_transforms["val"])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
