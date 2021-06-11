from my_data_module import MyDataModule
from lit_model import LitModel
import numpy as np
import torch
import pytorch_lightning as pl
from model import AlexNet, resnet34

np.random.seed(42)
torch.manual_seed(42)


def main():
    lit_model = LitModel(resnet34(num_classes=176))
    data = MyDataModule()
    logger = pl.loggers.TensorBoardLogger("training/logs")
    trainer = pl.Trainer(gpus=1)

    # pylint: disable=no-member
    trainer.tune(
        lit_model, datamodule=data
    )  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member


if __name__ == "__main__":
    main()
