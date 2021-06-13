from my_data_module import MyDataModule
from lit_model import LitModel
import numpy as np
import torch
import pytorch_lightning as pl
from model import AlexNet, resnet34
import argparse


np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[
        1
    ].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Basic arguments
    parser.add_argument("--data_class", type=str, default="MNIST")
    parser.add_argument("--model_class", type=str, default="MLP")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    MyDataModule.add_to_argparse(data_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    LitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    if args.load_checkpoint is not None:
        lit_model = LitModel.load_from_checkpoint(
            args.load_checkpoint, model=resnet34(num_classes=176), args=args
        )
    else:
        lit_model = LitModel(resnet34(num_classes=176), args=args)

    data = MyDataModule(args)
    logger = pl.loggers.WandbLogger(project="kaggle_leaves")
    logger.watch(lit_model)
    trainer = pl.Trainerfrom_argparse_args(args, logger=logger)

    # pylint: disable=no-member
    trainer.tune(
        lit_model, datamodule=data
    )  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member


if __name__ == "__main__":
    main()
