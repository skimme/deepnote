from my_data_module import MyDataModule
from lit_model import LitModel
import numpy as np
import torch
import pytorch_lightning as pl
import argparse
import importlib


np.random.seed(42)
torch.manual_seed(42)


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'model.resnet34'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


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
    parser.add_argument("--model_class", type=str, default="resnet34")
    parser.add_argument("--load_checkpoint", type=str, default=None)

    # Get the data and model classes, so that we can add their specific arguments
    # temp_args, _ = parser.parse_known_args()
    # model_class = _import_class(f"models.{temp_args.model_class}")

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
    model_class = _import_class(f"models.{args.model_class}")
    model = model_class(num_classes=176)

    if args.load_checkpoint is not None:
        lit_model = LitModel.load_from_checkpoint(
            args.load_checkpoint, model=model, args=args
        )
    else:
        lit_model = LitModel(model=model, args=args)

    data = MyDataModule(args)
    logger = pl.loggers.WandbLogger(project="kaggle_leaves")
    logger.watch(lit_model)
    logger.log_hyperparams(vars(args))

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{val_cer:.3f}",
        monitor="val_loss",
        mode="min",
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, logger=logger, weights_save_path="training/logs"
    )

    # pylint: disable=no-member
    trainer.tune(
        lit_model, datamodule=data
    )  # If passing --auto_lr_find, this will set learning rate

    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)
    # pylint: enable=no-member


if __name__ == "__main__":
    main()
