import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        o = self.decoder(z)
        loss = nn.functional.mse_loss(o, x)
        # Logging to Tensorboard by default
        self.log("train_loss", loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(dataset)

    # init model
    autoencoder = LitAutoEncoder()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(accelerator="gpu", devices=8) (if you have GPUs)
    logger = TensorBoardLogger("tb_logs", name="demo_model")
    trainer = pl.Trainer(max_epochs=3)
    trainer.fit(model=autoencoder,
                train_dataloaders=train_loader)