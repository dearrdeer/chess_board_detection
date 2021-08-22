import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import pytorch_lightning as pl


class BoardDetector(pl.LightningModule):
    def __init__(self, extractor_out_features: int = 256):
        super().__init__()
        self.mbnet = mobilenet_v3_small(False)
        mbnet_num_features = self.mbnet.classifier[3].in_features
        self.mbnet.classifier[3] = nn.Sequential(nn.Linear(mbnet_num_features, extractor_out_features, True), nn.ReLU())

        self.detector = nn.Sequential(nn.Linear(extractor_out_features, 8, True))  # Output coordinates

    def forward(self, x):
        emb = self.mbnet(x)
        pts = self.detector(emb)
        return pts

    def get_loss(self, pred_pts, true_pts):
        loss = nn.MSELoss()
        result = loss(pred_pts, true_pts)
        return result

    def training_step(self, batch, batch_idx):
        x, true_pts = batch
        pred_pts = self(x)
        loss = self.get_loss(pred_pts, true_pts)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, true_pts = batch
        pred_pts = self(x)
        loss = self.get_loss(pred_pts, true_pts)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

