from dataset.utils import prepare_datasets
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import LitProgressBar
import torch
from model import BoardDetector
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(description="Parser for model training parameters")
    parser.add_argument('-p', '--path', type=str, default='./idchess_zadanie', help='Path to folder with dataset')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Maximum training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='One training batch size')
    parser.add_argument('-g', '--gpu_num', type=int, default=0, help='How many GPUs can be used for training')
    namespace = parser.parse_args()

    dataset_path = namespace.path
    epochs = namespace.epochs
    batch_size = namespace.batch_size
    num_gpus = namespace.gpu_num

    train(dataset_path, epochs, batch_size, num_gpus)


def train(dataset_path, epochs, batch_size, num_gpus=0):
    print('Preparing datasets...')
    train_ds, val_ds, test_ds = prepare_datasets(dataset_path)
    print('Datasets are ready.')
    train_loader = DataLoader(train_ds, batch_size)
    val_loader = DataLoader(val_ds, batch_size)

    model = BoardDetector().to(device)
    print('Starting training')
    trainer = pl.Trainer(gpus=num_gpus, max_epochs=epochs,
                         callbacks=[LitProgressBar(), EarlyStopping(monitor='val_loss'),
                                    ModelCheckpoint(monitor="val_loss")],
                         default_root_dir="./checkpoints")

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("simple_mbnet.ckpt")
    print('Success')

if __name__ == '__main__':
    main()
