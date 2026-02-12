"""
Benchmark Classification Script for Models in models/ folder
Tests various ConvNeXtV2 architectures on the Imagenette dataset using PyTorch Lightning.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import CSVLogger

try:
    from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
from models.convnextv2_ct import (
    convnextv2_atto, convnextv2_femto, convnextv2_pico,
    convnextv2_nano, convnextv2_tiny, convnextv2_base,
    convnextv2_large, convnextv2_huge,
)
from models.unireplknet_ct import (
    unireplknet_atto, unireplknet_femto, unireplknet_pico,
    unireplknet_nano, unireplknet_tiny, unireplknet_small,
    unireplknet_base, unireplknet_large, unireplknet_huge,
)
from models.coreccn import CoreEncoder

import argparse
from typing import Optional

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Configuration
CONFIG = {
    'data_dir': './datasets/imagenette2',
    'preprocessed_dir': './datasets/imagenette2_preprocessed',
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 0.05,
    'num_workers': min(os.cpu_count() or 4, 8),
    'save_dir': './checkpoints',
    'logs_dir': './logs',
    'model': 'corecnn',
    'drop_path_rate': 0.3,
    'precision': 'bf16-mixed',  # bfloat16 mixed precision
}


# Available models
AVAILABLE_MODELS = {
    'convnextv2_atto': convnextv2_atto,
    'convnextv2_femto': convnextv2_femto,
    'convnextv2_pico': convnextv2_pico,
    'convnextv2_nano': convnextv2_nano,
    'convnextv2_tiny': convnextv2_tiny,
    'convnextv2_base': convnextv2_base,
    'convnextv2_large': convnextv2_large,
    'convnextv2_huge': convnextv2_huge,
    'unireplknet_atto': unireplknet_atto,
    'unireplknet_femto': unireplknet_femto,
    'unireplknet_pico': unireplknet_pico,
    'unireplknet_nano': unireplknet_nano,
    'unireplknet_tiny': unireplknet_tiny,
    'unireplknet_small': unireplknet_small,
    'unireplknet_base': unireplknet_base,
    'unireplknet_large': unireplknet_large,
    'unireplknet_huge': unireplknet_huge,
    'corecnn': CoreEncoder,
}


class PreprocessedDataset(Dataset):
    """Dataset that loads from preprocessed numpy arrays using memory mapping."""

    def __init__(self, images_path, labels_path, transform=None):
        # Use memory mapping for efficient loading without loading entire array into RAM
        self.images = np.load(images_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Copy the image slice to ensure it's writable for transforms
        image = torch.from_numpy(self.images[idx].copy())
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


class EpochSummaryCallback(Callback):
    """Callback to print clean epoch summaries."""

    def on_validation_epoch_end(self, trainer, pl_module):
        """Print summary at end of validation (after both train and val are done)."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch + 1

        # Get metrics - handle both step and epoch versions
        # During training, metrics are logged with _epoch suffix for aggregated values
        train_loss_key = 'train_loss_epoch' if 'train_loss_epoch' in metrics else 'train_loss'
        train_acc_key = 'train_acc_epoch' if 'train_acc_epoch' in metrics else 'train_acc'

        train_loss = metrics.get(train_loss_key, None)
        train_acc = metrics.get(train_acc_key, None)
        val_loss = metrics.get('val_loss', None)
        val_acc = metrics.get('val_acc', None)

        # Skip if metrics aren't ready yet (shouldn't happen, but just in case)
        if train_loss is None or val_loss is None:
            return

        # Convert to scalar values
        train_loss = train_loss.item()
        train_acc = train_acc.item() * 100 if train_acc is not None else 0.0
        val_loss = val_loss.item()
        val_acc = val_acc.item() * 100 if val_acc is not None else 0.0
        lr = trainer.optimizers[0].param_groups[0]['lr']

        # Print on new line to avoid progress bar conflicts
        print(f"\nEpoch {epoch}/{trainer.max_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {lr:.6f}")

        # Check if this is the best model
        if hasattr(trainer.checkpoint_callback, 'best_model_score'):
            current_best = trainer.checkpoint_callback.best_model_score
            if current_best is not None and abs(val_acc/100 - current_best.item()) < 1e-6:
                print(f"  -> New best model! (Val Acc: {val_acc:.2f}%)")
        print("-" * 60)


class ImagenetteDataModule(pl.LightningDataModule):
    """Lightning DataModule for Imagenette dataset."""

    def __init__(
        self,
        preprocessed_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.preprocessed_dir = preprocessed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = None
        self.num_channels = None

    def setup(self, stage: Optional[str] = None):
        """Load datasets."""
        # File paths for memory-mapped loading
        train_images_path = os.path.join(self.preprocessed_dir, 'train_images.npy')
        train_labels_path = os.path.join(self.preprocessed_dir, 'train_labels.npy')
        val_images_path = os.path.join(self.preprocessed_dir, 'val_images.npy')
        val_labels_path = os.path.join(self.preprocessed_dir, 'val_labels.npy')
        classes = np.load(os.path.join(self.preprocessed_dir, 'classes.npy'), allow_pickle=True)
        self.num_classes = len(classes)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomErasing(p=0.5),  # Safe alternative for augmentation
            transforms.RandomRotation(15),  # Add rotation for more geometric augmentation
        ])

        # No augmentation for validation
        val_transform = None

        if stage == 'fit' or stage is None:
            self.train_dataset = PreprocessedDataset(
                train_images_path, train_labels_path, transform=train_transform
            )
            self.val_dataset = PreprocessedDataset(
                val_images_path, val_labels_path, transform=val_transform
            )
            sample_image, _ = self.train_dataset[0]
            self.num_channels = int(sample_image.shape[0])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class ModelSelector(pl.LightningModule):
    """Lightning Module for classification."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        in_chans: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        drop_path_rate: float = 0.2,
        max_epochs: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        model_fn = AVAILABLE_MODELS[model_name]
        self.model = model_fn(
            num_classes=num_classes,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate
        )

        self.criterion = nn.CrossEntropyLoss()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model {model_name} created with {total_params:,} parameters")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics - show on progress bar during training, aggregate for epoch
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def run_benchmark(
    model_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    config: dict,
):
    """Run a single benchmark experiment."""
    print("\n" + "=" * 60)
    print(f"Benchmarking: {model_name}")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print("=" * 60)

    # Create data module
    data_module = ImagenetteDataModule(
        preprocessed_dir=config['preprocessed_dir'],
        batch_size=batch_size,
        num_workers=config['num_workers'],
    )

    # Setup to get number of classes
    data_module.setup('fit')
    num_classes = data_module.num_classes

    print(f"Number of classes: {num_classes}")
    print(f"Input channels: {data_module.num_channels}")
    print(f"Training samples: {len(data_module.train_dataset)}")
    print(f"Validation samples: {len(data_module.val_dataset)}")
    print("=" * 60)

    # Create model
    model = ModelSelector(
        model_name=model_name,
        num_classes=num_classes,
        in_chans=data_module.num_channels,
        learning_rate=learning_rate,
        weight_decay=config['weight_decay'],
        drop_path_rate=config['drop_path_rate'],
        max_epochs=num_epochs,
    )

    # Create run name
    run_name = f"{model_name}_bs{batch_size}_lr{learning_rate:.0e}".replace("+", "").replace("-", "m")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config['save_dir'], run_name),
        filename='best_model',
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    epoch_summary = EpochSummaryCallback()

    callbacks = [checkpoint_callback, lr_monitor, epoch_summary]
    if RICH_AVAILABLE:
        progress_bar = RichProgressBar(leave=False)  # Don't leave progress bars to reduce clutter
        callbacks.append(progress_bar)

    # Logger
    csv_logger = CSVLogger(
        save_dir=config['logs_dir'],
        name=run_name,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='auto',
        devices=1,
        precision=config['precision'],
        callbacks=callbacks,
        logger=csv_logger,
        deterministic=False,
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, data_module)

    # Print results
    best_val_acc = checkpoint_callback.best_model_score.item() * 100
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print("=" * 60)

    return {
        'model_name': model_name,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'best_val_acc': best_val_acc,
        'checkpoint_path': checkpoint_callback.best_model_path,
    }


def run_hyperparameter_sweep(config: dict):
    """Run hyperparameter sweep across multiple configurations."""
    batch_sizes = [32, 64]
    learning_rates = [1e-3, 5e-4]

    results = []
    total_runs = len(batch_sizes) * len(learning_rates)
    run_idx = 0

    print("\n" + "=" * 60)
    print(f"Starting Hyperparameter Sweep ({total_runs} runs)")
    print("=" * 60)

    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            run_idx += 1
            print(f"\n{'='*60}")
            print(f"Run {run_idx}/{total_runs}")
            print(f"{'='*60}")
            result = run_benchmark(
                model_name=config['model'],
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=config['num_epochs'],
                config=config,
            )
            results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("Sweep Summary")
    print("=" * 60)
    sorted_results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
    for result in sorted_results:
        print(
            f"{result['model_name']} (bs={result['batch_size']}, lr={result['learning_rate']:.0e}): "
            f"best_val_acc={result['best_val_acc']:.2f}%"
        )

    return sorted_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark ConvNeXtV2 models on Imagenette')
    parser.add_argument('--model', type=str, default=CONFIG['model'],
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Model architecture to benchmark')
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=CONFIG['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=CONFIG['num_epochs'],
                        help='Number of epochs')
    parser.add_argument('--sweep', action='store_true',
                        help='Run hyperparameter sweep')
    parser.add_argument('--preprocessed-dir', type=str, default=CONFIG['preprocessed_dir'],
                        help='Path to preprocessed data directory')
    parser.add_argument('--save-dir', type=str, default=CONFIG['save_dir'],
                        help='Directory to save checkpoints')
    parser.add_argument('--logs-dir', type=str, default=CONFIG['logs_dir'],
                        help='Directory to save logs')

    args = parser.parse_args()

    # Update config with command line arguments
    config = CONFIG.copy()
    config['model'] = args.model
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.learning_rate
    config['num_epochs'] = args.epochs
    config['preprocessed_dir'] = args.preprocessed_dir
    config['save_dir'] = args.save_dir
    config['logs_dir'] = args.logs_dir

    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['logs_dir'], exist_ok=True)

    # Check if preprocessed data exists
    if not os.path.exists(config['preprocessed_dir']):
        print(f"Error: Preprocessed data directory not found: {config['preprocessed_dir']}")
        print("Please preprocess the dataset first.")
        return

    print("\n" + "=" * 60)
    print("ConvNeXtV2 Benchmark on Imagenette")
    print("=" * 60)
    print(f"Model: {config['model']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Precision: {config['precision']}")
    print(f"Workers: {config['num_workers']}")
    print("=" * 60 + "\n")

    if args.sweep:
        run_hyperparameter_sweep(config)
    else:
        run_benchmark(
            model_name=config['model'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            num_epochs=config['num_epochs'],
            config=config,
        )


if __name__ == '__main__':
    main()
