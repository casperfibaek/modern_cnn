"""
ConvNeXtV2 Training on Imagenette Dataset
Trains a ConvNeXtV2-base model (not pretrained) on the Imagenette dataset.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from models.convnextv2 import convnext_pico
from tqdm import tqdm
import time


# Configuration
CONFIG = {
    'data_dir': './datasets/imagenette2',
    'preprocessed_dir': './datasets/imagenette2_preprocessed',
    'num_epochs': 20,
    'batch_sizes': [32],
    'learning_rates': [1e-3],
    'weight_decay': 0.05,
    'num_workers': min(os.cpu_count() or 4, 8),  # Use available CPU cores (max 8)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints',
    'logs_dir': './logs',
    'log_interval': 10,
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


def get_data_loaders(preprocessed_dir, batch_size, num_workers):
    """Create train and validation data loaders from preprocessed data."""

    # File paths for memory-mapped loading
    train_images_path = os.path.join(preprocessed_dir, 'train_images.npy')
    train_labels_path = os.path.join(preprocessed_dir, 'train_labels.npy')
    val_images_path = os.path.join(preprocessed_dir, 'val_images.npy')
    val_labels_path = os.path.join(preprocessed_dir, 'val_labels.npy')
    classes = np.load(os.path.join(preprocessed_dir, 'classes.npy'), allow_pickle=True)

    # Training augmentation on preprocessed 256x256 images
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    ])

    # No augmentation for validation
    val_transform = None

    train_dataset = PreprocessedDataset(
        train_images_path, train_labels_path, transform=train_transform
    )

    val_dataset = PreprocessedDataset(
        val_images_path, val_labels_path, transform=val_transform
    )

    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, len(classes)


def create_model(num_classes, device):
    """Create ConvNeXtV2-pico model without pretraining."""

    model = convnext_pico(num_classes=num_classes, drop_path_rate=0.2)
    model = model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch with augmentation support and mixed precision (bfloat16)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # Mixed precision forward pass with bfloat16
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(inputs)
            loss = criterion(logits, targets)

        # Backward pass in full precision (no GradScaler needed for bfloat16)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)

        correct += predicted.eq(targets).sum().item()

        if batch_idx % CONFIG['log_interval'] == 0:
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model with mixed precision (bfloat16)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Mixed precision forward pass with bfloat16
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(inputs)
                loss = criterion(logits, targets)

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def format_learning_rate_tag(learning_rate):
    """Create a filesystem-friendly learning-rate tag."""
    return f"{learning_rate:.0e}".replace("+", "").replace("-", "m")


def run_single_experiment(batch_size, learning_rate):
    """Run one training experiment and write logs to a dedicated file."""
    run_name = f"bs{batch_size}_lr{format_learning_rate_tag(learning_rate)}"
    log_path = os.path.join(CONFIG['logs_dir'], f"{run_name}.log")

    with open(log_path, 'w', encoding='utf-8') as log_file:
        def log(message):
            print(message)
            log_file.write(message + '\n')
            log_file.flush()

        log("=" * 60)
        log(f"Run: {run_name}")
        log("=" * 60)
        log(f"Device: {CONFIG['device']}")
        log(f"Batch size: {batch_size}")
        log(f"Epochs: {CONFIG['num_epochs']}")
        log(f"Learning rate: {learning_rate}")
        log("=" * 60)

        train_loader, val_loader, num_classes = get_data_loaders(
            CONFIG['preprocessed_dir'],
            batch_size,
            CONFIG['num_workers']
        )
        log(f"Number of classes: {num_classes}")
        log(f"Training samples: {len(train_loader.dataset)}")
        log(f"Validation samples: {len(val_loader.dataset)}")

        model = create_model(num_classes, CONFIG['device'])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=CONFIG['weight_decay']
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['num_epochs']
        )

        best_val_acc = 0.0
        training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

        log("\nStarting training...")
        start_time = time.time()

        for epoch in range(1, CONFIG['num_epochs'] + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, CONFIG['device'], epoch
            )
            val_loss, val_acc = validate(
                model, val_loader, criterion, CONFIG['device'], epoch
            )
            scheduler.step()

            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)

            log(f"\nEpoch {epoch}/{CONFIG['num_epochs']} Summary:")
            log(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            log(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            log(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': {
                        **CONFIG,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                    },
                }
                checkpoint_path = os.path.join(
                    CONFIG['save_dir'], f"best_model_{run_name}.pth"
                )
                torch.save(checkpoint, checkpoint_path)
                log(f"  -> New best model saved! (Val Acc: {val_acc:.2f}%)")
                log(f"  Checkpoint: {checkpoint_path}")

            log("-" * 60)

        total_time = time.time() - start_time
        log("\n" + "=" * 60)
        log("Training completed!")
        log(f"Total training time: {total_time / 60:.2f} minutes")
        log(f"Best validation accuracy: {best_val_acc:.2f}%")
        log("=" * 60)

    return {
        'run_name': run_name,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'best_val_acc': best_val_acc,
        'log_path': log_path,
        'history': training_history,
    }


def main():
    print("=" * 60)
    print("ConvNeXtV2 Hyperparameter Sweep on Imagenette")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs per run: {CONFIG['num_epochs']}")
    print(f"Batch sizes: {CONFIG['batch_sizes']}")
    print(f"Learning rates: {CONFIG['learning_rates']}")
    print("=" * 60)

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['logs_dir'], exist_ok=True)

    results = []
    total_runs = len(CONFIG['batch_sizes']) * len(CONFIG['learning_rates'])
    run_idx = 0

    for batch_size in CONFIG['batch_sizes']:
        for learning_rate in CONFIG['learning_rates']:
            run_idx += 1
            print(
                f"\nStarting run {run_idx}/{total_runs} "
                f"(batch_size={batch_size}, learning_rate={learning_rate})"
            )
            result = run_single_experiment(batch_size, learning_rate)
            results.append(result)
            print(f"Run completed. Log file: {result['log_path']}")

    print("\n" + "=" * 60)
    print("Sweep Summary")
    print("=" * 60)
    sorted_results = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)
    for result in sorted_results:
        print(
            f"{result['run_name']}: "
            f"best_val_acc={result['best_val_acc']:.2f}% "
            f"(log: {result['log_path']})"
        )

    return sorted_results


if __name__ == '__main__':
    main()
