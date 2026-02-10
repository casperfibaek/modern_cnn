"""
ConvNeXtV2 Training on Imagenette Dataset
Trains a ConvNeXtV2-base model (not pretrained) on the Imagenette dataset.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
from tqdm import tqdm
import time


# Configuration
CONFIG = {
    'data_dir': './datasets/imagenette2',
    'batch_size': 32,
    'num_epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': './checkpoints',
    'log_interval': 10,
}


def download_imagenette(data_dir):
    """Download and extract Imagenette dataset."""
    import urllib.request
    import tarfile

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"

    if os.path.exists(data_dir):
        print(f"Dataset already exists at {data_dir}")
        return

    print("Downloading Imagenette dataset...")
    os.makedirs(os.path.dirname(data_dir), exist_ok=True)

    tar_path = data_dir + ".tgz"
    urllib.request.urlretrieve(url, tar_path)

    print("Extracting dataset...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(os.path.dirname(data_dir))

    os.remove(tar_path)
    print("Dataset downloaded and extracted successfully!")


def get_data_loaders(data_dir, batch_size, num_workers):
    """Create train and validation data loaders."""

    # ConvNeXtV2 expects 224x224 images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )

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

    return train_loader, val_loader, len(train_dataset.classes)


def create_model(num_classes, device):
    """Create ConvNeXtV2-base model without pretraining."""

    # ConvNeXtV2 base configuration
    config = ConvNextV2Config(
        num_labels=num_classes,
        hidden_sizes=[128, 256, 512, 1024],  # Base model dimensions
        depths=[3, 3, 27, 3],  # Base model depths
        image_size=224,
    )

    model = ConvNextV2ForImageClassification(config)
    model = model.to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.logits.max(1)
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
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.logits, targets)

            running_loss += loss.item()
            _, predicted = outputs.logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': running_loss / (total / CONFIG['batch_size']),
                'acc': 100. * correct / total
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    print("=" * 60)
    print("ConvNeXtV2-Base Training on Imagenette")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print("=" * 60)

    # Download dataset
    download_imagenette(CONFIG['data_dir'])

    # Create data loaders
    train_loader, val_loader, num_classes = get_data_loaders(
        CONFIG['data_dir'],
        CONFIG['batch_size'],
        CONFIG['num_workers']
    )
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = create_model(num_classes, CONFIG['device'])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs']
    )

    # Create checkpoint directory
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    training_history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device'], epoch
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG['device'], epoch
        )

        # Update scheduler
        scheduler.step()

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)

        # Print epoch summary
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG,
            }
            torch.save(checkpoint, os.path.join(CONFIG['save_dir'], 'best_model.pth'))
            print(f"  → New best model saved! (Val Acc: {val_acc:.2f}%)")

        print("-" * 60)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total training time: {total_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)

    return model, training_history


if __name__ == '__main__':
    main()
