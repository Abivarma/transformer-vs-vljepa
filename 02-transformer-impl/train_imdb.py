"""
Training script for IMDB Sentiment Analysis using Transformer.

This script demonstrates training the Transformer model on the IMDB dataset
for binary sentiment classification (positive/negative reviews).

Target: >80% accuracy on test set
"""

import re
import time
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformer import TransformerClassifier


class IMDBDataset(Dataset):
    """Simple IMDB dataset wrapper."""

    def __init__(self, texts: List[str], labels: List[int], vocab, max_len: int = 256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and convert to indices
        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens[: self.max_len]]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text.split()


def build_vocab(texts: List[str], max_vocab_size: int = 10000) -> dict:
    """Build vocabulary from texts."""
    counter = Counter()
    for text in texts:
        tokens = IMDBDataset.tokenize(text)
        counter.update(tokens)

    # Get most common words
    most_common = counter.most_common(max_vocab_size - 2)  # Reserve for <PAD> and <UNK>

    # Create vocabulary
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx

    return vocab


def collate_fn(batch):
    """Collate function for DataLoader."""
    texts, labels = zip(*batch)

    # Pad sequences
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return texts_padded, labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (texts, labels) in enumerate(dataloader):
        texts, labels = texts.to(device), labels.to(device)

        # Create padding mask
        mask = TransformerClassifier.create_padding_mask(texts, pad_idx=0)

        # Forward pass
        optimizer.zero_grad()
        logits = model(texts, mask)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)

            # Create padding mask
            mask = TransformerClassifier.create_padding_mask(texts, pad_idx=0)

            # Forward pass
            logits = model(texts, mask)
            loss = criterion(logits, labels)

            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    """Main training function."""
    # Hyperparameters
    MAX_VOCAB_SIZE = 10000
    MAX_SEQ_LEN = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.0001
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    D_FF = 1024
    DROPOUT = 0.1

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Load IMDB dataset using PyTorch's built-in dataset
    print("Loading IMDB dataset...")
    try:
        from torchtext.data.utils import get_tokenizer
        from torchtext.datasets import IMDB

        # Load dataset
        train_iter = IMDB(split="train")
        test_iter = IMDB(split="test")

        # Extract texts and labels
        train_texts, train_labels = [], []
        for label, text in train_iter:
            train_texts.append(text)
            train_labels.append(1 if label == "pos" else 0)

        test_texts, test_labels = [], []
        for label, text in test_iter:
            test_texts.append(text)
            test_labels.append(1 if label == "pos" else 0)

    except ImportError:
        print("torchtext not available. Using dummy data for demonstration.")
        # Create dummy data for demonstration
        train_texts = [
            "This movie is great and fantastic",
            "I love this film very much",
            "Terrible movie waste of time",
            "Awful and boring film",
        ] * 100
        train_labels = [1, 1, 0, 0] * 100

        test_texts = train_texts[:40]
        test_labels = train_labels[:40]

    print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")

    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_texts, MAX_VOCAB_SIZE)
    print(f"Vocabulary size: {len(vocab)}")

    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, MAX_SEQ_LEN)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, MAX_SEQ_LEN)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Create model
    print("Creating model...")
    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_classes=2,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nStarting training...")
    best_test_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)

        # Train
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - start_time

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Time: {train_time:.2f}s")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "02-transformer-impl/best_model.pt")
            print(f"Saved best model with test accuracy: {best_test_acc:.4f}")

    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc * 100:.2f}%)")

    # Check if we achieved the target
    if best_test_acc >= 0.80:
        print("✅ Target achieved: >80% accuracy on IMDB test set!")
    else:
        print(f"⚠️ Target not yet achieved. Current: {best_test_acc * 100:.2f}%, Target: 80%")

    return best_test_acc


if __name__ == "__main__":
    main()
