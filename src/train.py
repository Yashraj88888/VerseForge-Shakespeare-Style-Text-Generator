"""
train.py
VerseForge: Shakespeare-Style Autoregressive Text Generation
Handles end-to-end model training for all four architectures:
  1. Simple RNN
  2. Bidirectional RNN
  3. LSTM
  4. GRU
Tracks training duration, epoch-wise loss/accuracy, saves best checkpoints,
and serializes training histories for comparative evaluation and plotting.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data_preprocessing import prepare_data, DATA_FILE
from src.models import get_model, MODEL_BUILDERS
from src.utils import set_seed, save_vocab


def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    vocab_size: int,
    seq_length: int = 25,
    embedding_dim: int = 64,
    hidden_units: int = 128,
    learning_rate: float = 0.001,
    epochs: int = 10,
    batch_size: int = 64,
    patience: int = 3,
    models_dir: str = "models",
) -> Tuple[keras.Model, Dict[str, Any]]:
    """
    Builds, compiles, and trains a single model architecture with EarlyStopping
    and ModelCheckpoint. Records exact training duration and best epoch.
    """
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, f"{model_name}.keras")

    print(f"\n{'='*60}")
    print(f"  Training Model: {model_name}")
    print(f"{'='*60}")
    print(f"Architecture: {model_name} | Hidden: {hidden_units} | Embed: {embedding_dim}")
    print(f"Train samples: {len(X_train):,} | Val samples: {len(X_val):,} | Batch: {batch_size}")

    model = get_model(
        model_name=model_name,
        vocab_size=vocab_size,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed_time = time.time() - start_time

    val_losses = history.history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss = float(val_losses[best_epoch - 1])

    print(f"\n[Completed] {model_name} trained in {elapsed_time:.1f}s")
    print(f"Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")

    hist_data = {
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "training_time_sec": elapsed_time,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(val_losses),
    }

    # Save history json
    hist_path = os.path.join(models_dir, f"{model_name}_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(hist_data, f, indent=2)

    return model, hist_data


def train_all_models(
    seq_length: int = 25,
    step: int = 4,
    embedding_dim: int = 64,
    hidden_units: int = 128,
    learning_rate: float = 0.001,
    epochs: int = 8,
    batch_size: int = 64,
    patience: int = 3,
    models_dir: str = "models",
    vocab_dir: str = "data",
) -> Dict[str, Any]:
    """
    Prepares dataset and sequentially trains all four models under identical conditions:
    Simple RNN, Bidirectional RNN, LSTM, and GRU.
    """
    set_seed(42)
    print("\n[VerseForge] Preparing dataset and tokenizing...")
    data = prepare_data(
        seq_length=seq_length,
        step=step,
        max_vocab_size=6000,
        min_freq=2,
    )

    token_to_id = data["token_to_id"]
    id_to_token = data["id_to_token"]
    vocab_size = len(token_to_id)
    vocab_path = os.path.join(vocab_dir, "vocab.json")
    save_vocab(token_to_id, vocab_path)
    print(f"[VerseForge] Saved vocabulary ({vocab_size} tokens) to {vocab_path}")

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Save test data for evaluation
    np.savez_compressed(
        os.path.join(models_dir, "test_data.npz"),
        X_test=X_test,
        y_test=y_test,
    )

    all_histories = {}
    trained_models = {}

    for model_name in MODEL_BUILDERS.keys():
        model, hist = train_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            vocab_size=vocab_size,
            seq_length=seq_length,
            embedding_dim=embedding_dim,
            hidden_units=hidden_units,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            models_dir=models_dir,
        )
        all_histories[model_name] = hist
        trained_models[model_name] = model

    return {
        "models": trained_models,
        "histories": all_histories,
        "data": data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VerseForge RNN models")
    parser.add_argument("--epochs", type=int, default=8, help="Maximum epochs per model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--step", type=int, default=4, help="Subsampling stride step for sequences")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--hidden-units", type=int, default=128, help="Hidden recurrent units")
    args = parser.parse_args()

    train_all_models(
        epochs=args.epochs,
        batch_size=args.batch_size,
        step=args.step,
        embedding_dim=args.embed_dim,
        hidden_units=args.hidden_units,
    )
