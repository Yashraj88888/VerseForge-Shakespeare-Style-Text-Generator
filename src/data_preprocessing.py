"""
data_preprocessing.py
VerseForge: Shakespeare-Style Autoregressive Text Generation
Handles dataset downloading, inspection, word-level tokenization,
vocabulary building (fit strictly on training data), chronological splitting,
and input-target sequence construction.
"""

import os
import re
import urllib.request
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "shakespeare.txt")

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def download_dataset(url: str = DATA_URL, destination: str = DATA_FILE) -> str:
    """
    Downloads the Tiny Shakespeare dataset programmatically.
    Uses requests with fallback to urllib and SSL unverified context if needed on macOS.
    If downloading fails, raises an informative error with manual download instructions.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        print(f"[Dataset] Already present at {destination} ({os.path.getsize(destination):,} bytes).")
        return destination

    print(f"[Dataset] Downloading Tiny Shakespeare from {url} to {destination}...")
    try:
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(destination, "w", encoding="utf-8") as f:
            f.write(response.text)
        size = os.path.getsize(destination)
        print(f"[Dataset] Download successful via requests! Saved {size:,} bytes to {destination}.")
        return destination
    except Exception as e_req:
        try:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as response, open(destination, "wb") as out_file:
                out_file.write(response.read())
            size = os.path.getsize(destination)
            print(f"[Dataset] Download successful via urllib SSL context! Saved {size:,} bytes to {destination}.")
            return destination
        except Exception as e:
            error_msg = (
                f"[Error] Failed to download dataset automatically: {e_req}; {e}\n"
                f"Manual Download Instructions:\n"
                f"1. Download input.txt from: {url}\n"
                f"2. Save the file to: {destination}\n"
            )
            raise RuntimeError(error_msg)


def load_raw_text(file_path: str = DATA_FILE) -> str:
    """Loads raw text from the specified path."""
    if not os.path.exists(file_path):
        download_dataset(destination=file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def tokenize_text(text: str, lowercase: bool = True) -> List[str]:
    """
    Tokenizes raw text at the word level while preserving punctuation tokens.
    Normalizes consecutive whitespace.
    """
    if lowercase:
        text = text.lower()
    # Normalize excessive newlines and whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Tokenize words, numbers, and separate punctuation marks into distinct tokens
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text)
    return tokens


def build_vocabulary(
    train_tokens: List[str],
    max_vocab_size: int = 8000,
    min_freq: int = 2
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Constructs vocabulary strictly on the training tokens to prevent data leakage.
    Assigns:
      0 -> PAD_TOKEN
      1 -> UNK_TOKEN
    Followed by tokens sorted by frequency.
    """
    counts = Counter(train_tokens)
    filtered = [token for token, count in counts.most_common() if count >= min_freq]
    
    # Reserve top tokens up to max_vocab_size - 2
    top_tokens = filtered[: max_vocab_size - 2]

    token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for idx, token in enumerate(top_tokens, start=2):
        token_to_id[token] = idx

    id_to_token = {idx: token for token, idx in token_to_id.items()}
    return token_to_id, id_to_token


def tokens_to_ids(tokens: List[str], token_to_id: Dict[str, int]) -> List[int]:
    """Maps token strings to vocabulary IDs, replacing unseen tokens with <UNK>."""
    unk_id = token_to_id[UNK_TOKEN]
    return [token_to_id.get(t, unk_id) for t in tokens]


def create_sequences(
    token_ids: List[int],
    seq_length: int = 25,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates fixed-length input sequences and next-word targets:
      X[i] = [token_ids[i], token_ids[i+1], ..., token_ids[i+seq_length-1]]
      y[i] = token_ids[i+seq_length]
    """
    X_list = []
    y_list = []
    for i in range(0, len(token_ids) - seq_length, step):
        X_list.append(token_ids[i : i + seq_length])
        y_list.append(token_ids[i + seq_length])

    return np.array(X_list, dtype=np.int32), np.array(y_list, dtype=np.int32)


def prepare_data(
    file_path: str = DATA_FILE,
    seq_length: int = 25,
    step: int = 3,
    max_vocab_size: int = 8000,
    min_freq: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict:
    """
    Complete preprocessing pipeline:
    1. Downloads & loads raw text.
    2. Reports character, word, and token statistics.
    3. Tokenizes text into words and punctuation tokens.
    4. Chronologically splits raw tokens into Train (80%), Val (10%), Test (10%).
    5. Fits vocabulary ONLY on Train tokens (no data leakage).
    6. Converts tokens to IDs and constructs fixed-length input-target pairs.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Splits must sum to 1.0"
    
    text = load_raw_text(file_path)
    char_count = len(text)
    raw_word_count = len(text.split())

    all_tokens = tokenize_text(text)
    total_tokens = len(all_tokens)

    # Chronological partition points
    train_end = int(total_tokens * train_ratio)
    val_end = int(total_tokens * (train_ratio + val_ratio))

    train_tokens = all_tokens[:train_end]
    val_tokens = all_tokens[train_end:val_end]
    test_tokens = all_tokens[val_end:]

    # Fit vocabulary strictly on train_tokens
    token_to_id, id_to_token = build_vocabulary(
        train_tokens, max_vocab_size=max_vocab_size, min_freq=min_freq
    )
    vocab_size = len(token_to_id)

    # Encode tokens
    train_ids = tokens_to_ids(train_tokens, token_to_id)
    val_ids = tokens_to_ids(val_tokens, token_to_id)
    test_ids = tokens_to_ids(test_tokens, token_to_id)

    # Construct sequences
    X_train, y_train = create_sequences(train_ids, seq_length=seq_length, step=step)
    X_val, y_val = create_sequences(val_ids, seq_length=seq_length, step=step)
    X_test, y_test = create_sequences(test_ids, seq_length=seq_length, step=step)

    stats = {
        "raw_characters": char_count,
        "raw_words": raw_word_count,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "train_tokens_count": len(train_tokens),
        "val_tokens_count": len(val_tokens),
        "test_tokens_count": len(test_tokens),
        "train_sequences": len(X_train),
        "val_sequences": len(X_val),
        "test_sequences": len(X_test),
        "seq_length": seq_length,
        "step": step,
    }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "stats": stats,
    }


if __name__ == "__main__":
    download_dataset()
    data = prepare_data()
    print("\n--- Preprocessing Dataset Statistics ---")
    for k, v in data["stats"].items():
        print(f"  {k}: {v}")
    print("\nSample sequence alignment:")
    print("X_train[0]:", data["X_train"][0])
    print("y_train[0]:", data["y_train"][0])
    sample_words = [data["id_to_token"][i] for i in data["X_train"][0]]
    target_word = data["id_to_token"][data["y_train"][0]]
    print("Input words:", " ".join(sample_words))
    print("Target word:", target_word)
