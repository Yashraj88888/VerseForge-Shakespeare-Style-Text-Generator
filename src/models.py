"""
models.py
VerseForge: Shakespeare-Style Autoregressive Text Generation
Defines four neural network architectures in TensorFlow/Keras:
  1. Simple RNN: Embedding -> SimpleRNN -> Dense(Softmax)
  2. Bidirectional RNN: Embedding -> Bidirectional(SimpleRNN) -> Dense(Softmax)
  3. LSTM: Embedding -> LSTM -> Dense(Softmax)
  4. GRU: Embedding -> GRU -> Dense(Softmax)
"""

from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_simple_rnn(
    vocab_size: int,
    seq_length: int = 25,
    embedding_dim: int = 64,
    hidden_units: int = 128,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Constructs a baseline Simple Recurrent Neural Network.
    Architecture:
      Input (seq_length,) -> Embedding(vocab_size, embedding_dim)
      -> SimpleRNN(hidden_units) -> Dense(vocab_size, activation='softmax')
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_length,), dtype=tf.int32, name="input_sequence"),
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="word_embedding",
            ),
            layers.SimpleRNN(hidden_units, return_sequences=False, name="simple_rnn"),
            layers.Dense(vocab_size, activation="softmax", name="next_word_projection"),
        ],
        name="Simple_RNN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_bidirectional_rnn(
    vocab_size: int,
    seq_length: int = 25,
    embedding_dim: int = 64,
    hidden_units: int = 128,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Constructs a Bidirectional Simple RNN for context encoding.
    
    IMPORTANT NOTE ON AUTOREGRESSIVE VALIDITY:
    The Bidirectional RNN operates exclusively on the historical context sequence
    [w_1, w_2, ..., w_T] and never has access to the target word w_{T+1} or future tokens.
    The forward pass reads w_1 -> w_T and the backward pass reads w_T -> w_1 within the
    observed historical window. The concatenated hidden state at the final context boundary
    is projected to predict the next word w_{T+1}. This preserves causal autoregressive
    integrity while allowing bidirectional feature extraction across the known prompt.
    """
    # 64 units per direction => 128 concatenated units, keeping total capacity comparable
    half_units = hidden_units // 2
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_length,), dtype=tf.int32, name="input_sequence"),
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="word_embedding",
            ),
            layers.Bidirectional(
                layers.SimpleRNN(half_units, return_sequences=False),
                name="bidirectional_rnn",
            ),
            layers.Dense(vocab_size, activation="softmax", name="next_word_projection"),
        ],
        name="Bidirectional_RNN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(
    vocab_size: int,
    seq_length: int = 25,
    embedding_dim: int = 64,
    hidden_units: int = 128,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Constructs a Long Short-Term Memory (LSTM) network.
    Architecture:
      Input (seq_length,) -> Embedding(vocab_size, embedding_dim)
      -> LSTM(hidden_units) -> Dense(vocab_size, activation='softmax')
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_length,), dtype=tf.int32, name="input_sequence"),
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="word_embedding",
            ),
            layers.LSTM(hidden_units, return_sequences=False, name="lstm_layer"),
            layers.Dense(vocab_size, activation="softmax", name="next_word_projection"),
        ],
        name="LSTM",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru(
    vocab_size: int,
    seq_length: int = 25,
    embedding_dim: int = 64,
    hidden_units: int = 128,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Constructs a Gated Recurrent Unit (GRU) network.
    Architecture:
      Input (seq_length,) -> Embedding(vocab_size, embedding_dim)
      -> GRU(hidden_units) -> Dense(vocab_size, activation='softmax')
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(seq_length,), dtype=tf.int32, name="input_sequence"),
            layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                name="word_embedding",
            ),
            layers.GRU(hidden_units, return_sequences=False, name="gru_layer"),
            layers.Dense(vocab_size, activation="softmax", name="next_word_projection"),
        ],
        name="GRU",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


MODEL_BUILDERS = {
    "Simple_RNN": build_simple_rnn,
    "Bidirectional_RNN": build_bidirectional_rnn,
    "LSTM": build_lstm,
    "GRU": build_gru,
}


def get_model(
    model_name: str,
    vocab_size: int,
    seq_length: int = 25,
    embedding_dim: int = 64,
    hidden_units: int = 128,
    learning_rate: float = 0.001,
) -> keras.Model:
    """Instantiates and returns the specified compiled model architecture."""
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model name '{model_name}'. Choose from: {list(MODEL_BUILDERS.keys())}"
        )
    return MODEL_BUILDERS[model_name](
        vocab_size=vocab_size,
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
