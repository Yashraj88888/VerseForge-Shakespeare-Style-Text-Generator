"""
generate.py
VerseForge: Shakespeare-Style Autoregressive Text Generation
Implements genuine autoregressive next-word prediction and sequential generation
with numerically stable temperature-controlled sampling.
"""

import os
import sys
from typing import List, Dict, Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data_preprocessing import tokenize_text, tokens_to_ids, PAD_TOKEN, UNK_TOKEN


def sample_with_temperature(
    probabilities: np.ndarray,
    temperature: float = 1.0,
    seed: Optional[int] = None,
) -> int:
    """
    Samples an index from a categorical distribution with temperature scaling.
    Applies numerically stable log-softmax:
      scaled_logits = log(p + eps) / T
      scaled_logits -= max(scaled_logits)
      probs = exp(scaled_logits) / sum(exp(scaled_logits))
    """
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")

    if temperature < 1e-2:
        # Near-zero temperature approximates deterministic greedy argmax
        return int(np.argmax(probabilities))

    # Convert probabilities to log-space with epsilon smoothing
    eps = 1e-9
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = np.clip(probs, eps, 1.0)
    logits = np.log(probs) / temperature

    # Numerical stabilization: subtract maximum logit to avoid overflow
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    scaled_probs = exp_logits / np.sum(exp_logits)

    # Renormalize to ensure sum == 1.0 exactly
    scaled_probs = scaled_probs / np.sum(scaled_probs)

    if seed is not None:
        rng = np.random.default_rng(seed)
        return int(rng.choice(len(scaled_probs), p=scaled_probs))
    else:
        return int(np.random.choice(len(scaled_probs), p=scaled_probs))


def generate_text(
    model: keras.Model,
    seed_text: str,
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    seq_length: int = 25,
    num_words_to_generate: int = 50,
    temperature: float = 1.0,
    sampling_seed: Optional[int] = None,
) -> str:
    """
    Autoregressively generates text word-by-word from a seed phrase.
    Process:
      1. Tokenizes seed phrase identically to training.
      2. Pads or truncates to fixed context window length.
      3. Predicts next-word probability distribution.
      4. Samples next word with temperature.
      5. Appends word, updates context, and repeats.
    """
    pad_id = token_to_id.get(PAD_TOKEN, 0)
    
    # Tokenize seed text consistently
    seed_tokens = tokenize_text(seed_text)
    if not seed_tokens:
        seed_tokens = ["thou"]  # Fallback non-empty seed
    
    current_ids = tokens_to_ids(seed_tokens, token_to_id)
    generated_tokens = list(seed_tokens)

    for step in range(num_words_to_generate):
        # Prepare context window of length seq_length
        if len(current_ids) < seq_length:
            context = [pad_id] * (seq_length - len(current_ids)) + current_ids
        else:
            context = current_ids[-seq_length:]

        input_batch = np.array([context], dtype=np.int32)
        
        # Model forward pass -> output shape (1, vocab_size)
        preds = model(input_batch, training=False).numpy()[0]
        
        step_seed = (sampling_seed + step) if sampling_seed is not None else None
        next_id = sample_with_temperature(preds, temperature=temperature, seed=step_seed)
        
        next_word = id_to_token.get(next_id, UNK_TOKEN)
        generated_tokens.append(next_word)
        current_ids.append(next_id)

    # Detokenize with clean punctuation spacing
    text_out = ""
    for token in generated_tokens:
        if token in {".", ",", "!", "?", ";", ":", "'"} or text_out == "":
            text_out += token
        else:
            text_out += " " + token

    return text_out


def run_sampling_suite(
    models: Dict[str, keras.Model],
    token_to_id: Dict[str, int],
    id_to_token: Dict[int, str],
    seeds: List[str],
    temperatures: List[float] = [0.5, 0.8, 1.2],
    num_words: int = 50,
    output_dir: str = "results/generated_samples",
    base_seed: int = 42,
) -> Dict[str, Dict[float, List[Dict[str, str]]]]:
    """
    Generates text across all models, seeds, and temperatures.
    Saves formatted output files to results/generated_samples/.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_samples = {}

    for model_name, model in models.items():
        all_samples[model_name] = {}
        model_out_path = os.path.join(output_dir, f"{model_name}_samples.txt")
        
        with open(model_out_path, "w", encoding="utf-8") as f:
            f.write(f"==========================================================\n")
            f.write(f"VerseForge Generated Samples: {model_name}\n")
            f.write(f"==========================================================\n\n")

            for temp in temperatures:
                all_samples[model_name][temp] = []
                f.write(f"--- Sampling Temperature: {temp} ---\n\n")
                
                for s_idx, seed in enumerate(seeds):
                    gen_text = generate_text(
                        model=model,
                        seed_text=seed,
                        token_to_id=token_to_id,
                        id_to_token=id_to_token,
                        num_words_to_generate=num_words,
                        temperature=temp,
                        sampling_seed=base_seed + s_idx * 100,
                    )
                    all_samples[model_name][temp].append({
                        "seed": seed,
                        "generated_text": gen_text
                    })
                    f.write(f"Seed: \"{seed}\"\n")
                    f.write(f"Output: {gen_text}\n\n")
                f.write("\n")

    return all_samples
