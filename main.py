"""
main.py
VerseForge: Shakespeare-Style Autoregressive Text Generation
Course: Foundations of Generative AI (ICT4403) — Assignment 2
Student: Yashraj Sakunde (Reg No: 230953172)

Unified entry point for the VerseForge project.
Supports:
  - Full end-to-end pipeline: data preparation -> training -> evaluation -> generation
  - Individual modes: 'all', 'train', 'evaluate', 'generate'
"""

import sys
import os
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_preprocessing import download_dataset, prepare_data, DATA_FILE
from src.train import train_all_models
from src.evaluate import evaluate_models
from src.generate import generate_text
from src.utils import load_vocab, set_seed
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(
        description="VerseForge: Shakespeare-Style Autoregressive Text Generation"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "train", "evaluate", "generate"],
        default="all",
        help="Execution mode (default: 'all')",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Epochs per architecture")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--step", type=int, default=4, help="Sequence sampling stride")
    parser.add_argument("--seed-text", type=str, default="To be or not to", help="Generation seed phrase")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--model-name", type=str, default="LSTM", help="Model to generate from")
    parser.add_argument("--length", type=int, default=50, help="Number of words to generate")

    args = parser.parse_args()
    set_seed(42)

    print("\n" + "=" * 65)
    print("  VerseForge: Shakespeare-Style Autoregressive Text Generation")
    print("  Foundations of Generative AI (ICT4403) — Assignment 2")
    print("  Author: Yashraj Sakunde | Reg No: 230953172")
    print("=" * 65)

    if args.mode in ["all", "train"]:
        print("\n>>> Phase 1: Downloading Dataset & Training All 4 Architectures...")
        download_dataset()
        train_results = train_all_models(
            epochs=args.epochs,
            batch_size=args.batch_size,
            step=args.step,
        )

    if args.mode in ["all", "evaluate"]:
        print("\n>>> Phase 2: Quantitative & Qualitative Comparative Evaluation...")
        eval_results = evaluate_models()
        print("\n[Complete] Model Comparison Summary:")
        for r in eval_results["quantitative"]:
            print(
                f"  {r['model']:<18} | Loss: {r['test_loss']:.4f} | "
                f"Perplexity: {r['test_perplexity']:.2f} | Acc: {r['test_accuracy']*100:.2f}% | "
                f"Time: {r['training_time_sec']:.1f}s"
            )
        print(f"\nBest Overall Model: {eval_results['best_model']}")

    if args.mode == "generate":
        print(f"\n>>> Phase: Text Generation using {args.model_name}...")
        model_path = os.path.join("models", f"{args.model_name}.keras")
        if not os.path.exists(model_path):
            print(f"[Error] Model checkpoint not found at {model_path}. Train the model first.")
            return

        token_to_id, id_to_token = load_vocab("data/vocab.json")
        model = keras.models.load_model(model_path)
        out = generate_text(
            model=model,
            seed_text=args.seed_text,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            num_words_to_generate=args.length,
            temperature=args.temperature,
            sampling_seed=42,
        )
        print(f"\nSeed: \"{args.seed_text}\"")
        print(f"Temperature: {args.temperature}")
        print(f"Generated text:\n{out}\n")


if __name__ == "__main__":
    main()
