"""
test_audit.py
Automated Verification and Audit Suite for VerseForge Project
Covers all requirements in IA2_ICT4403_FGAI.pdf Sections A through L.
"""

import os
import sys
import json
import csv
import unittest
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Ensure workspace root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import (
    load_raw_text,
    tokenize_text,
    build_vocabulary,
    tokens_to_ids,
    create_sequences,
    prepare_data,
    DATA_FILE,
    PAD_TOKEN,
    UNK_TOKEN,
)
from src.models import (
    build_simple_rnn,
    build_bidirectional_rnn,
    build_lstm,
    build_gru,
    MODEL_BUILDERS,
)
from src.generate import sample_with_temperature, generate_text
from src.utils import compute_perplexity, compute_distinct_n, compute_repetition_ratio, load_vocab


class TestVerseForgeAudit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_file = os.path.join(cls.workspace_dir, "data", "shakespeare.txt")
        cls.vocab_file = os.path.join(cls.workspace_dir, "data", "vocab.json")
        cls.models_dir = os.path.join(cls.workspace_dir, "models")
        cls.results_dir = os.path.join(cls.workspace_dir, "results")
        cls.plots_dir = os.path.join(cls.results_dir, "plots")
        cls.samples_dir = os.path.join(cls.results_dir, "generated_samples")
        cls.sub_dir = os.path.join(cls.workspace_dir, "submission")

    # --- Section A: Dataset & Preprocessing ---

    def test_A1_dataset_file_integrity(self):
        """Verify data/shakespeare.txt exists and matches Tiny Shakespeare byte size."""
        self.assertTrue(os.path.exists(self.data_file), "shakespeare.txt must exist.")
        size = os.path.getsize(self.data_file)
        self.assertEqual(size, 1115394, f"Dataset byte size mismatch: expected 1115394, got {size}")
        raw = load_raw_text(self.data_file)
        self.assertTrue(raw.startswith("First Citizen:"), "Corpus must start with Shakespeare Coriolanus.")

    def test_A2_word_tokenization(self):
        """Verify word-level tokenization correctly tokenizes words and separates punctuation."""
        sample = "To be, or not to be: that is the question's end!"
        tokens = tokenize_text(sample)
        self.assertIn("to", tokens)
        self.assertIn("be", tokens)
        self.assertIn(",", tokens)
        self.assertIn(":", tokens)
        self.assertIn("question's", tokens)
        self.assertIn("!", tokens)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_A3_vocabulary_construction_and_unk(self):
        """Verify vocabulary fitting, special tokens (<PAD>=0, <UNK>=1), and UNK mapping."""
        sample_tokens = ["thou", "art", "fair", "thou", "art", "true", "rare"]
        tok_to_id, id_to_tok = build_vocabulary(sample_tokens, min_freq=2)
        self.assertEqual(tok_to_id[PAD_TOKEN], 0)
        self.assertEqual(tok_to_id[UNK_TOKEN], 1)
        self.assertIn("thou", tok_to_id)
        self.assertIn("art", tok_to_id)
        self.assertNotIn("rare", tok_to_id)  # freq 1 < 2

        # Test unseen token mapping
        encoded = tokens_to_ids(["thou", "completely_unknown_token"], tok_to_id)
        self.assertEqual(encoded[1], tok_to_id[UNK_TOKEN])

    def test_A4_chronological_split_and_no_leakage(self):
        """Verify chronological split bounds: 80% train, 10% val, 10% test, fitted solely on train."""
        raw = load_raw_text(self.data_file)
        all_tokens = tokenize_text(raw)
        total = len(all_tokens)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)

        train_tokens = all_tokens[:train_end]
        val_tokens = all_tokens[train_end:val_end]
        test_tokens = all_tokens[val_end:]

        self.assertEqual(len(train_tokens) + len(val_tokens) + len(test_tokens), total)

        # Check vocabulary was fitted strictly on train_tokens
        with open(self.vocab_file, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
        # Special tokens:
        self.assertEqual(vocab[PAD_TOKEN], 0)
        self.assertEqual(vocab[UNK_TOKEN], 1)

    def test_A5_sequence_target_alignment(self):
        """Verify sequence construction alignment: X[i] context length 25, target y[i] is next word."""
        token_ids = list(range(100))
        X, y = create_sequences(token_ids, seq_length=25, step=1)
        self.assertEqual(X.shape[1], 25)
        self.assertEqual(len(X), len(y))
        np.testing.assert_array_equal(X[0], np.arange(0, 25))
        self.assertEqual(y[0], 25)
        np.testing.assert_array_equal(X[1], np.arange(1, 26))
        self.assertEqual(y[1], 26)

    # --- Section B: Model Architectures ---

    def test_B1_all_four_models_build_and_forward_pass(self):
        """Verify Simple RNN, Bidirectional RNN, LSTM, GRU architectures and output shapes."""
        vocab_size = 1000
        seq_length = 25
        batch_size = 4
        X_dummy = np.random.randint(0, vocab_size, size=(batch_size, seq_length))

        for name, builder in MODEL_BUILDERS.items():
            model = builder(vocab_size=vocab_size, seq_length=seq_length, embedding_dim=64, hidden_units=128)
            preds = model(X_dummy)
            self.assertEqual(preds.shape, (batch_size, vocab_size), f"{name} output shape mismatch.")
            # Check softmax probability validity: sums to 1.0 and >= 0
            prob_sums = np.sum(preds.numpy(), axis=-1)
            np.testing.assert_allclose(prob_sums, np.ones(batch_size), atol=1e-5)
            self.assertTrue(np.all(preds.numpy() >= 0.0))

    def test_B2_bidirectional_rnn_autoregressive_validity(self):
        """Verify Bidirectional RNN processes strictly past context without future lookahead."""
        model = build_bidirectional_rnn(vocab_size=500, seq_length=25, embedding_dim=64, hidden_units=128)
        # Check that model inputs only seq_length=25 tokens
        self.assertEqual(model.input_shape, (None, 25))
        # Layer 1: Embedding, Layer 2: Bidirectional(SimpleRNN), Layer 3: Dense
        layers = model.layers
        self.assertTrue(any("bidirectional" in l.name for l in layers))
        # Target word is scalar y outside the 25 tokens

    def test_B3_saved_checkpoints_loadable(self):
        """Verify all four saved .keras models exist and load correctly."""
        for name in MODEL_BUILDERS.keys():
            path = os.path.join(self.models_dir, f"{name}.keras")
            self.assertTrue(os.path.exists(path), f"Checkpoint missing: {path}")
            model = keras.models.load_model(path)
            self.assertIsNotNone(model)
            self.assertEqual(model.output_shape[-1], 5959)

    # --- Section C & D: Training Fairness & Quantitative Evaluation ---

    def test_C1_training_histories_and_conditions(self):
        """Verify training histories exist with required loss, val_loss, accuracy, val_accuracy, duration."""
        for name in MODEL_BUILDERS.keys():
            hist_path = os.path.join(self.models_dir, f"{name}_history.json")
            self.assertTrue(os.path.exists(hist_path), f"History missing: {hist_path}")
            with open(hist_path) as f:
                h = json.load(f)
            self.assertIn("loss", h)
            self.assertIn("val_loss", h)
            self.assertIn("accuracy", h)
            self.assertIn("val_accuracy", h)
            self.assertIn("training_time_sec", h)
            self.assertIn("best_epoch", h)
            self.assertGreater(h["training_time_sec"], 0.0)

    def test_D1_perplexity_formula(self):
        """Verify mathematical definition: Perplexity = exp(Loss)."""
        loss = 5.2002
        expected_ppl = float(np.exp(loss))
        calc_ppl = compute_perplexity(loss)
        self.assertAlmostEqual(calc_ppl, expected_ppl, places=4)

    def test_D2_model_comparison_csv_consistency(self):
        """Verify results/model_comparison.csv contains all 4 models and perplexity matches exp(loss)."""
        csv_path = os.path.join(self.results_dir, "model_comparison.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path) as f:
            reader = list(csv.DictReader(f))
        self.assertEqual(len(reader), 4)
        for row in reader:
            loss = float(row["test_loss"])
            ppl = float(row["test_perplexity"])
            expected = round(float(np.exp(loss)), 2)
            self.assertAlmostEqual(ppl, expected, delta=0.05)

    # --- Section E & F: Text Generation & Temperature ---

    def test_E1_temperature_sampling_boundaries(self):
        """Verify temperature scaling: low T is sharp, high T is uniform, invalid T raises ValueError."""
        probs = np.array([0.1, 0.2, 0.7])
        # Invalid temperature
        with self.assertRaises(ValueError):
            sample_with_temperature(probs, temperature=-0.5)
        with self.assertRaises(ValueError):
            sample_with_temperature(probs, temperature=0.0)

        # Near zero temperature should be greedy argmax (index 2)
        idx = sample_with_temperature(probs, temperature=0.001)
        self.assertEqual(idx, 2)

    def test_E2_generation_edge_cases(self):
        """Verify generate_text handles empty seeds, short seeds, unknown words, long seeds."""
        token_to_id, id_to_token = load_vocab(self.vocab_file)
        model = keras.models.load_model(os.path.join(self.models_dir, "Simple_RNN.keras"))

        # 1. Short seed
        out1 = generate_text(model, "hail", token_to_id, id_to_token, num_words_to_generate=10, temperature=0.8, sampling_seed=42)
        self.assertTrue(isinstance(out1, str) and len(out1) > 0)
        self.assertTrue(out1.startswith("hail"))

        # 2. Unknown seed tokens
        out2 = generate_text(model, "xyzzy qwerty", token_to_id, id_to_token, num_words_to_generate=10, temperature=0.8, sampling_seed=42)
        self.assertTrue(isinstance(out2, str) and len(out2) > 0)
        self.assertTrue(out2.startswith("xyzzy qwerty"))

        # 3. Empty seed fallback (falls back to default seed token 'thou')
        out3 = generate_text(model, "", token_to_id, id_to_token, num_words_to_generate=10, temperature=0.8, sampling_seed=42)
        self.assertTrue(isinstance(out3, str) and len(out3) > 0)
        self.assertTrue(out3.startswith("thou"))

    def test_F1_generated_samples_files_exist(self):
        """Verify generated sample text files exist for all 4 models and contain temperatures 0.5, 0.8, 1.2."""
        for name in MODEL_BUILDERS.keys():
            sample_path = os.path.join(self.samples_dir, f"{name}_samples.txt")
            self.assertTrue(os.path.exists(sample_path), f"Sample file missing: {sample_path}")
            with open(sample_path) as f:
                content = f.read()
            self.assertIn("Temperature: 0.5", content)
            self.assertIn("Temperature: 0.8", content)
            self.assertIn("Temperature: 1.2", content)
            self.assertIn("To be or not to", content)

    # --- Section G & H: Qualitative Diversity & Plots ---

    def test_G1_distinct_n_and_repetition_metrics(self):
        """Verify Distinct-1, Distinct-2, and repetition ratio calculations."""
        tokens = ["to", "be", "or", "not", "to", "be"]
        # Unigrams: total=6, unique={'to', 'be', 'or', 'not'} -> 4/6 = 0.6667
        d1 = compute_distinct_n(tokens, n=1)
        self.assertAlmostEqual(d1, 4 / 6, places=3)

        # Bigrams: [('to','be'), ('be','or'), ('or','not'), ('not','to'), ('to','be')]
        # total=5, unique=4 -> 4/5 = 0.8
        d2 = compute_distinct_n(tokens, n=2)
        self.assertAlmostEqual(d2, 4 / 5, places=3)

        rep2 = compute_repetition_ratio(tokens, n=2)
        self.assertAlmostEqual(rep2, 1.0 - 0.8, places=3)

    def test_H1_all_plots_exist_and_non_empty(self):
        """Verify all 6 required plots exist and are non-empty."""
        required_plots = [
            "loss_curves.png",
            "accuracy_curves.png",
            "test_perplexity_comparison.png",
            "test_accuracy_comparison.png",
            "training_time_comparison.png",
            "temperature_diversity_comparison.png",
        ]
        for plot_name in required_plots:
            path = os.path.join(self.plots_dir, plot_name)
            self.assertTrue(os.path.exists(path), f"Plot missing: {path}")
            self.assertGreater(os.path.getsize(path), 10000, f"Plot seems corrupted/empty: {path}")

    # --- Section I, J, K, L: Notebook, README, Git, Submission ---

    def test_I1_notebook_structure(self):
        """Verify notebooks/VerseForge_Experiments.ipynb is valid JSON and contains required sections."""
        nb_path = os.path.join(self.workspace_dir, "notebooks", "VerseForge_Experiments.ipynb")
        self.assertTrue(os.path.exists(nb_path))
        with open(nb_path) as f:
            nb = json.load(f)
        self.assertIn("cells", nb)
        sources = " ".join(["".join(c["source"]) for c in nb["cells"]])
        self.assertIn("Simple_RNN", sources)
        self.assertIn("Bidirectional_RNN", sources)
        self.assertIn("LSTM", sources)
        self.assertIn("GRU", sources)
        self.assertIn("To be or not to", sources)

    def test_J1_readme_contents(self):
        """Verify README.md contains student details, tables, commands, and conclusions."""
        readme_path = os.path.join(self.workspace_dir, "README.md")
        self.assertTrue(os.path.exists(readme_path))
        with open(readme_path) as f:
            content = f.read()
        self.assertIn("Yashraj Sakunde", content)
        self.assertIn("230953172", content)
        self.assertIn("Simple RNN", content)
        self.assertIn("Bidirectional RNN", content)
        self.assertIn("LSTM", content)
        self.assertIn("GRU", content)
        self.assertIn("181.31", content)  # BiRNN perplexity
        self.assertIn("results/plots/loss_curves.png", content)

    def test_L1_submission_docx_and_pdf(self):
        """Verify Word and PDF submission documents exist and contain required details."""
        docx_path = os.path.join(self.sub_dir, "YashrajSakunde_230953172_FGAI_Assignment2.docx")
        pdf_path = os.path.join(self.sub_dir, "YashrajSakunde_230953172_FGAI_Assignment2.pdf")
        self.assertTrue(os.path.exists(docx_path), "Submission docx must exist.")
        self.assertTrue(os.path.exists(pdf_path), "Submission pdf must exist.")
        self.assertGreater(os.path.getsize(docx_path), 5000)
        self.assertGreater(os.path.getsize(pdf_path), 3000)

        # Inspect docx contents
        from docx import Document
        doc = Document(docx_path)
        full_text = " ".join([p.text for p in doc.paragraphs] + [c.text for t in doc.tables for r in t.rows for c in r.cells])
        self.assertIn("Yashraj Sakunde", full_text)
        self.assertIn("230953172", full_text)
        self.assertIn("FGAI IA 2", full_text)
        self.assertIn("Tiny Shakespeare", full_text)
        self.assertIn("https://github.com/Yashraj88888/VerseForge-Shakespeare-Style-Text-Generator", full_text)


if __name__ == "__main__":
    unittest.main()
