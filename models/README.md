# VerseForge Model Checkpoints

This directory stores trained TensorFlow/Keras neural network models and training histories:

- `Simple_RNN.keras` & `Simple_RNN_history.json`
- `Bidirectional_RNN.keras` & `Bidirectional_RNN_history.json`
- `LSTM.keras` & `LSTM_history.json`
- `GRU.keras` & `GRU_history.json`

## How to Regenerate Model Checkpoints:
To train and generate all four model checkpoints from scratch, run from the project root:
```bash
python3 main.py --mode train
```
or:
```bash
python3 src/train.py --epochs 8 --batch-size 64 --step 4
```
Training takes approximately 4–5 minutes on modern CPU/GPU hardware.
