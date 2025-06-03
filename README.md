# Reproducing the Lottery Ticket Hypothesis (ICLR 2019)

This repository contains a reproducibility study of the paper:

**"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"**
Jonathan Frankle, Michael Carbin — ICLR 2019
[Paper Link](https://arxiv.org/abs/1803.03635)

## Objective

To reproduce the key findings of the Lottery Ticket Hypothesis:

> There exist sub-networks within a randomly-initialized dense neural network that, when trained in isolation from their original initialization, can match or exceed the accuracy of the original network.

This reproduction focuses on:

- Training a baseline LeNet model on MNIST.
- Applying **one-shot magnitude-based pruning** (20% of smallest weights).
- Resetting surviving weights to their initial values.
- Retraining the pruned model and comparing accuracy against the original.
- Visualizing accuracy curves before and after pruning.

---

## Highlights

- 5 Trials × 5 Iterative Magnitude Pruning Rounds
- LeNet on MNIST Dataset
- Reset to Initial Weights after Pruning
- Early Stopping on Validation Accuracy
- AMP (Mixed Precision) Support for GPU
- Figure 3 & 4-style Plots:
  - Accuracy vs Iteration
  - Accuracy vs Sparsity (Weights Remaining)

---

## Optional: Full Iterative Reproduction Mode

For users who want to **fully match the original paper’s iterative pruning results and plots**, a complete iterative pruning function is also provided (but commented out by default).

- This mode:
  - Prunes 20% of smallest-magnitude weights **iteratively**
  - Resets weights and retrains after each pruning round
  - Repeats for **5 iterations × 5 trials** with **early stopping**
  - Uses up to **50 epochs per iteration**
- Can be enabled by uncommenting the `run_experiment()` block.

---

## Experimental Setup

- **Dataset**: MNIST (handwritten digits)
- **Model**: LeNet (simple CNN)
- **Optimizer**: Adam, learning rate = 0.001
- **One-Shot Pruning**: 20% smallest-magnitude weights
- **Epochs**: 20
- **Validation**: 10% of training set
- **Optional Iterative Mode**:
  - Epochs up to 50 with early stopping
  - 5 iterations over 5 trials

---

## Results

With one-shot pruning, the retrained sparse subnetwork achieves accuracy close to the dense baseline model.
Optional iterative pruning (disabled by default) replicates the full pruning-retrain-reset loop described in the original paper and can reproduce their result plots more closely.

---

## How to Run

```bash
python main.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## License

MIT License
