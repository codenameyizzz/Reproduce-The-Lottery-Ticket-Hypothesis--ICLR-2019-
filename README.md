---

# Reproducing the Lottery Ticket Hypothesis (ICLR 2019)

This repository contains a reproducibility study of the paper:

**"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"**
Jonathan Frankle, Michael Carbin — ICLR 2019
[Paper Link](https://arxiv.org/abs/1803.03635)

## Objectives

To reproduce the key findings of the Lottery Ticket Hypothesis:

> There exist sub-networks within a randomly-initialized dense neural network that, when trained in isolation from their original initialization, can match or exceed the accuracy of the original network.

This reproduction focuses on:

- Training a baseline LeNet model on MNIST.
- Applying **one-shot magnitude-based pruning** (20% of smallest weights).
- Resetting surviving weights to their initial values.
- Retraining the pruned model and comparing accuracy against the original.
- Visualizing accuracy curves before and after pruning.

## Highlights

- 5 Trials × 5 Iterative Magnitude Pruning Rounds
- LeNet on MNIST Dataset
- Reset to Initial Weights after Pruning
- Early Stopping on Validation Accuracy
- AMP (Mixed Precision) Support for GPUObjective

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

- Results saved as: `results.npy`
- Plots saved as:
  - `figure_accuracy_vs_iteration.png`
  - `figure_accuracy_vs_sparsity.png`

---

## License

MIT License
