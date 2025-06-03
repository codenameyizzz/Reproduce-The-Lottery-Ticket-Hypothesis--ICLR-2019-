# Reproducing the Lottery Ticket Hypothesis (ICLR 2019)

This repository contains a reproducibility study of the paper:

**"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"**
Jonathan Frankle, Michael Carbin — ICLR 2019
[Paper](https://arxiv.org/abs/1803.03635)

---

## Highlights

- 5 Trials × 5 Iterative Magnitude Pruning Rounds
- LeNet on MNIST Dataset
- Reset to Initial Weights after Pruning
- Early Stopping on Validation Accuracy
- AMP (Mixed Precision) Support for GPU

---

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
