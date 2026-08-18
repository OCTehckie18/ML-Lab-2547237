# Lab 10 — Learning the XOR Boolean Function Using an MLP

**Registration Number:** 2547237  
**Course:** MCA 521-4 — Machine Learning  
**Lab:** Lab Exercise 10  
**Branch:** `Lab-10`

---

## Aim

1. To understand how to implement neural networks using different deep learning libraries (**Keras** and **TensorFlow**).
2. To solve the non-linear XOR problem using an MLP and study the effect of hyperparameters such as learning rate, activation functions, number of neurons, and epochs on model performance.

---

## The XOR Problem

The XOR Boolean function is a classic example of a **non-linearly separable** problem. A single-layer perceptron cannot learn it; an MLP with at least one hidden layer is required.

| Input 1 | Input 2 | XOR Output |
|---------|---------|------------|
| 0       | 0       | 0          |
| 0       | 1       | 1          |
| 1       | 0       | 1          |
| 1       | 1       | 0          |

---

## Project Structure

```
Lab-10/
└── 2547237_Lab10_XOR_MLP.ipynb   # Main notebook (all implementations)
```

---

## Notebook Contents

### Step 1 — Dataset
Defines all 4 XOR input–output combinations as NumPy arrays.

### Step 2 — Keras (TensorFlow High-Level API)
- **Architecture:** Input(2) → Dense(4, tanh) → Dense(1, sigmoid)
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam (lr = 0.1)
- **Epochs:** 1000
- Uses `keras.Sequential` and `model.fit()`.

### Step 3 — TensorFlow Low-Level API
- Same architecture, implemented using `tf.Variable`, `tf.GradientTape`, and a manual training loop.
- Full explicit control over forward pass, loss, gradient computation, and weight updates.

### Optional Exercises
- **Decision boundary plots** — visualises the non-linear boundary learned by each model.
- **Training curve comparison** — loss vs. epochs for both implementations.
- **Hyperparameter study:**
  - Effect of learning rate (0.001, 0.01, 0.1, 0.5)
  - Effect of activation function (tanh, relu, sigmoid, elu)
  - Effect of number of hidden neurons (2, 4, 8, 16)

---

## Key Findings

| Hyperparameter | Observation |
|---|---|
| **Hidden layer** | Required — XOR is not linearly separable |
| **Activation** | `tanh` > `relu` > `sigmoid` for XOR convergence |
| **Learning rate** | ~0.1 with Adam is optimal; too low → slow, too high → oscillates |
| **Neurons** | ≥ 2 hidden neurons suffice; more = faster convergence |
| **Epochs** | 500–1000 sufficient at lr=0.1 |

---

## Prerequisites

```bash
pip install tensorflow numpy matplotlib
```

## Running the Notebook

```bash
cd Lab-10
jupyter notebook 2547237_Lab10_XOR_MLP.ipynb
```

Execute all cells sequentially. Plots are auto-saved as PNG files in the same directory.

---

**Registration Number:** 2547237
