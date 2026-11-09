# Artificial Intelligence — Practical Work (TP) & Class Activities

Repository: **TP_I4AMS_DAS_B_Artificial_Intelligence**  
Academic Year: Year 4 — Institute of Technology of Cambodia (ITC)  
Department: Applied Mathematics and Statistics (I4-AMS-DAS-B)   
Course: **Artificial Intelligence**    
TP Lecturer: **VANN Visal**      
Student Name: **KIM Putdararith**       
Student ID: **e20230638**

---

## Overview

This repository is dedicated to coursework, class activities, practical work (TP / *Travaux Pratiques*), lab sessions, and assignments for the Artificial Intelligence course.

Contents include:
- Weekly practical exercises and lab notebooks
- Hands-on machine learning and deep learning implementations
- In-class activities and coursework submissions

---

## Lab & Assignment Directory

| Lab / Assignment | Topic | Key Techniques & Concepts | Status | Link |
| :--- | :--- | :--- | :---: | :---: |
| **TP 01** | Introduction to PyTorch | Tensors, Gradients, Broadcasting, Autograd | Completed | [View Notebook](tpw1/TP_Introduction_to_PyTorch.ipynb) |
| **Notebook 01** | PyTorch Tutorial & Fundamentals | Tensor operations, GPU acceleration, Random seeds, Vectorization | Completed | [View Notebook](tpw1/Notebook1_Introduction_to_PyTorch.ipynb) |
| **TP 02 Class Activity** | Deep Neural Network Step by Step | Deep L-layer NN, Forward/Backward propagation, Leaky ReLU, Autograd verification, Gradient descent | Completed | [View Notebook](tpw2/TP2_DeepNN_Step_by_Step_Class_activity.ipynb) |
| **TP 02 Assignment** | Odd or Even Classification with Deep NN | Binary classification on MNIST, Deep architecture tuning, Initialization analysis, Error analysis | Completed | [View Notebook](tpw2/TP2_Odd_or_Even_Assignment_DeepNetwork.ipynb) |
| **TP 03** | Initialization and Regularization | Weight initialization (zeros, random, He), L2 regularization, Dropout, Moons & Spirals classification, Custom Tanh | Completed | [View Notebook](tpw3/TP3_Initialization_Regularization.ipynb) |

---

## Repository Structure & Solutions

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── tpw1/
│   ├── TP_Introduction_to_PyTorch.ipynb          # ★ Week 1 Primary Solution Notebook (Exercises 1–12 Completed)
│   └── Notebook1_Introduction_to_PyTorch.ipynb   # Guided tutorial / lecture notebook (with completed exercises)
├── tpw2/
│   ├── TP2_DeepNN_Step_by_Step_Class_activity.ipynb  # ★ Week 2 Class Activity (Exercises 1–3 Completed)
│   └── TP2_Odd_or_Even_Assignment_DeepNetwork.ipynb  # ★ Week 2 Assignment (Tasks 1–6 Completed)
└── tpw3/
    └── TP3_Initialization_Regularization.ipynb       # ★ Week 3 Assignment & Class Activity (Exercises 1–8 Completed)
```

---

## Completed Solutions

### Week 1 (`tpw1/`) — Introduction to PyTorch
The complete solutions for Week 1 exercises are located in [**`tpw1/TP_Introduction_to_PyTorch.ipynb`**](tpw1/TP_Introduction_to_PyTorch.ipynb) (and [**`tpw1/Notebook1_Introduction_to_PyTorch.ipynb`**](tpw1/Notebook1_Introduction_to_PyTorch.ipynb)):

#### Part A · Warm-up
- **Exercise 1**: Random tensor creation `(7, 7)` and matrix multiplication with transposed `(1, 7)`.
- **Exercise 2**: Reproducibility with `torch.manual_seed(0)`.
- **Exercise 3**: GPU seed configuration via `torch.cuda.manual_seed(1234)` with documentation explanation.
- **Exercise 4**: Creating tensors on CPU, moving to `device`, and matrix multiplication.
- **Exercise 5**: Aggregation operations (`max`, `min`, `argmax`, `argmin`).
- **Exercise 6**: Removing size-1 dimensions with `squeeze()` (`(1, 1, 1, 10)` → `(10,)`).
- **Exercise 7**: Converting NumPy arrays to PyTorch `float32` tensors and computing mean height.

#### Part B · Applied Exercises
- **Exercise 8 · From images to a data matrix**: Converting `(N, H, W, C)` to PyTorch's `(N, C, H, W)` using `permute`, flattening to `(8, 3072)`, and pixel verification.
- **Exercise 9 · Predict broadcast shapes**: Predicting broadcasting outcomes for tensor pairs and verifying with `np.broadcast_shapes`.
- **Exercise 10 · Standardize features**: Feature column standardization without loops using `mean` and `std` with `keepdim=True`.
- **Exercise 11 · Find the bug**: Diagnosing and fixing silent broadcasting dimensional mismatch in Mean Squared Error (`mse`).
- **Exercise 12 · Replace the loops**: Vectorizing a linear model ($z = Xw + b$) and min-max scaling, verifying numerical equality with `torch.allclose`, and benchmarking performance gains.

---

### Week 2 (`tpw2/`) — Deep Neural Networks

#### 1. Class Activity: Building a Deep Neural Network, Step by Step
Located in [**`tpw2/TP2_DeepNN_Step_by_Step_Class_activity.ipynb`**](tpw2/TP2_DeepNN_Step_by_Step_Class_activity.ipynb):

- **Exercise 1 · Shapes & Number of Parameters**:
  - Computed layer-by-layer shapes for $\mathbf{W}^{[l]}$, $\mathbf{b}^{[l]}$, and $\mathbf{Z}^{[l]}$ analytically for $[784, 64, 32, 1]$ with batch size $m = 10000$.
  - Calculated total parameters (52,353) and verified against `initialize_parameters_deep`.
  - Compared model capacity against $[784, 20, 7, 5, 1]$ (15,893 parameters; 3.3× ratio).
- **Exercise 2 · Leaky ReLU Activation**:
  - Implemented `leaky_relu` and `leaky_relu_backward` with slope $0.01$.
  - Implemented modular `linear_activation_forward_leaky`, `linear_activation_backward_leaky`, `L_model_forward_leaky`, and `L_model_backward_leaky`.
  - Verified numerical gradient correctness against PyTorch autograd via `compare_with_autograd` ($\Delta \le 3.0 \times 10^{-8}$).
- **Exercise 3 · Debugging Backward Propagation**:
  - Diagnosed Bug 1: Column-convention dimension indexing `m = AL.shape[1]` corrected to row convention `m = AL.shape[0]`.
  - Diagnosed Bug 2: Missing factor in output sigmoid gradient corrected to $d\mathbf{Z}^{[L]} = \mathbf{A}^{[L]} - \mathbf{Y}$ (or $d\mathbf{A} \odot \sigma(Z)(1 - \sigma(Z))$).
  - Implemented `L_model_backward_fixed` and validated gradient equality against autograd ($\Delta \approx 10^{-8}$).
  - Documented concise single-sentence explanations for each bug.

#### 2. Assignment: Odd or Even Classification with Deep Networks
Located in [**`tpw2/TP2_Odd_or_Even_Assignment_DeepNetwork.ipynb`**](tpw2/TP2_Odd_or_Even_Assignment_DeepNetwork.ipynb):

- **Task 1 · `predict` and `accuracy`**:
  - Implemented probability thresholding ($a^{[L]} > 0.5 \to 1.0$) and accuracy calculation returning Python floats.
  - Verified baseline random guessing on untrained network ($57.6\%$ dev accuracy, $44.9\%$ predicted odd).
- **Task 2 · The Training Loop**:
  - Implemented generalized `L_layer_model` handling forward pass, cross-entropy cost tracking, backpropagation, and parameter updates.
- **Task 3 · Architectural Depth Comparison**:
  - Trained and evaluated Logistic Regression (`[784, 1]`: $89.3\%$ dev), 2-layer (`[784, 7, 1]`: $95.5\%$ dev), and 4-layer (`[784, 20, 7, 5, 1]`: $96.9\%$ dev).
  - Plotted cost curves and documented bias vs. variance analysis and gradient spike dynamics.
- **Task 4 · Weight Initialization Analysis**:
  - Evaluated small initialization (`scale=0.01`) on a 4-layer network, analyzing signal decay and gradient vanishing ($51.2\%$ accuracy) vs. variance-preserving Xavier/He initialization.
- **Task 5 · Best Model Selection**:
  - Tested 4 distinct configurations across width, depth, and learning rates.
  - Selected `[784, 128, 32, 1]` ($\text{lr}=0.25$, $2500$ iterations) based on dev set performance (**$97.85\%$ dev accuracy**), achieving **$97.58\%$ on the test set**.
- **Task 6 · Error Analysis**:
  - Evaluated per-digit test error rates (highest: 5, 8, 9; lowest: 0, 6).
  - Visualized misclassified samples with predicted probabilities.
  - Analyzed perceptual confusion mechanisms across digit pairs (4 vs. 9, 3 vs. 8).

---

### Week 3 (`tpw3/`) — Initialization and Regularization
Located in [**`tpw3/TP3_Initialization_Regularization.ipynb`**](tpw3/TP3_Initialization_Regularization.ipynb):

- **Exercise 1 · The Moons Dataset**:
  - Generated binary classification dataset using scikit-learn's `make_moons` ($1000$ samples, `noise=0.07`, `random_state=42`).
  - Converted into PyTorch `float32` tensors with labels shaped $(m, 1)$.
  - Split 80/20 into train ($800$ samples) and test ($200$ samples) sets and plotted the training distribution.
- **Exercise 2 · MoonModel with He Initialization**:
  - Implemented `MoonModel` subclassing `nn.Module` with non-linear activations (ReLU) outputting one logit.
  - Initialized linear layers with He initialization (`kaiming_normal_` for weights, zeros for biases).
  - Model architecture: `[2, 30, 30, 1]` ($1051$ trainable parameters).
- **Exercise 3 · Loss and Optimizer**:
  - Configured `nn.BCEWithLogitsLoss` and `torch.optim.SGD` optimizer ($\text{lr} = 0.3$).
  - Documented numerical stability advantages of `nn.BCEWithLogitsLoss` (fused log-sum-exp stabilization preventing `NaN` from overflow/underflow) over `nn.Sigmoid()` + `nn.BCELoss()`.
- **Exercise 4 · Training and Testing Loop**:
  - Trained `MoonModel` for $1000$ epochs, tracking training/test loss and accuracy every $100$ epochs.
  - Achieved **$100.0\%$ test accuracy** (exceeding the $96.0\%$ threshold).
- **Exercise 5 · Decision Boundaries**:
  - Visualized side-by-side decision boundaries for training ($100.0\%$) and test ($100.0\%$) sets using `plot_decision_boundary`.
- **Exercise 6 · Pure PyTorch Tanh**:
  - Implemented `my_tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))` using pure PyTorch exponentials and arithmetic.
  - Verified numerical equivalence against `torch.tanh` on $[-5, 5]$ ($\Delta \le 1.2 \times 10^{-7}$) and plotted comparison.
  - Analyzed overflow behavior at $z = 100$ ($e^{100} \to \text{inf} \implies \text{NaN}$) and implemented numerically stable reformulation $\tanh(z) = 2\sigma(2z) - 1$ ($= 1.0$).
- **Exercise 7 · CS231n 3-Class Spiral Dataset**:
  - Created 3-class spiral dataset ($300$ points) and converted to PyTorch tensors (`float32` features, `long` labels).
  - Built `SpiralModel` multi-class network (`[2, 64, 64, 3]`), trained with `nn.CrossEntropyLoss` and `Adam` optimizer ($\text{lr} = 0.02$).
  - Reached **$100.0\%$ test accuracy** (exceeding the $95.0\%$ target) and plotted train/test decision boundaries.
- **Exercise 8 · Regularization on Large Spiral Model**:
  - Built a large capacity model (`[2, 128, 128, 3]`) to study overfitting and regularization dynamics.
  - Trained and compared three configurations:
    1. **Baseline** (No regularization)
    2. **L2 Regularization** (`weight_decay=1e-3` on weights only)
    3. **Dropout** (`nn.Dropout(p=0.25)`)
  - Plotted side-by-side decision boundaries illustrating overfitting contortions in the baseline vs. smooth, generalized decision boundaries under L2 and Dropout.
  - Documented hyperparameter tuning protocol (using a validation/dev set, never the test set).

---
