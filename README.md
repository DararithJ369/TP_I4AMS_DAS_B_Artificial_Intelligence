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
└── tpw2/
    ├── TP2_DeepNN_Step_by_Step_Class_activity.ipynb  # ★ Week 2 Class Activity (Exercises 1–3 Completed)
    └── TP2_Odd_or_Even_Assignment_DeepNetwork.ipynb  # ★ Week 2 Assignment (Tasks 1–6 Completed)
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
