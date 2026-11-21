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
| **TP 04** | Optimization Algorithms & Techniques | SGD, Mini-batches, Momentum, Nesterov, Adaptive Optimizers (Adagrad, RMSprop, Adam, AdamW), LR Schedules, Batch Normalization, CIFAR-10 | Completed | [View Notebook](tpw4/TP4_Optimization.ipynb) |

---

## Repository Structure & Solutions

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── tpw1/
│   ├── TP_Introduction_to_PyTorch.ipynb          # Week 1 Primary Solution Notebook (Exercises 1–12 Completed)
│   └── Notebook1_Introduction_to_PyTorch.ipynb   # Guided tutorial / lecture notebook (with completed exercises)
├── tpw2/
│   ├── TP2_DeepNN_Step_by_Step_Class_activity.ipynb  # Week 2 Class Activity (Exercises 1–3 Completed)
│   └── TP2_Odd_or_Even_Assignment_DeepNetwork.ipynb  # Week 2 Assignment (Tasks 1–6 Completed)
├── tpw3/
│   └── TP3_Initialization_Regularization.ipynb       # Week 3 Assignment & Class Activity (Exercises 1–8 Completed)
└── tpw4/
    └── TP4_Optimization.ipynb                        # Week 4 Primary Solution Notebook (Exercises 1–9 Completed)
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

Every exercise in this notebook includes a comprehensive **Conceptual Guide & Rationale** addressing:
1. **What**: The exact assignment requirements and mathematical targets.
2. **Why**: The underlying theoretical motivation, gradient flow behavior, and numerical constraints.
3. **How**: Step-by-step PyTorch implementations, tensor mechanics, and training cycles.
4. **Visualization Rationale**: Justifications for chosen charts (scatter distributions, dual-function curve overlays, and multi-panel decision boundary contours).

- **Exercise 1 · The Moons Dataset**:
  - **What & Why:** Generated non-linearly separable benchmark data using scikit-learn's `make_moons` ($1000$ samples, `noise=0.07`, `random_state=42`) to demonstrate why linear models fail and justify deep architectures.
  - **How:** Converted into PyTorch `float32` tensors with labels shaped $(m, 1)$ via `.unsqueeze(1)` to prevent silent $(m, m)$ broadcasting bugs in BCE loss; split 80/20 into train ($800$) and test ($200$) sets.
  - **Chart Choice:** 2D scatter plot using `RdYlBu` colormap (Class 0 / Upper Moon = Red, Class 1 / Lower Moon = Blue) to visually display non-linear curvature.
- **Exercise 2 · MoonModel with He Initialization**:
  - **What & Why:** Built `MoonModel` subclassing `nn.Module` with non-linear activations (ReLU) outputting one logit. Symmetrical zero weights collapse hidden layers into a single neuron; He initialization ($\sigma = \sqrt{2/n_{in}}$) compensates for ReLU's half-zeroing effect, preserving variance.
  - **How:** Architecture `[2, 30, 30, 1]` ($1051$ parameters). `init_he(model)` applies `kaiming_normal_` on weights and zeros on biases (safe since random weights break symmetry).
- **Exercise 3 · Loss and Optimizer**:
  - **What & Why:** Configured `nn.BCEWithLogitsLoss` and `torch.optim.SGD` ($\text{lr} = 0.3$). Decoupling Sigmoid and BCELoss causes `log(0) = -inf`/`NaN` on extreme logits ($z=100$) and vanishing gradients.
  - **How:** `nn.BCEWithLogitsLoss` fuses Sigmoid + BCE via the log-sum-exp trick ($\mathcal{L} = \max(z,0) - zy + \log(1+e^{-|z|})$), with gradient simplifying cleanly to $\sigma(z) - y$.
- **Exercise 4 · Training and Testing Loop**:
  - **What & Why:** Built a 5-step PyTorch training cycle (`model.train()`, forward, loss, `zero_grad()`, `backward()`, `step()`) and an evaluation block (`model.eval()`, `torch.inference_mode()`) to assess generalization on unseen data.
  - **How:** Trained for $1000$ epochs, tracking metrics every $100$ epochs. Reached **$100.0\%$ test accuracy by Epoch 200** (exceeding the $96.0\%$ benchmark).
- **Exercise 5 · Decision Boundaries**:
  - **What & Why:** Visualized spatial classification boundaries to confirm the network learned smooth continuous geometry rather than memorizing discrete points.
  - **How & Chart Choice:** Side-by-side dense meshgrid contour plots ($300 \times 300 = 90,000$ points) evaluating train ($100.0\%$) and test ($100.0\%$) distributions, confirming identical boundary separation and strong generalization.
- **Exercise 6 · Pure PyTorch Tanh**:
  - **What & Why:** Implemented $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ from scratch to study zero-centered activations and floating-point overflow dynamics.
  - **How & Chart Choice:** On $[-5, 5]$, custom Tanh matches `torch.tanh` with max diff $\le 1.2 \times 10^{-7}$. Overlaid dual-line plot displays steep gradient at $z=0$ and saturation at $|z| > 2$. For $z = 100$, $e^{100} \to \text{inf}$ yields `NaN`; the reformulation $\tanh(z) = 2\sigma(2z) - 1$ safely evaluates to $1.0$.
- **Exercise 7 · CS231n 3-Class Spiral Dataset**:
  - **What & Why:** Multi-class classification ($K=3$, $300$ points) with Softmax cross-entropy on intertwined concentric spiral arms.
  - **How & Chart Choice:** Architecture `[2, 64, 64, 3]` trained with `nn.CrossEntropyLoss` and `Adam` ($\text{lr} = 0.02$) to **$100.0\%$ test accuracy**. Multi-class contour plot confirms the network learned three rotating spiral decision boundaries.
- **Exercise 8 · Regularization on Large Spiral Model**:
  - **What & Why:** Compared high-capacity model (`[2, 128, 128, 3]`) across Baseline (overfitted), L2 Regularization (`weight_decay=1e-3` on weights only), and Dropout ($p=0.25$).
  - **Chart Choice & Analysis:** 3-panel side-by-side decision boundary figure visually demonstrates that L2 regularization smooths out high-frequency boundary wrinkles into continuous spiral arms, while Dropout builds wide-margin decision boundaries. Confirmed hyperparameters must be tuned on a validation/dev set to prevent test data leakage.

#### Experimental Benchmarks & Results (TP3)

##### 1. Weight Initialization Benchmark (Circles Dataset)
| Initialization Method | Mathematical Formulation | Train Accuracy | Test Accuracy | Observations & Failure Modes |
| :--- | :--- | :---: | :---: | :--- |
| **Zeros** | $W = 0, \; b = 0$ | $50.0\%$ | $50.0\%$ | Symmetrical neurons: all hidden units compute identical gradients |
| **Large Random** | $W \sim \mathcal{N}(0, 10^2)$ | $79.0\%$ | $76.0\%$ | Saturated activations cause exploding activations and slow convergence |
| **He (Kaiming)** | $W \sim \mathcal{N}(0, 2 / n^{[l-1]})$ | **$98.7\%$** | **$93.0\%$** | Preserves activation variance across ReLU layers; optimal convergence |
| **PyTorch Default** | Uniform Kaiming Initialization | $97.3\%$ | $92.0\%$ | Standard framework baseline with effective gradient flow |

##### 2. Regularization Benchmark (3-Class CS231n Spirals Dataset)
| Model Architecture | Regularization Technique | Train Accuracy | Test Accuracy | Decision Boundary Geometry |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline (Overfit)** | None (`drop_p = 0.0`, `wd = 0.0`) | $99.2\%$ | $100.0\%$ | Jagged, highly fragmented contours wrapping tightly around individual points |
| **L2 Regularization** | `weight_decay = 1e-3` (weights only) | $99.2\%$ | $100.0\%$ | Constrained parameter norms produce smooth, rounded spiral separation |
| **Dropout** | Inverted Dropout ($p = 0.25$) | $99.2\%$ | $100.0\%$ | Prevents co-adaptation; robust, wide-margin decision boundaries |

---

### Week 4 (`tpw4/`) — Optimization: SGD, Mini-batches, Momentum & Modern Techniques
The complete solutions and experimental benchmarks for Week 4 are located in [**`tpw4/TP4_Optimization.ipynb`**](tpw4/TP4_Optimization.ipynb):

#### Core Exercises & Theoretical Synthesis

- **Exercise 1 · The Learning Rate of SGD**:
  - **What & Why:** Evaluated plain `optim.SGD` across six learning rates $\alpha \in \{1.0, 0.3, 0.1, 0.03, 0.01, 0.001\}$ on 10,000 MNIST training samples for 5 epochs. Step size $\alpha$ governs convergence rate vs numerical stability in $\theta \leftarrow \theta - \alpha \nabla_\theta J$.
  - **How & Chart Choice:** Plotted exponentially smoothed step loss ($\beta=0.98$) and validation accuracy per epoch. $\alpha = 0.3$ proved optimal ($95.05\%$ accuracy); $\alpha = 1.0$ caused severe oscillations ($33.57\%$), and $\alpha = 0.001$ stagnated prematurely ($53.40\%$).

- **Exercise 2 · Batch Size: One Example, Mini-batches, and Full Set**:
  - **What & Why:** Benchmarked batch sizes $B \in \{1, 32, 256, 10000\}$ under $\alpha=0.01$ and $\alpha=0.1$ to quantify the trade-off between gradient noise and hardware SIMD vectorization throughput.
  - **How & Findings:** 
    - At $\alpha=0.01$, stochastic GD ($B=1$, 10,000 updates/epoch) reached $93.57\%$ but took $2.59$ s/epoch due to Python loop overhead; mini-batch ($B=32$) completed in $0.16$ s/epoch ($16\times$ faster). Full-batch ($B=10000$) took only 1 update/epoch and failed to move ($9.80\%$).
    - At $\alpha=0.1$, stochastic GD ($B=1$) suffered violent noise divergence ($11.30\%$), whereas mini-batch ($B=32$) achieved the sweet spot ($93.60\%$).

- **Exercise 3 · Momentum & Nesterov Acceleration**:
  - **What & Why:** Evaluated classical Polyak momentum ($\beta=0.9$), Nesterov accelerated gradient, small learning rate ($\alpha=0.001, \beta=0.9$), and excessive momentum ($\beta=0.99$) against plain SGD ($\alpha=0.01$).
  - **How & Findings:** Momentum $\beta=0.9$ boosted validation accuracy from $88.40\%$ to $93.98\%$ by multiplying the effective step size by $\frac{1}{1-\beta} = 10\times$ along consistent gradients and canceling ravines. Nesterov momentum achieved $93.93\%$ with predictive look-ahead damping. $\beta=0.99$ created excessive inertia and overshooting ($92.03\%$).

- **Exercise 4 · Optimizers and Their Learning Rates**:
  - **What & Why:** Comprehensive matrix comparison of 6 optimizers (SGD, SGD+Mom, Adagrad, RMSprop, Adam, AdamW) across 4 decade learning rates $\alpha \in \{10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}\}$.
  - **Grid Benchmark:**
    - SGD: Best at $10^{-1}$ ($93.62\%$).
    - SGD + Momentum: Broad optimum at $10^{-2}$ ($94.00\%$) and $10^{-1}$ ($93.70\%$).
    - Adagrad: Best at $10^{-2}$ ($93.98\%$); monotonic denominator accumulation stalls smaller LRs.
    - RMSprop: Peak performance at $10^{-3}$ ($94.93\%$); diverged/collapsed at $10^{-1}$ ($21.25\%$).
    - Adam & AdamW: Outstanding convergence and stability at $10^{-3}$ ($94.15\%$ and $94.02\%$).

- **Exercise 5 · Learning Rate Schedules**:
  - **What & Why:** Trained for 10 epochs using `optim.SGD(lr=0.05, momentum=0.9)` comparing: Constant LR, StepLR (step=3, $\gamma=0.3$), ExponentialLR ($\gamma=0.7$), CosineAnnealingLR ($T_{\max}=10$), and Early StepLR (step=1, $\gamma=0.1$).
  - **Chart Choice & Findings:** Log-scale learning rate profiles plotted alongside loss/accuracy curves. Cosine Annealing achieved the highest validation accuracy ($95.95\%$) by decaying smoothly into deep local valleys without discontinuous shocks. Decaying too early froze the model prematurely at $93.87\%$.

- **Exercise 6 · Batch Normalization**:
  - **What & Why:** Evaluated `MLP` with vs without `nn.BatchNorm1d` across $\alpha \in \{1.0, 0.1, 0.01, 0.001\}$, and analyzed `model.eval()` vs `model.train()` modes on mini-batches of 8 samples.
  - **How & Findings:**
    - Without BN, $\alpha = 1.0$ diverged completely (loss exploded to `inf`). With BN, the model stabilized and reached $90.40\%$.
    - Across all learning rates, BN converged faster and reached higher validation accuracy ($95.43\%$ vs $93.82\%$).
    - **Eval Mode vs Train Mode:** Evaluating the BN model in `model.eval()` mode (clean population statistics) yielded **$95.77\%$**, whereas evaluating in `model.train()` mode with small batches ($B=8$) dropped to **$85.85\%$** (a **$9.92$ percentage point collapse** due to high sample-variance distortion).

- **Exercise 7 · Best Model on Full 54,000 Training Set**:
  - **Configuration:** Architecture with Batch Normalization (`batch_norm=True`), `AdamW` optimizer ($\alpha = 10^{-3}$, `weight_decay = 1e-4`), batch size 64, and `CosineAnnealingLR` ($T_{\max}=10, \eta_{\min}=10^{-5}$) trained on all 54,000 images for 10 epochs.
  - **Result:** Monotonic training loss drop to $0.0070$ and peak validation accuracy of **$98.25\%$**.

- **Exercise 8 · Unbiased Test Set Generalization**:
  - **What & Why:** Evaluated the best trained model strictly once on the 10,000 unseen MNIST test images.
  - **Result:**
    - **Final Test Accuracy:** **$98.33\%$** (Cross-Entropy Loss: $0.0572$).
    - **Generalization Gap:** $|98.25\% - 98.33\%| = 0.08\%$, demonstrating optimal generalization without overfitting.

- **Exercise 9 · CIFAR-10 Optimization Benchmark**:
  - **What & Why:** Extended evaluation from MNIST to CIFAR-10 (50,000 $32 \times 32 \times 3$ natural color images, 10 classes) by partitioning into 45,000 train / 5,000 val sets, normalizing channel-wise with training statistics, flattening to 3072 features, and training an expanded MLP `[3072, 512, 256, 10]` (~1.7M parameters) for 10 epochs.
  - **Configurations & Findings:**
    - **Configuration (a) — AdamW + Batch Normalization + Cosine Annealing:** Reached peak validation accuracy of **$58.34\%$** and rapid loss reduction to $0.6672$.
    - **Configuration (b) — AdamW without Batch Normalization:** Reached **$55.98\%$** validation accuracy ($2.36$ percentage points lower), showing that Batch Normalization stabilizes training across wider hidden layers.
    - **Configuration (c) — Plain SGD ($\alpha=0.01$):** Stagnated at **$50.68\%$** peak validation accuracy with higher loss ($1.1398$), demonstrating that modern adaptive optimization and normalization become crucial on rugged, high-dimensional loss landscapes.

---
