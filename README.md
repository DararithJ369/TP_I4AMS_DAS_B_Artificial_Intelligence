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

---

## Repository Structure & Solutions

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
└── tpw1/
    ├── TP_Introduction_to_PyTorch.ipynb          # ★ Primary Solution Notebook (All Exercises 1–12 Completed)
    └── Notebook1_Introduction_to_PyTorch.ipynb   # Guided tutorial / lecture notebook (with completed exercises)
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

