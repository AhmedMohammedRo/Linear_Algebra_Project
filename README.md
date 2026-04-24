# 📉 Linear Regression from Scratch
### Linear Algebra for Computer Science — Project 23

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Validation-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> **Bridging pure mathematics and machine learning** — implementing Linear Regression from first principles using nothing but NumPy and the mathematics of Linear Algebra.

---

## 🎯 Project Overview

This project implements **Linear Regression from scratch** using core Linear Algebra concepts, as part of *Linear Algebra for Computer Science (Math 204)* at the Faculty of Computer & Information Sciences.

Rather than using black-box ML libraries, every algorithm is derived and implemented mathematically — then validated against industry-standard tools to prove correctness.

**Dataset:** Canadian Vehicle CO₂ Emissions (~7,385 vehicles)  
**Goal:** Predict CO₂ emissions (g/km) from engine characteristics  
**Result:** R² = 0.7345, predictions within ~30 g/km of actual values

---

## 🧠 The Mathematics

### Normal Equation — Closed Form Solution

The exact solution to Linear Regression is derived by minimizing the least squares cost:

$$\mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

- **One-shot** — solves directly with no iterations
- **Exact** — gives the globally optimal solution
- **Limitation** — matrix inversion scales as O(n³), expensive for large datasets

### Gradient Descent — Iterative Optimization

Instead of inverting a matrix, we follow the gradient of the cost function downhill:

$$J(\mathbf{w}) = \frac{1}{2m} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

$$\mathbf{w} := \mathbf{w} - \alpha \cdot \frac{1}{m} \mathbf{X}^\top (\mathbf{X}\mathbf{w} - \mathbf{y})$$

- **Iterative** — converges over 1000 steps
- **Scalable** — works efficiently on massive datasets
- **Key insight** — arrives at the same weights as the Normal Equation

### Ridge Regression — Regularization

Prevents overfitting by adding a penalty term to the cost function:

$$\mathbf{w} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

The identity matrix **I** is modified so the bias term is never penalized (I₀₀ = 0), preserving the intercept while shrinking the feature weights.

---

## 📊 Results

### Method Comparison

| Method | MSE | R² Score | Notes |
|--------|-----|----------|-------|
| **Normal Equation** (ours) | ~913 | **0.7345** | One-shot exact solution |
| **Gradient Descent** (ours) | ~913 | **0.7345** | Converges in ~100 iterations |
| **Ridge Regression** (λ=10) | ~913 | **0.7345** | Slightly shrunk weights |
| scikit-learn (reference) | ~913 | **0.7345** | Industry standard |

**Weight difference vs scikit-learn: `3.16e-13`** — essentially zero, confirming mathematical accuracy.

### Sample Predictions

| Predicted (g/km) | Actual (g/km) | Error |
|-----------------|---------------|-------|
| 250.8 | 253.0 | 2.2 |
| 304.5 | 344.0 | 39.5 |
| 347.7 | 322.0 | 25.7 |

### Gradient Descent Convergence

The cost function drops from ~33,000 to ~447 in the first 100 iterations, then plateaus — confirming convergence to the optimal solution.

```
Cost
33000 |█
      |█
      | █
      |  ██
 1000 |    ████████████████████████████████
  447 |                                    ─────────────
      └─────────────────────────────────────────────────
      0        200       400       600       800      1000
                              Iterations
```

---

## 🏗️ Project Structure

```
Linear_Algebra_Project/
│
├── 📂 src/
│   ├── model.py          # Normal Equation + Gradient Descent + Ridge
│   ├── utils.py          # Data loading, feature selection, normalization
│   └── comparison.py     # Three-way comparison with sklearn
│
├── 📂 data/
│   └── CO2_Emissions.csv # Canadian vehicle emissions dataset
│
├── 📂 plots/
│   └── gradient_descent_convergence.png
│
├── 📂 notebooks/
│   └── exploration.ipynb # Data exploration & visualization
│
├── main.py               # Full pipeline — runs all 4 tasks
├── requirements.txt
└── README.md
```

---

## ⚙️ Implementation Details

### Feature Engineering

Two features were deliberately chosen over the full dataset:

| Feature | Reason |
|---------|--------|
| Engine Size (L) | Strong physical relationship with fuel burn |
| Cylinders | Structural engine complexity indicator |

> Fuel consumption columns were excluded — they directly encode CO₂ (CO₂ ∝ fuel burn), which would make the regression trivially easy and mathematically uninteresting.

### Data Pipeline

1. Load CSV → drop null rows
2. Select meaningful features explicitly
3. Normalize with `StandardScaler` (critical for Gradient Descent convergence)
4. Add bias column (column of 1s) for the intercept term
5. 80/20 train/test split

### Numerical Stability

`np.linalg.solve(XᵀX, Xᵀy)` is used instead of `np.linalg.inv(XᵀX) @ Xᵀy` — solving the linear system directly is more numerically stable than explicitly computing the matrix inverse.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AhmedMohammedRo/Linear_Algebra_Project
cd Linear_Algebra_Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python main.py
```

This will:
- Train all three models (Normal Equation, Gradient Descent, Ridge)
- Display the convergence plot
- Print the full comparison table in the terminal
- Show sample predictions vs actual values

### 4. Explore the notebook (optional)
```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## 🔑 Key Takeaways

**1. Two paths, one destination**  
Normal Equation and Gradient Descent both arrive at identical weights (`diff < 1e-10`), proving that the gradient of the least squares cost function has exactly one global minimum.

**2. Normalization is not optional**  
Without `StandardScaler`, Gradient Descent either diverges or needs thousands more iterations. Feature scaling is what makes the cost surface spherical and easy to navigate.

**3. Regularization is a linear algebra operation**  
Ridge regression adds `λI` to `XᵀX` before inversion — this tiny change guarantees the matrix is invertible even when features are correlated, and shrinks weights to prevent overfitting.

**4. Our implementation matches sklearn to 13 decimal places**  
This validates that the mathematics was implemented correctly with no shortcuts.

---

## 👥 Team

| Name |
|------|
| Omar Shaker |
| Ahmad Roshdy |
| Mark Tamer |
| Khalid Osam |
| Carlos Emad |
| Ahmad Fouad |
| Yousef Hany |
| Mohammad Elsayed |

---

## 📚 Course Information

| | |
|--|--|
| **Course** | Linear Algebra for Computer Science — Math 204 |
| **Level** | First-Year Undergraduate |
| **Instructor** | Dr. Doaa Elsakout |
| **Academic Year** | 2025 / 2026 |
| **Deliverable** | 15-minute group presentation |
| **Weight** | 10% of final grade |

---

<div align="center">

*Built with mathematics, not magic.*

</div>
