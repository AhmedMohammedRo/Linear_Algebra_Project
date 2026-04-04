# 📊 Linear Regression from Scratch

![Python](https://img.shields.io/badge/Python-3.14-blue)
![ML](https://img.shields.io/badge/Field-Machine%20Learning-blue)
![Linear Algebra](https://img.shields.io/badge/Math-Linear%20Algebra-orange)
![Data Science](https://img.shields.io/badge/Domain-Data%20Science-teal)

---

## 📌 Project Overview

This project implements **Linear Regression** using core concepts from Linear Algebra, focusing on:

- Normal Equation (Closed-form solution)
- Gradient Descent optimization
- Ridge Regression (Regularization)
- Evaluation on real datasets

The goal is to bridge **mathematical theory** with **practical machine learning applications**.

---

## 🧠 Concepts Covered

- Least Squares Solution  
- Matrix Multiplication & Inversion  
- Gradient Descent  
- Regularization (Ridge Regression)  
- Overfitting vs Underfitting  

---

## ⚙️ Features

- ✅ Linear Regression using **Normal Equation**
- ✅ Linear Regression using **Gradient Descent**
- ✅ Ridge Regression implementation
- ✅ Comparison between methods
- ✅ Tested on real-world datasets
- ✅ Clean and modular Python code

---

## 🛠️ Tech Stack

- Python  
- NumPy  
- Pandas  
- Jupyter Notebook (for visualization)  
- scikit-learn 

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/AhmedMohammedRo/Linear_Algebra_Project/
cd Linear_Algebra_Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📐 Mathematical Background

### 🔹 Normal Equation
```
θ = (XᵀX)⁻¹Xᵀy
```
- Direct solution  
- No iterations needed  
- Expensive for large datasets  

---

### 🔹 Gradient Descent
```
θ := θ - α ∇J(θ)
```
- Iterative optimization  
- Works well for large datasets  
- Requires tuning learning rate  

---

### 🔹 Ridge Regression
```
θ = (XᵀX + λI)⁻¹Xᵀy
```
- Prevents overfitting  
- Adds penalty term  

---

## 📊 Results & Comparison

| Method            | Speed ⚡              | Accuracy 🎯            | Scalability 📈        |
|------------------|---------------------|------------------------|----------------------|
| Normal Equation  | Fast (small data)   | High                   | Poor                 |
| Gradient Descent | Medium              | High                   | Excellent            |
| Ridge Regression | Medium              | Better generalization  | Excellent            |

---

## 🎯 Key Insights

- Normal Equation is elegant but not scalable  
- Gradient Descent is more practical in real-world ML  
- Regularization significantly improves model performance  

---

## 👥 Team Members

- Omar Shaker 
- Ahmad Roshdy
- Mark Tamer
- Khalid
- Carlos emad
- Ahmad Fouad
- Yousef Hany
- Mohammad Elsayed

---

## 📎 Future Improvements

- Add Polynomial Regression  
- Implement Lasso Regression  
- Hyperparameter tuning  
- Build a simple UI for visualization  

---

## ⭐ Final Note

This project demonstrates how Linear Algebra directly powers Machine Learning, turning abstract math into real-world predictive systems.
