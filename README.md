# 💼 IT Salary Predictor (Machine Learning from Scratch)

A Machine Learning project that predicts IT engineer salaries using **Linear Regression implemented from scratch** with **Batch Gradient Descent**.

This project does not use scikit-learn — the full learning algorithm (cost function, gradient computation, and gradient descent) is manually implemented using NumPy.

---

## 📌 Project Overview

The model predicts salary based on the following features:

- Experience  
- Years in Company  
- Certifications  
- Projects Completed  

The goal of this project is to deeply understand:

- Linear Regression  
- Cost Function (MSE)  
- Gradient Computation  
- Batch Gradient Descent  
- Model convergence visualization  

---

## 📂 Results

### Cost vs Iterations

<p align="center">
  <img src="it-salary-predictor/screenshot/Figure_1.png" width="600">
</p>

### Training Output

<p align="center">
  <img src="it-salary-predictor/screenshot/swappy-20260215_014026.png" width="600">
</p>

---

## 📂 Project Structure

```bash
it-salary-predictor/
│
├── train.py               # Training script
├── model.py               # Linear Regression implementation
├── salary_dataset.csv     # Dataset
├── requirements.txt
├── .gitignore
└── cost_plot.png          # Generated training cost plot
```bash

---

## 🧠 How It Works

### 1️⃣ Model Implementation (`model.py`)

- `predict()` → Computes predictions  
- `compute_cost()` → Mean Squared Error cost function  
- `compute_gradient()` → Computes gradients  
- `gradient_descent()` → Optimizes parameters  

### 2️⃣ Training Script (`train.py`)

- Loads dataset  
- Initializes weights and bias  
- Runs Gradient Descent  
- Prints predictions  
- Saves cost convergence plot (`cost_plot.png`)  

---

## 📊 Cost Function

The model minimizes the Mean Squared Error:

J(w,b) = (1 / 2m) * Σ (f(x) - y)^2

---

## 🚀 How to Run

### 1️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```bash

### 2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3 Run Training

```bash
python train.py
```

After training, the script will output predictions, final optimized weights and bias, and save the cost convergence plot as `cost_plot.png`.

---

## 📂 Technologies Used

- Python  
- NumPy  
- Pandas  
- Matplotlib  

--- 

## 📂 Project Purpose

This project demonstrates:

- Understanding of core Machine Learning fundamentals  
- Ability to implement algorithms from scratch  
- Knowledge of optimization and gradient-based learning  
- Clean project structuring for reproducibility  

---

## 📂 Author

Amine El-baydaouy  
Machine Learning Enthusiast
