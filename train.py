import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from model import gradient_descent, compute_gradient, compute_cost

data = pd.read_csv("salary_dataset.csv")

x = data[['Experience','YearsInCompany','Certifications','ProjectsCompleted']].values
y = data['Salary'].values

b_init = 8000
w_init = np.array([100, 200, 300, 400])

initial_w = np.zeros_like(w_init)
initial_b = 0
iterations = 1500
alpha = 1.0e-3
w_final, b_final, j_hist = gradient_descent(x, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations )
print(f"b,w found by gradient descent:{b_final},{w_final}")
m,_ = x.shape
for i in range(m):
    print(f"prediction: {np.dot(x[i], w_final) + b_final}, target value: {y[i]}")

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(j_hist)
ax2.plot(100 + np.arange(len(j_hist[100:])), j_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.savefig("cost_plot.png")
print("Plot saved as cost_plot.png")