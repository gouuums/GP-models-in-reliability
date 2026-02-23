import numpy as np
from code_amgpra import AMGPRA
import matplotlib.pyplot as plt

def g_hf(x):
    return 2 - (x[:,0]**2 + 4)*(x[:,1] - 1)/20 + np.sin(5*x[:,0]/2)

def g_lf(x):
    return g_hf(x) - np.sin(5*x[:,0]/22 + 5*x[:,1]/44 + 5/4)

n_mc = 100000
x1 = np.random.normal(1.5, 1.0, n_mc)
x2 = np.random.normal(2.5, 1.0, n_mc)
S_val = np.column_stack([x1, x2])

functions = [g_hf, g_lf]
costs = [1.0, 0.1]
xlimits = np.array([[-4.0, 7.0], [-2.0, 8.0]])

solver = AMGPRA(functions, costs, xlimits, S_candidate=S_val)
pf_result = solver.run()

print(f"\nFinal probability of failure : {pf_result:.4e}")

def plot_results(solver):
    x = np.linspace(solver.xlimits[0,0], solver.xlimits[0,1], 100)
    y = np.linspace(solver.xlimits[1,0], solver.xlimits[1,1], 100)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    
    Z, _ = solver.model.predict(grid)
    Z = Z.reshape(X.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=[0], colors='blue', linestyles='--')
    
    plt.scatter(solver.xt[0][:,0], solver.xt[0][:,1], c='red', marker='*', label='HF Training Points')
    if len(solver.xt) > 1:
        plt.scatter(solver.xt[1][:,0], solver.xt[1][:,1], c='green', marker='s', alpha=0.5, label='LF1 Training Points')
    if len(solver.xt) > 2:
        plt.scatter(solver.xt[2][:,0], solver.xt[2][:,1], c='orange', marker='o', alpha=0.3, label='LF2 Training Points')
        
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('AMGPRA - Final Metamodel Limit State and Training Points')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(solver)
