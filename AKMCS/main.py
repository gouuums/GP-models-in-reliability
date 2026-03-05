# Application - AMGPRA, Example 5.1 - Reproduction: 5 times
def g0_function(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    
    term1 = (x1**2 + 4) * (x2 - 1) / 20.0
    term2 = np.sin(2.5 * x1) 
    
    return (2.0 - term1 + term2).reshape(-1, 1)

if __name__ == "__main__":
    n_runs = 5
    n_mcs_train = 10000      
    n_mcs_eval = 1000000     
    dim = 2
    
    all_pfs = []
    all_calls = []
        
    for run_idx in range(n_runs):
        print(f"--- Run: {run_idx + 1}/{n_runs} ---")
        
        S_train = np.zeros((n_mcs_train, dim))
        S_train[:, 0] = np.random.normal(loc=1.5, scale=1.0, size=n_mcs_train)
        S_train[:, 1] = np.random.normal(loc=2.5, scale=1.0, size=n_mcs_train)
        
        solver = AKMCS(performance_function=g0_function, population=S_train, n_init=6, eff_stop=0.001)
        solver.run() 
        
        S_eval = np.zeros((n_mcs_eval, dim))
        S_eval[:, 0] = np.random.normal(loc=1.5, scale=1.0, size=n_mcs_eval)
        S_eval[:, 1] = np.random.normal(loc=2.5, scale=1.0, size=n_mcs_eval)
        
        y_pred_final = solver.model.predict_values(S_eval)
        final_pf = np.sum(y_pred_final <= 0) / n_mcs_eval
        
        all_pfs.append(final_pf)
        all_calls.append(len(solver.doe_y))
        
        print(f"End Run {run_idx + 1}: Number of calls to g0 = {len(solver.doe_y)}, Pf = {final_pf:.6e}\n")

    print("=========================================")
    print("Average results for 5 runs")
    print("=========================================")
    print(f"Estimated Pf: {np.mean(all_pfs):.6e} (Ref. ~ 3.13e-2)")
    print(f"Average number of calls to g0: {np.mean(all_calls):.1f} (Ref. ~ 45.2)")
    

    
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    calls_axis = range(solver.n_init, solver.n_init + len(solver.pf_history))
    plt.plot(calls_axis, solver.pf_history, marker='o', linestyle='-', color='b', markersize=4)
    plt.axhline(y=0.0313, color='r', linestyle='--', label='Ref. Pf (MCS) ~ 0.0313')
    plt.xlabel('Number of calls to function g0')
    plt.ylabel('Estimated Pf')
    plt.title('Convergence of the Pf')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    
    x1_range = np.linspace(-4, 7, 200)
    x2_range = np.linspace(-3, 8, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    X_grid = np.column_stack((X1.flatten(), X2.flatten()))
    
    Z = g0_function(X_grid).reshape(X1.shape)
    
    contour_bg = plt.contourf(X1, X2, Z, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour_bg, label='g0(x) value')
    
    plt.contour(X1, X2, Z, levels=[0], colors='red', linewidths=2.5)
    
    plt.scatter(solver.doe_x[:, 0], solver.doe_x[:, 1], color='white', edgecolor='black', 
                s=40, label='Evaluated points - DoE', zorder=3)
    
    plt.xlim([-4, 7])
    plt.ylim([-3, 8])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('High fidelity model g0 contours \nand limit-sate (red)')
    plt.legend()

    plt.tight_layout()
    plt.show()
