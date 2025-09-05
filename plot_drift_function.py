import numpy as np
import matplotlib.pyplot as plt

def plot_drift_function():
    """
    Plot the non-linear drift function: v = mu + t/(1.0 + t)
    where mu = 0.5 and t ranges from 0 to 5
    """
    # Parameters
    mu = 0.5
    t_start = 0.0
    t_end = 5.0
    n_steps = 1000
    
    # Generate time points
    t = np.linspace(t_start, t_end, n_steps)
    
    # Calculate drift values
    v = mu + t / (1.0 + t)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, v, 'b-', linewidth=2, label=f'v = {mu} + t/(1.0 + t)')
    
    # Add horizontal line for mu (asymptote)
    plt.axhline(y=mu, color='r', linestyle='--', alpha=0.7, label=f'μ = {mu} (asymptote)')
    
    # Add horizontal line for initial value
    initial_v = mu + 0 / (1.0 + 0)  # v at t=0
    plt.axhline(y=initial_v, color='g', linestyle=':', alpha=0.7, label=f'Initial v = {initial_v}')
    
    # Customize the plot
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Drift Rate (v)', fontsize=12)
    plt.title('Non-linear Drift Function: v = μ + t/(1.0 + t)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Set axis limits
    plt.xlim(t_start, t_end)
    plt.ylim(0, max(v) * 1.1)
    
    # Add some annotations
    plt.annotate(f'v(0) = {initial_v:.2f}', xy=(0, initial_v), xytext=(0.5, initial_v + 0.1),
                arrowprops=dict(arrowstyle='->', color='green'), fontsize=10)
    
    plt.annotate(f'v(5) = {v[-1]:.2f}', xy=(5, v[-1]), xytext=(3.5, v[-1] + 0.1),
                arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print some key values
    print(f"Initial drift rate (t=0): v = {v[0]:.4f}")
    print(f"Final drift rate (t=5): v = {v[-1]:.4f}")
    print(f"Change in drift rate: Δv = {v[-1] - v[0]:.4f}")
    print(f"Asymptotic value (t→∞): v = {mu:.4f}")

if __name__ == "__main__":
    plot_drift_function()
