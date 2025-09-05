import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_drift_function_2d():
    """
    Plot the 2D non-linear drift function: 
    v = mu*(1.0 + k*t) - (l - k/(1.0 + k*t))*x
    
    where mu=0.5, k=0.5, l=0.0, t goes from 0 to 5, and x goes from 0 to 1
    """
    # Parameters
    mu = 0.5
    k = 0.5
    l = 0.0
    
    # Generate time points
    t = np.linspace(0, 2, 10000)
    
    # Position x is a function of time: x goes from 0.0 to 1.0 as t goes from 0.0 to 2.0
    x = t / 2.0  # Linear mapping: x = t/2
    
    # Calculate drift values (now 1D since x depends on t)
    v = mu * (1.0 + k * t) - (l - k / (1.0 + k * t)) * x
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Main drift function plot
    ax1.plot(t, v, 'b-', linewidth=3, label='Drift rate v(t)')
    ax1.set_xlabel('Time (t)', fontsize=12)
    ax1.set_ylabel('Drift Rate (v)', fontsize=12)
    ax1.set_title('Drift Function: v = μ(1+kt) - (l-k/(1+kt))x(t)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Position function plot
    ax2.plot(t, x, 'r-', linewidth=3, label='Position x(t) = t/2')
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Position (x)', fontsize=12)
    ax2.set_title('Position Function: x(t) = t/2', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.show()
    plt.show()
    
    # Print some key values
    print(f"Parameters: μ = {mu}, k = {k}, l = {l}")
    print(f"Position function: x(t) = t/2")
    print(f"Initial drift rate (t=0, x=0): v = {v[0]:.4f}")
    print(f"Final drift rate (t=2, x=1): v = {v[-1]:.4f}")
    
    # Show the mathematical form
    print(f"\nMathematical form:")
    print(f"v = {mu}*(1.0 + {k}*t) - ({l} - {k}/(1.0 + {k}*t))*x(t)")
    print(f"v = {mu} + {mu*k}*t - {l}*x(t) + {k}*x(t)/(1.0 + {k}*t)")
    print(f"where x(t) = t/2")
    
    # Calculate range of values
    v_min, v_max = np.min(v), np.max(v)
    print(f"\nRange of drift rates: [{v_min:.4f}, {v_max:.4f}]")
    print(f"Total variation: {v_max - v_min:.4f}")
    
    # Show some intermediate values
    mid_idx = len(t) // 2
    print(f"Mid drift rate (t=1, x=0.5): v = {v[mid_idx]:.4f}")

if __name__ == "__main__":
    plot_drift_function_2d()
