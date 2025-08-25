import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import glob


def plot_traces(ax, evidence_traces, n_traces_plot, time_traces, title='Simulated Evidence Traces'):
    """
    Plot evidence traces. Each trace is a list of evidence values over time.
    :param ax: Plot axis
    :param evidence_traces: List of evidence traces, y-axis
    :param n_traces_plot: Number of traces to plot
    :param time_traces: List of time traces, x-axis
    :param title: Title of the plot
    Example:
    >>> evidence_traces = [[0.0, 0.33, 0.67, 0.54, 0.78, 0.5, 0.89, 0.82, 0.97, 0.7, 1.0]]
    >>> time_traces = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    >>> fig, ax = plt.subplots()
    >>> plot_traces(ax, evidence_traces, 1, time_traces)
    >>> plt.show()
    """
    # Plot evidence traces n_traces_plot number of times
    for i in range(n_traces_plot):
        ax.plot(time_traces[i], evidence_traces[i])
    ax.set_xlabel('Time')
    ax.set_ylabel('Evidence')
    ax.set_title(title)


def plot_rts(ax, rt_sim, rt_real):
    # third column of subplots: histogram of response times simulated vs real
    bins = 50
    rt_sim = np.concatenate(rt_sim, axis=0)
    range_min = np.min([np.min(rt_real.numpy()), np.min(rt_sim)])
    range_max = np.max([np.max(rt_real.numpy()), np.max(rt_sim)])
    ax.hist(rt_real.numpy(), bins=bins, range=(range_min, range_max), density=True, alpha=0.5, label='real',
            color='blue')
    ax.hist(rt_sim, bins=bins, range=(range_min, range_max), density=True, alpha=0.5,
            label='simulated', color='orange')
    ax.set_xlabel('Response time')
    ax.set_ylabel('Density')
    ax.set_title('Response time distribution')
    ax.legend()


def plot_rts_multi(ax, rt_dict):
    """
    Plot response times
    :param ax: Plot axis
    :param rt_dict: Dictionary with keys 'rts', 'labels', 'colors'. 'rts' is a list of response times
    """

    bins = 50
    if len(rt_dict['rts']) == 1:
        range_min = np.min(rt_dict['rts'][0])
        range_max = np.max(rt_dict['rts'][0])
        rt_dict['rts'] = np.expand_dims(rt_dict['rts'][0], 0)
    else:
        range_min = np.min([np.min(rt) for rt in rt_dict['rts']])
        range_max = np.max([np.max(rt) for rt in rt_dict['rts']])
    for i, rt in enumerate(rt_dict['rts']):
        ax.hist(rt, bins=bins, range=(range_min, range_max), density=True, alpha=0.5, label=rt_dict['labels'][i],
                color=rt_dict['colors'][i])
    ax.set_xlabel('Response time')
    ax.set_ylabel('Density')
    ax.set_title('Response time distribution')
    ax.legend()


def plot_evidence_update(ax, evidence_new_sim, evidence_new_real, evidence_old_sim, evidence_old_real):
    # Plot evidence update behavior by plotting old evidence on x axis and new evidence on y axis and adding trend line
    # get mean update for each bin of old evidence
    bins = 100
    evidence_old_binned_sim = np.linspace(np.min(evidence_old_sim), np.max(evidence_old_sim), bins)
    evidence_new_binned_sim = np.zeros(bins - 1)
    if evidence_old_real is not None:
        evidence_old_binned_test = np.linspace(np.min(evidence_old_real), np.max(evidence_old_real), bins)
        evidence_new_binned_test = np.zeros(bins - 1)
    else:
        evidence_old_binned_test, evidence_new_binned_test = None, None
    for i in range(bins - 1):
        idx = np.where(
            (evidence_old_sim >= evidence_old_binned_sim[i]) & (evidence_old_sim < evidence_old_binned_sim[i + 1]))[0]
        evidence_new_binned_sim[i] = np.mean(evidence_new_sim[idx])
        if evidence_old_real is not None:
            idx_real = np.where((evidence_old_real >= evidence_old_binned_test[i]) & (
                    evidence_old_real < evidence_old_binned_test[i + 1]))[0]
            evidence_new_binned_test[i] = np.mean(evidence_new_real[idx_real])
    if evidence_old_real is not None:
        ax.scatter(evidence_old_real, evidence_new_real, s=1, alpha=0.1, color='blue')
    ax.scatter(evidence_old_sim, evidence_new_sim, s=1, alpha=0.1, color='orange')
    if evidence_old_binned_test is not None:
        ax.scatter(evidence_old_binned_test[:-1], evidence_new_binned_test, color='blue', s=3, label='Real')
    ax.scatter(evidence_old_binned_sim[:-1], evidence_new_binned_sim, color='orange', s=3, label='Simulated')
    ax.set_xlabel('Old evidence')
    ax.set_ylabel('New evidence')
    ax.grid(True)
    ax.set_title('Evidence update behavior')
    ax.legend()


def plot_pca(ax, evidence_traces_sim, evidence_traces_test):
    pca = PCA(n_components=2)
    pca.fit(evidence_traces_test)
    evidence_traces_sim = np.concatenate(evidence_traces_sim, axis=0).reshape(len(evidence_traces_sim), -1)
    pca_real = pca.transform(evidence_traces_test)
    pca_simulated = pca.transform(evidence_traces_sim)
    ax.scatter(pca_real[:, 0], pca_real[:, 1], s=1, label='Real', color='blue', alpha=0.5)
    ax.scatter(pca_simulated[:, 0], pca_simulated[:, 1], s=1, label='Simulated', color='orange', alpha=0.5)
    ax.set_title('PCA of evidence traces')
    ax.legend()

def plot_drift_rates(drift_rates):
    """
    Plot original and recovered drift rates against each other. Perfect recovery would be a 45 degree line.
    :param drift_rates: List of tuples of original and recovered drift rates, e.g. [(0.5, 0.48), (0.75, 0.68), ...]
    :return: Matplotlib plot object
    Example:
    >>> drifts = [(0.5, 0.48), (0.75, 0.68), (0.9, 0.9), (1.2, 1.1), (1.5, 1.4)]
    >>> plot_drift_rates(drifts).show()
    """
    # Split tuples into two separate lists: original and recovered drift rates
    original, recovered = zip(*drift_rates)

    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(original, recovered, c='blue', label='Data Points')

    # Add 45 degree line for reference
    x = y = plt.xlim()
    plt.plot(x, y, 'r--', label='Perfect Recovery')

    # Adding labels and title
    plt.xlabel('Original Drift Rates')
    plt.ylabel('Recovered Drift Rates')
    plt.title('Original vs Recovered Drift Rates')
    plt.legend()
    plt.grid(True)

    return plt

def save_as_gif(folder):
    """
    save given .png files in the folder as a gif animation

    :param folder: str, path to the folder containing .png files
    :return: None

    Example:
    >>> save_as_gif('/Users/imtezcan/Desktop/plots_example/gru/to_gif')
    """
    # Collect all PNG files in the current directory
    png_files = sorted(glob.glob(f"{dir}/*.png"))

    # Open the images
    images = [Image.open(file) for file in png_files]

    # Save as GIF
    images[0].save(
        "animated.gif",
        save_all=True,
        append_images=images[1:],
        duration=200,  # Duration between frames in milliseconds
        loop=0  # Loop forever
    )

def smooth(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')