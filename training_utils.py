import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
import seaborn as sns

def compute_kld_np(dist1, dist2, num_bins=50):
    # Create histograms with same bin range
    min_val = min(np.min(dist1), np.min(dist2))
    max_val = max(np.max(dist1), np.max(dist2))

    hist1, _ = np.histogram(dist1, bins=num_bins, range=(min_val, max_val))
    hist2, _ = np.histogram(dist2, bins=num_bins, range=(min_val, max_val))

    # Convert to probabilities
    p = hist1 / hist1.sum()
    q = hist2 / hist2.sum()

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    # Calculate KL divergence
    kl_div = np.sum(p * (np.log(p) - np.log(q)))

    return float(kl_div)


def compute_kld(rt_real, rt_pred, n_bins=50):
    min_val = torch.min(rt_real.min(), rt_pred.min()).item()
    max_val = torch.max(rt_real.max(), rt_pred.max()).item()

    hist1, bin_edges = torch.histogram(rt_real, bins=n_bins, range=[min_val, max_val])
    hist2, _ = torch.histogram(rt_pred, bins=n_bins, range=[min_val, max_val])

    # Normalize histograms to get probability distributions
    p = hist1 / hist1.sum()
    q = hist2 / hist2.sum()

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    return torch.sum(p * (torch.log(p) - torch.log(q))).detach().cpu()

def fit_kdes(rt1, rt2, bandwidth='scott', kernel='gaussian'):
    kde1 = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(rt1.reshape(-1, 1))
    kde2 = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(rt2.reshape(-1, 1))

    return kde1, kde2


def fit_kde_unimodal(rt, bandwidth='scott', kernel='gaussian'):
    rt_positive = rt[rt >= 0]
    rt_negative = rt[rt < 0]

    kde_positive = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(rt_positive.reshape(-1, 1))
    kde_negative = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(rt_negative.reshape(-1, 1))

    return kde_positive, kde_negative

def compute_lls(kde_positive, kde_negative, rt):
    rt_positive = rt[rt >= 0]
    rt_negative = rt[rt < 0]

    ll_pop = kde_positive.score_samples(rt_positive.reshape(-1, 1)).sum()
    ll_non = kde_negative.score_samples(rt_negative.reshape(-1, 1)).sum()

    ll_pon = kde_negative.score_samples(rt_positive.reshape(-1, 1)).sum()
    ll_nop = kde_positive.score_samples(rt_negative.reshape(-1, 1)).sum()

    return (ll_pop, ll_non), (ll_pon, ll_nop)

def get_lls(rt1, rt2, bandwidth='scott', spacing=1000, bins=50, kernel='gaussian'):
    kde1, kde2 = fit_kdes(rt1, rt2, bandwidth=bandwidth, kernel=kernel)

    # Calculate log-likelihoods
    # How well does set 1 fit the distribution of set 1
    ll_1_on_1 = kde1.score(rt1.reshape(-1, 1)) / rt1.shape[0]

    # How well does set 1 fit the distribution of set 2
    ll_1_on_2 = kde2.score(rt1.reshape(-1, 1)) / rt1.shape[0]

    # How well does set 2 fit the distribution of set 1
    ll_2_on_1 = kde1.score(rt2.reshape(-1, 1)) / rt2.shape[0]

    # How well does set 2 fit the distribution of set 2
    ll_2_on_2 = kde2.score(rt2.reshape(-1, 1)) / rt2.shape[0]

    # Visualize the KDE fits to check if they capture bimodality
    x_grid = np.linspace(min(rt1.min(), rt2.min()),
                         max(rt1.max(), rt2.max()), spacing)

    # Get density estimates
    pdf1 = np.exp(kde1.score_samples(x_grid.reshape(-1, 1)))
    pdf2 = np.exp(kde2.score_samples(x_grid.reshape(-1, 1)))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(rt1, bins=bins, density=True, alpha=0.5, label='RT Set 1')
    plt.hist(rt2, bins=bins, density=True, alpha=0.5, label='RT Set 2')
    plt.plot(x_grid, pdf1, 'r-', label='KDE Set 1')
    plt.plot(x_grid, pdf2, 'b-', label='KDE Set 2')
    plt.legend()
    plt.xlabel('Response Time')
    plt.ylabel('Density')
    plt.title('KDE of Bimodal Response Time Distributions')
    plt.show()

    ll_matrix = np.array([
        [ll_1_on_1, ll_2_on_1],
        [ll_1_on_2, ll_2_on_2]
    ])
    # Create a heatmap to visualize the log-likelihoods
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(ll_matrix, annot=True, fmt=".2f", cmap="Blues",
                     cbar_kws={'label': 'Log-Likelihood'})
    ax.set_xticklabels(['RT1', 'RT2'])
    ax.set_yticklabels(['KDE1', 'KDE2'], rotation=0)
    plt.xlabel('Response Times')
    plt.ylabel('Kernel Density Estimators')
    plt.title('Log-Likelihood Heatmap')
    plt.show()

    return ll_1_on_1, ll_2_on_1, ll_1_on_2, ll_2_on_2


def get_lls_multi(rt_dict, bandwidth='scott', spacing=1000, bins=50, kernel='gaussian'):
    n_sets = len(rt_dict['rts'])
    kdes = []

    # Fit KDEs for each RT set
    for rts in rt_dict['rts']:
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(rts.reshape(-1, 1))
        kdes.append(kde)

    # Calculate log-likelihoods matrix
    ll_matrix = np.zeros((n_sets, n_sets))
    for i in range(n_sets):
        for j in range(n_sets):
            # Score RT set i against KDE j
            ll_matrix[i, j] = kdes[j].score(rt_dict['rts'][i].reshape(-1, 1)) / rt_dict['rts'][i].shape[0]

    # Plotting
    # KDE visualization
    plt.figure(figsize=(10, 6))

    # Find global min and max for x_grid
    min_val = min(rt.min() for rt in rt_dict['rts'])
    max_val = max(rt.max() for rt in rt_dict['rts'])
    x_grid = np.linspace(min_val, max_val, spacing)

    # Plot histograms and KDEs
    for i, (rts, label) in enumerate(zip(rt_dict['rts'], rt_dict['labels'])):
        plt.hist(rts, bins=bins, density=True, alpha=0.5, label=f'{label} (hist)')
        pdf = np.exp(kdes[i].score_samples(x_grid.reshape(-1, 1)))
        plt.plot(x_grid, pdf, '-', label=f'{label} (KDE)')

    plt.legend()
    plt.xlabel('Response Time')
    plt.ylabel('Density')
    plt.title('KDE of Response Time Distributions')
    plt.show()

    # Log-likelihood heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(ll_matrix, annot=True, fmt=".2f", cmap="Blues",
                     cbar_kws={'label': 'Log-Likelihood'})
    ax.set_xticklabels(rt_dict['labels'])
    ax.set_yticklabels(rt_dict['labels'], rotation=0)
    plt.xlabel('Response Times')
    plt.ylabel('Kernel Density Estimators')
    plt.title('Log-Likelihood Heatmap')
    plt.show()

    return ll_matrix

def get_lls_unimodal(rt1, rt2, bandwidth='scott'):
    kde1, kde2 = fit_kdes(rt1, rt2, bandwidth=bandwidth)

    # Calculate log-likelihoods
    # How well does set 1 fit the distribution of set 1
    log_likelihood_1_on_1 = kde1.score_samples(rt1.reshape(-1, 1))

    # How well does set 1 fit the distribution of set 2
    log_likelihood_1_on_2 = kde2.score_samples(rt1.reshape(-1, 1))

    # How well does set 2 fit the distribution of set 1
    log_likelihood_2_on_1 = kde1.score_samples(rt2.reshape(-1, 1))

    # How well does set 2 fit the distribution of set 2
    log_likelihood_2_on_2 = kde2.score_samples(rt2.reshape(-1, 1))

    # Calculate total log-likelihood (sum of individual log-likelihoods)
    total_log_likelihood_1_on_1 = np.sum(log_likelihood_1_on_1)
    total_log_likelihood_1_on_2 = np.sum(log_likelihood_1_on_2)
    total_log_likelihood_2_on_1 = np.sum(log_likelihood_2_on_1)
    total_log_likelihood_2_on_2 = np.sum(log_likelihood_2_on_2)

    # Visualize the KDE fits to check if they capture bimodality
    x_grid = np.linspace(min(rt1.min(), rt2.min()),
                         max(rt1.max(), rt2.max()), 1000)

    # Get density estimates
    pdf1 = np.exp(kde1.score_samples(x_grid.reshape(-1, 1)))
    pdf2 = np.exp(kde2.score_samples(x_grid.reshape(-1, 1)))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.hist(rt1, bins=50, density=True, alpha=0.5, label='RT Set 1')
    plt.hist(rt2, bins=50, density=True, alpha=0.5, label='RT Set 2')
    plt.plot(x_grid, pdf1, 'r-', label='KDE Set 1')
    plt.plot(x_grid, pdf2, 'b-', label='KDE Set 2')
    plt.legend()
    plt.xlabel('Response Time')
    plt.ylabel('Density')
    plt.title('KDE of Bimodal Response Time Distributions')
    plt.show()

    return total_log_likelihood_1_on_1, total_log_likelihood_2_on_1, total_log_likelihood_1_on_2, total_log_likelihood_2_on_2

def compute_kde(rt_real, rt_pred, t_max, seq_len):
    grid = np.linspace(0, t_max, seq_len)
    kde_real = gaussian_kde(rt_real.T)(grid)
    kde_generated = gaussian_kde(rt_pred.T)(grid)
    return np.mean((kde_generated - kde_real) ** 2)
