import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def generate_gaussian_data(n_samples):
    np.random.seed(42)  # For reproducibility
    # Two Gaussian blobs
    data1 = np.random.normal(loc=-2, scale=0.5, size=(n_samples // 2, 1))
    data2 = np.random.normal(loc=3, scale=1.0, size=(n_samples // 2, 1))
    return np.vstack([data1, data2])


def generate_epanechnikov_data(n_samples):
    return np.random.uniform(low=-1, high=1, size=(n_samples, 1))


def generate_exponential_data(n_samples=500):
    return np.random.exponential(scale=1.0, size=(n_samples, 1))


def generate_cosine_data(n_samples=500):
    x = np.linspace(0, 20 * np.pi, n_samples).reshape(-1, 1)
    y = np.cos(x) + 0.01 * np.random.normal(scale=0.1, size=(n_samples, 1))
    return y


def generate_uniform_data(n_samples=5000):
    return np.random.uniform(-1, 1, size=(n_samples, 1))


def generate_dataset():
    return None


if __name__ == '__main__':
    X = generate_uniform_data(n_samples=5000)
    plt.hist(X, bins=30, density=True, alpha=0.5, label='Histogram of data')
    plt.savefig('histogram_v.png')

    # Define parameter grid
    param_grid = {
        'kernel': [
            'gaussian',
            'tophat',
            'epanechnikov',
            'exponential',
            'linear',
            'cosine',
        ],
        'bandwidth': np.linspace(0.1, 1.0, 10),
    }

    # Set up GridSearch
    grid = GridSearchCV(KernelDensity(), param_grid, cv=5)
    grid.fit(X)

    print('Best Parameters:', grid.best_params_)

    # Generate test points for plotting
    X_plot = np.linspace(-6, 8, 1000)[:, np.newaxis]

    # Best model
    kde = grid.best_estimator_

    # Evaluate density
    log_dens = kde.score_samples(X_plot)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.hist(X, bins=30, density=True, alpha=0.5, label='Histogram of data')
    plt.plot(
        X_plot[:, 0], np.exp(log_dens), label=f"KDE ({grid.best_params_['kernel']})"
    )
    plt.legend()
    plt.savefig('best_kernel_density_estimation_v2.png')

    # # 2. Set up KDE models
    # kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    # bandwidth = 0.3  # Manually tuned based on visualization

    # # 3. Plot
    # X_plot = np.linspace(-2, 2, 1000).reshape(-1, 1)

    # plt.figure(figsize=(12, 8))
    # for kernel in kernels:
    #     kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    #     kde.fit(X)
    #     log_dens = kde.score_samples(X_plot)
    #     plt.plot(X_plot[:, 0], np.exp(log_dens), label=kernel)

    # plt.hist(X, bins=50, density=True, alpha=0.5, label='Data histogram')
    # plt.legend()
    # plt.title("KDE with Different Kernels on Periodic (Sine) Data")
    # plt.show()
