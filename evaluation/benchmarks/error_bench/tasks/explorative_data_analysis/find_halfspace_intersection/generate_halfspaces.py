import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, HalfspaceIntersection


def find_halfspaces_intersection(halfspaces, interior_point):
    """
    Find the intersection of halfspaces using HalfspaceIntersection.
    :param halfspaces (numpy.ndarray): Halfspaces in the form A*x + b <= 0
    :param interior_point (numpy.ndarray): A point inside the intersection
    :return: numpy.ndarray: Vertices of the intersection polytope
    """
    return HalfspaceIntersection(halfspaces, interior_point).intersections


def generate_sample_bounded_halfspaces(n_halfspaces=3, n_dimensions=2):
    """
    Generate random bounded Halfspaces of the form A*x + b <= 0.
    By defining a Convex Hull, we can ensure that the halfspaces are bounded, so the intersections are not empty.
    The intersection of the halfspaces is guaranteed to be a convex polytope.

    :param n_halfspaces (int): Number of halfspaces to generate for each sample
    :param n_dimensions (int): Number of dimensions for the halfspaces of each sample
    :return pandas.DataFrame: DataFrame containing the halfspaces and their labels
    """
    # Generate random points
    points = np.random.uniform(-10, 10, (n_halfspaces, n_dimensions))

    # Create a convex hull from the points
    hull = ConvexHull(points)

    # Create halfspaces from the convex hull equations
    halfspaces = hull.equations

    # Create a guaranteed interior point
    interior_point = np.mean(points, axis=0)

    # Generate labels for the halfspaces, i.e. the intersection of the three halfspaces
    h_intersections = HalfspaceIntersection(
        hull.equations, interior_point
    ).intersections

    return halfspaces, interior_point, h_intersections


def generate_dataset(num_rows=100, n_halfspaces=3, n_dimensions=2):
    """
    Generates a dataset of halfspaces and their intersections.
    Example format:
    'h1_a1', 'h1_a2', 'h1_b', 'h2_a1', 'h2_a2', 'h2_b', 'h3_a1', 'h3_a2', 'h3_b', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'
    where h1, h2, h3 are the halfspaces parametrized by a1, a2, b (i.e. A*x + b <= 0), and x1, y1, x2, y2, x3, y3 are the vertices of the intersection.

    :param num_rows (int): Number of samples to generate
    :param n_halfspaces (int): Number of halfspaces to generate for each sample
    :param n_dimensions (int): Number of dimensions for the halfspaces of each sample
    """
    data = []
    for _ in range(num_rows):
        halfspaces, interior_point, h_intersections = (
            generate_sample_bounded_halfspaces(n_halfspaces, n_dimensions)
        )
        # Flatten the halfspaces and add to the data
        flat_halfspaces = halfspaces.flatten()
        data.append(np.concatenate((flat_halfspaces, h_intersections.flatten())))

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    # Generate column names
    columns = []
    for i in range(n_halfspaces):
        for j in range(n_dimensions):
            columns.append(f'h{i+1}_a{j+1}')
        columns.append(f'h{i+1}_b')

    for i in range(len(h_intersections)):
        for j in range(n_dimensions):
            columns.append(f'x{i+1}_{j+1}')

    df.columns = columns

    return df


def plot_sample_halfspaces_w_intersection(halfspaces, h_intersections):
    """
    Plot the halfspaces and their intersection.
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html#scipy.spatial.HalfspaceIntersection
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, aspect='equal')

    # Set limits to cover your intersections properly
    x_bound = min(np.min(h_intersections, axis=0))
    y_bound = max(np.max(h_intersections, axis=0))

    xlim, ylim = (x_bound - 1, y_bound + 1), (x_bound - 1, y_bound + 1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    x = np.linspace(xlim[0], xlim[1], 400)

    symbols = ['-', '+', 'x']  # Only 3 symbols for 3 halfspaces
    signs = [0, 0, -1]  # Adjust signs appropriately
    fmt = {'color': None, 'edgecolor': 'b', 'alpha': 0.3}

    for h, sym, sign in zip(halfspaces, symbols, signs):
        hlist = h.tolist()
        fmt['hatch'] = sym
        if h[1] == 0:
            x_val = -h[2] / h[0]
            ax.axvline(x_val, label='{}x + {}y + {} = 0'.format(*np.round(hlist, 2)))
            xi = np.linspace(xlim[sign], x_val, 100)
            ax.fill_between(xi, ylim[0], ylim[1], **fmt)
        else:
            y = (-h[2] - h[0] * x) / h[1]
            ax.plot(x, y, label='{}x + {}y + {} = 0'.format(*np.round(hlist, 2)))
            ax.fill_between(x, y, ylim[sign], **fmt)

    # Plot intersections
    x_pts, y_pts = h_intersections[:, 0], h_intersections[:, 1]
    ax.plot(x_pts, y_pts, 'o', markersize=8, color='red', label='Intersections')

    ax.legend()
    plt.savefig('halfspaces_intersection.png')
    return None


if __name__ == '__main__':
    # Generate and save the dataset
    n_halfspaces = 3
    n_dimensions = 2
    df = generate_dataset(
        num_rows=100, n_halfspaces=n_halfspaces, n_dimensions=n_dimensions
    )
    print(df.head())

    # Save the dataset to a CSV file
    df.to_csv('halfspaces_dataset.csv', index=False)

    # Plot a sample of halfspaces and their intersection
    sample_idx = 0
    halfspaces = df.iloc[sample_idx][
        : n_halfspaces * (n_dimensions + 1)
    ].values.reshape(n_halfspaces, (n_dimensions + 1))
    intersections = df.iloc[sample_idx][
        n_halfspaces * (n_dimensions + 1) :
    ].values.reshape(n_halfspaces, n_dimensions)

    plot_sample_halfspaces_w_intersection(halfspaces, intersections)
