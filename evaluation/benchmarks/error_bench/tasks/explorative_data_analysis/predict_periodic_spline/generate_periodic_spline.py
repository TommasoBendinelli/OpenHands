import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


def periodic_spline_to_be_learned(x):
    """
    Function to be approximated by the SplineTransformer.
    """
    return np.sin(x) + 0.3 * np.cos(x * 3)


def generate_dataset(
    function=periodic_spline_to_be_learned, num_train=30, num_test=200
):
    """
    Generate a dataset of a periodic function where the function is known.
    function: function to be approximated
    num_points: number of points to sample
    """
    # 2. Define the domain
    x_min, x_max = 0, 4 * np.pi

    # Generate 30 training points
    x_train = np.linspace(x_min, x_max, num_train)
    y_train = function(x_train)

    # Generate 200 test points (dense grid for plotting or testing)
    x_test = np.linspace(x_min, x_max, num_test)
    y_test = function(x_test)

    # Create DataFrames
    train_df = pd.DataFrame({'x': x_train, 'y': y_train})
    test_df = pd.DataFrame({'x': x_test, 'y': y_test})

    return train_df, test_df


def period_spline_transformer(knots=10, degree=3):
    """
    Apply SplineTransformer to the input data.
    x: input data
    n_knots: number of knots
    degree: degree of the spline
    """
    # Create a SplineTransformer with periodic boundary conditions
    transformer = SplineTransformer(
        knots=knots, degree=degree, extrapolation='periodic'
    )

    # Fit the transformer to the data
    model = make_pipeline(transformer, Ridge(alpha=1e-3))

    return model


if __name__ == '__main__':
    # Generate the dataset and save it
    train_df, test_df = generate_dataset()
    print(train_df.head())

    # Save the dataset to a CSV file
    train_df.to_csv('periodic_spline_dataset_train.csv', index=False)
    test_df.to_csv('periodic_spline_dataset_test.csv', index=False)

    # Fit the model
    model = period_spline_transformer(knots=np.linspace(0, 4 * np.pi, 20)[:, None])
    model.fit(train_df[['x']], train_df['y'])
    # Predict on the test set
    y_pred = model.predict(test_df[['x']])

    # Plot the ground truth and the predictions
    plt.figure(figsize=(10, 6))
    plt.plot(test_df['x'], test_df['y'], label='Original Function', color='blue')
    plt.plot(test_df['x'], y_pred, label='Spline Prediction', color='red')
    plt.title('Periodic Spline Dataset')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig('periodic_spline_plot.png')
