import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.
    """

    X, y, true_coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )

    return X, y, float(true_coef)


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers.
    """

    rng = np.random.RandomState(random_state)

    X_out = X.copy()
    y_out = y.copy()

    X_out[:n_outliers] = 10 + 0.75 * rng.normal(
        size=(n_outliers, 1)
    )

    y_out[:n_outliers] = -15 + 20 * rng.normal(
        size=n_outliers
    )

    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot dataset highlighting outliers.
    """

    fig, ax = plt.subplots()

    ax.scatter(
        X[n_outliers:],
        y[n_outliers:],
        label="Normal Data"
    )

    ax.scatter(
        X[:n_outliers],
        y[:n_outliers],
        label="Artificial Outliers"
    )

    ax.set_title("Dataset with Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):

    model = LinearRegression()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_huber_regression(X, y):

    model = HuberRegressor()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):

    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):

    model = TheilSenRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):

    return {
        key: abs(value - true_coef)
        for key, value in coef_dict.items()
    }


def best_robust_model(errors):

    robust_errors = {
        "huber_regression":
            errors["huber_regression"],
        "ransac_regression":
            errors["ransac_regression"],
        "theilsen_regression":
            errors["theilsen_regression"]
    }

    return min(
        robust_errors,
        key=robust_errors.get
    )


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    RANSAC outlier detection summary.
    """

    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    total_outliers_detected = int(
        np.sum(outlier_mask)
    )

    added_outliers_detected = int(
        np.sum(outlier_mask[:n_outliers])
    )

    return (
        total_outliers_detected,
        added_outliers_detected
    )


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot regression fits.
    """

    fig, ax = plt.subplots()

    ax.scatter(X, y, label="Data")

    x_line = np.linspace(
        X.min(),
        X.max(),
        200
    ).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(X, y)

    huber = HuberRegressor()
    huber.fit(X, y)

    ransac = RANSACRegressor(
        random_state=random_state
    )
    ransac.fit(X, y)

    theilsen = TheilSenRegressor(
        random_state=random_state
    )
    theilsen.fit(X, y)

    ax.plot(
        x_line,
        lr.predict(x_line),
        label="Linear Regression"
    )

    ax.plot(
        x_line,
        huber.predict(x_line),
        label="Huber Regression"
    )

    ax.plot(
        x_line,
        ransac.predict(x_line),
        label="RANSAC Regression"
    )

    ax.plot(
        x_line,
        theilsen.predict(x_line),
        label="Theil-Sen Regression"
    )

    ax.set_title("Regression Model Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Visualize RANSAC inliers and outliers.
    """

    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots()

    ax.scatter(
        X[inlier_mask],
        y[inlier_mask],
        label="Inliers"
    )

    ax.scatter(
        X[outlier_mask],
        y[outlier_mask],
        label="Outliers"
    )

    ax.set_title(
        "RANSAC Inliers vs Outliers"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig
