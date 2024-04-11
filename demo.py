import numpy as np
from plotter import create_regression_plots

first_dataset_matrix: np.ndarray = np.array(
    [
        [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    ]
)
second_dataset_matrix: np.ndarray = np.array(
    [
        [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    ]
)
third_dataset_matrix: np.ndarray = np.array(
    [
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19],
        [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]
    ]
)

create_regression_plots(
    [first_dataset_matrix, second_dataset_matrix, third_dataset_matrix],
    (None, None, False)
)

# Most inappropriate Dataset for Regression = third_dataset_matrix
