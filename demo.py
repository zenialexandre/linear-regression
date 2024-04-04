import numpy as np
import matplotlib.pyplot as plt
from regression_utils import correlation_coefficient, linear_regression

first_dataset_matrix: list = np.array(
    [
        [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    ]
)
second_dataset_matrix: list = np.array(
    [
        [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
        [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    ]
)
third_dataset_matrix: list = np.array(
    [
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19],
        [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]
    ]
)

def create_dataset_matrix_figure(
    first_dataset_matrix: list,
    second_dataset_matrix: list,
    third_dataset_matrix: list
) -> None:
    (figure, figure_data) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    figure.canvas.manager.set_window_title('Dataset Analysis (Linear Regression)')

    create_dataset_matrix_scatter_plots(
        figure_data,
        first_dataset_matrix,
        second_dataset_matrix,
        third_dataset_matrix
    )
    plt.tight_layout()
    plt.show()

def create_dataset_matrix_scatter_plots(
    figure_data: any,
    first_dataset_matrix: list,
    second_dataset_matrix: list,
    third_dataset_matrix: list
) -> None:
    first_dataset_matrix_correlation_coefficient: float = correlation_coefficient(first_dataset_matrix)
    second_dataset_matrix_correlation_coefficient: float = correlation_coefficient (second_dataset_matrix)
    third_dataset_matrix_correlation_coefficient: float = correlation_coefficient(third_dataset_matrix)

    (_, first_dataset_matrix_beta_1, first_dataset_matrix_beta_0) = linear_regression(first_dataset_matrix)
    (_, second_dataset_matrix_beta_1, second_dataset_matrix_beta_0) = linear_regression(second_dataset_matrix)
    (_, third_dataset_matrix_beta_1, third_dataset_matrix_beta_0) = linear_regression(third_dataset_matrix)

    (first_best_fit_a, first_best_fit_b) = np.polyfit(first_dataset_matrix[0], first_dataset_matrix[1], 1)
    figure_data[0].scatter(x=first_dataset_matrix[0], y=first_dataset_matrix[1], color='red')
    figure_data[0].plot(first_dataset_matrix[0], first_best_fit_a * first_dataset_matrix[0] + first_best_fit_b)
    figure_data[0].set_title(
        f'y = {first_dataset_matrix_beta_1}x + {first_dataset_matrix_beta_0} | r = {first_dataset_matrix_correlation_coefficient}'
    )

    (second_best_fit_a, second_best_fit_b) = np.polyfit(second_dataset_matrix[0], second_dataset_matrix[1], 1)
    figure_data[1].scatter(x=second_dataset_matrix[0], y=second_dataset_matrix[1], color='blue', label='Second Dataset')
    figure_data[1].plot(second_dataset_matrix[0], second_best_fit_a * second_dataset_matrix[0] + second_best_fit_b)
    figure_data[1].set_title(
        f'y = {second_dataset_matrix_beta_1}x + {second_dataset_matrix_beta_0} | r = {second_dataset_matrix_correlation_coefficient}'
    )

    (third_best_fit_a, third_best_fit_b) = np.polyfit(third_dataset_matrix[0], third_dataset_matrix[1], 1)
    figure_data[2].scatter(x=third_dataset_matrix[0], y=third_dataset_matrix[1], color='green', label='Third Dataset')
    figure_data[2].plot(third_dataset_matrix[0], third_best_fit_a * third_dataset_matrix[0] + third_best_fit_b)
    figure_data[2].set_title(
        f'y = {third_dataset_matrix_beta_1}x + {third_dataset_matrix_beta_0} | r = {third_dataset_matrix_correlation_coefficient}'
    )

create_dataset_matrix_figure(
    first_dataset_matrix, 
    second_dataset_matrix,
    third_dataset_matrix
)
