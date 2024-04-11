import matplotlib.pyplot as plt
from regression_utils import correlation_coefficient, linear_regression
import numpy as np

def create_plots(
        matrix_list: list
) -> None:
    (figure, figure_data) = plt.subplots(nrows=1, ncols=len(matrix_list), figsize=(15, 5))
    figure.canvas.manager.set_window_title('Dataset Analysis (Regression)')

    create_generic_dataset_matrix_scatter_plots(figure_data, matrix_list)
    plt.tight_layout()
    plt.show()

def create_generic_dataset_matrix_scatter_plots(
        figure_data: any, 
        matrix_list: list
) -> None:
    for index, matrix in enumerate(matrix_list):
        matrix_correlation_coefficient = correlation_coefficient(matrix)
        matrix_linear_regression = linear_regression(matrix)
        (best_fit_a, best_fit_b) =  np.polyfit(matrix[0], matrix[1], 1)
        print(matrix[0])

        figure_data[index].scatter(x=matrix[0], y=matrix[1], color='red')
        figure_data[index].plot(matrix[0], best_fit_a * matrix[0] + best_fit_b)
        figure_data[index].set_title(
            f'y = {matrix_linear_regression[1]}x + {matrix_linear_regression[2]} | r = {matrix_correlation_coefficient}'
        )
