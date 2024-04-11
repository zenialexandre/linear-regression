import matplotlib.pyplot as plt
import numpy as np
from regression_utils import correlation_coefficient, linear_regression

def create_regression_plots(
    matrices: list
) -> None:
    (figure, figure_data) = plt.subplots(nrows=1, ncols=len(matrices), figsize=(15, 5))
    figure.canvas.manager.set_window_title('Dataset Analysis (Linear Regression)')

    create_scatter_plots(figure_data, matrices)
    plt.tight_layout()
    plt.show()

def create_scatter_plots(
    figure_data: plt.figure, 
    matrices: list
) -> None:
    for index, matrix in enumerate(matrices):
        matrix_correlation_coefficient: float = correlation_coefficient(matrix)
        (_, matrix_beta_1, matrix_beta_0) = linear_regression(matrix)
        (best_fit_a, best_fit_b) = np.polyfit(matrix[0], matrix[1], 1)

        figure_data[index].scatter(x=matrix[0], y=matrix[1], color='red')
        figure_data[index].plot(matrix[0], best_fit_a * matrix[0] + best_fit_b)
        figure_data[index].set_title(
            f'y = {matrix_beta_1}x + {matrix_beta_0} | r = {matrix_correlation_coefficient}'
        )
