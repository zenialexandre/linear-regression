import matplotlib.pyplot as plt
import numpy as np
from regression_utils import correlation_coefficient, linear_regression

def create_regression_plots(
    matrices: list,
    information_3d: tuple[list, np.ndarray, bool]
) -> None:
    (figure, figure_data) = plt.subplots(
        nrows=1,
        ncols=(lambda: len(matrices) + 1 if information_3d[2] == True else len(matrices))(),
        figsize=(18, 5)
    )
    figure.canvas.manager.set_window_title('Dataset Analysis (Linear Regression)')

    create_scatter_plots(
        figure,
        figure_data,
        matrices,
        information_3d
    )
    plt.tight_layout()
    plt.show()

def create_scatter_plots(
    figure: plt.figure,
    figure_data: any,
    matrices: list,
    information_3d: tuple[list, np.ndarray, bool]
) -> None:
    create_2d_scatter_plots(
        figure_data,
        matrices
    )

    if information_3d[2] == True:
        create_3d_scatter_plots(
            figure,
            information_3d[0],
            information_3d[1]
        )

def create_2d_scatter_plots(
    figure_data: any,
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

def create_3d_scatter_plots(
    figure: plt.figure,
    multiple_linear_regression_result: list,
    information_3d_matrix: np.ndarray
) -> None:
    figure_data_3d = figure.add_subplot(1, 3, 3, projection='3d')

    figure_data_3d.scatter(
        xs=information_3d_matrix[0],
        ys=information_3d_matrix[1],
        zs=information_3d_matrix[2],
        c='green'
    )

    figure_data_3d.plot_trisurf(
        information_3d_matrix[0],
        information_3d_matrix[1],
        multiple_linear_regression_result,
        cmap='viridis'
    )

    figure_data_3d.set_title('Multiple Linear Regression Analysis (Sizes/Rooms/Prices)')
    figure_data_3d.set_xlabel('X1-Sizes')
    figure_data_3d.set_ylabel('X2-Rooms')
    figure_data_3d.set_zlabel('Y-Prices')
