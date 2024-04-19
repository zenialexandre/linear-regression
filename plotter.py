import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regression_utils import correlation_coefficient, linear_regression
from scipy.interpolate import make_interp_spline

def create_regression_plots(
    matrices: list,
    information_3d: tuple[list, np.ndarray, bool],
    additional_plot_titles: list = None
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
        information_3d,
        additional_plot_titles
    )
    plt.tight_layout()
    plt.show()

def create_scatter_plots(
    figure: plt.figure,
    figure_data: any,
    matrices: list,
    information_3d: tuple[list, np.ndarray, bool],
    additional_plot_titles: list = None
) -> None:
    create_2d_scatter_plots(
        figure_data,
        matrices,
        additional_plot_titles
    )

    if information_3d[2] == True:
        create_3d_scatter_plots(
            figure,
            information_3d[0],
            information_3d[1]
        )

def create_2d_scatter_plots(
    figure_data: any,
    matrices: list,
    additional_plot_titles: list = None
) -> None:
    for index, matrix in enumerate(matrices):
        final_plot_title: str = ''
        matrix_correlation_coefficient: float = correlation_coefficient(matrix)
        (_, matrix_beta_1, matrix_beta_0) = linear_regression(matrix)
        (best_fit_a, best_fit_b) = np.polyfit(matrix[0], matrix[1], 1)

        if (additional_plot_titles != None):
            final_plot_title = \
                f'{additional_plot_titles[index]} \n {matrix_beta_1}x + {matrix_beta_0} | r = {matrix_correlation_coefficient}'
        else:
            final_plot_title = f' {matrix_beta_1}x + {matrix_beta_0} | r = {matrix_correlation_coefficient}'

        figure_data[index].scatter(x=matrix[0], y=matrix[1], color='red')
        figure_data[index].plot(matrix[0], best_fit_a * matrix[0] + best_fit_b)
        figure_data[index].set_title(final_plot_title)

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


def create_2d_scatter_plots_poly_reg(dataframe: pd.DataFrame, title: str, polyfit_values: list = None):
    (figure, figure_data) = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(18, 5)
    )
    figure.canvas.manager.set_window_title('Dataset Analysis (Linear Regression)')

    spline = make_interp_spline(dataframe[0], polyfit_values[0])
    X_ = np.linspace(dataframe[0].min(), dataframe[0].max(), 300)
    Y_ = spline(X_)

    figure_data.scatter(x=dataframe[0], y=dataframe[1], color="blue")
    figure_data.plot(X_, Y_)
    figure_data.set_title(title)
    plt.tight_layout()
    plt.show()
