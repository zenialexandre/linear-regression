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

def create_dataset_matrix_scatter_plots(
    first_dataset_matrix: list,
    second_dataset_matrix: list,
    third_dataset_matrix: list
) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.canvas.manager.set_window_title('Scatter Plots of the Datasets')

    ax[0].scatter(x=first_dataset_matrix[0], y=first_dataset_matrix[1], color='red')
    ax[0].set_title('Scatter Plot of the First Dataset')
    
    ax[1].scatter(x=second_dataset_matrix[0], y=second_dataset_matrix[1], color='blue')
    ax[1].set_title('Scatter Plot of the Second Dataset')
    
    ax[2].scatter(x=third_dataset_matrix[0], y=third_dataset_matrix[1], color='green')
    ax[2].set_title('Scatter Plot of the Third Dataset')

    plt.tight_layout()
    plt.show()

create_dataset_matrix_scatter_plots(
    first_dataset_matrix,
    second_dataset_matrix,
    third_dataset_matrix
)
