import numpy as np
import math

def correlation_coefficient(dataset_matrix: list) -> float:
    (dataset_matrix_x_mean, dataset_matrix_y_mean) = get_dataset_matrix_x_and_y_means(dataset_matrix) 
    correlation_coefficient_numerator: float = 0.0
    correlation_coefficient_denominator_x: float = 0.0
    correlation_coefficient_denominator_y: float = 0.0

    for index in range(len(dataset_matrix[0])):
        correlation_coefficient_numerator += \
            (dataset_matrix[0][index] - dataset_matrix_x_mean) * (dataset_matrix[1][index] - dataset_matrix_y_mean)
        correlation_coefficient_denominator_x += (dataset_matrix[0][index] - dataset_matrix_x_mean) ** 2
        correlation_coefficient_denominator_y += (dataset_matrix[1][index] - dataset_matrix_y_mean) ** 2

    return correlation_coefficient_numerator / math.sqrt(
        correlation_coefficient_denominator_x * correlation_coefficient_denominator_y
    )

def linear_regression(dataset_matrix: list) -> list:
    (dataset_matrix_x_mean, dataset_matrix_y_mean) = get_dataset_matrix_x_and_y_means(dataset_matrix)
    beta_1: float = get_calculated_beta_1(
        dataset_matrix,
        dataset_matrix_x_mean,
        dataset_matrix_y_mean
    )
    beta_0 = get_calculated_beta_0(
        beta_1,
        dataset_matrix_x_mean,
        dataset_matrix_y_mean
    )
    y_hat: list = []

    for cell_x in dataset_matrix[0]: y_hat.append(beta_0 + beta_1 * cell_x)
    return y_hat

def get_dataset_matrix_x_and_y_means(dataset_matrix: list) -> tuple[float, float]:
    return (np.mean(dataset_matrix[0]), np.mean(dataset_matrix[1]))

def get_calculated_beta_1(
    dataset_matrix: list,
    dataset_matrix_x_mean: float,
    dataset_matrix_y_mean: float
) -> float:
    calculus_numerator: float = 0.0
    calculus_denominator: float = 0.0

    for index in range(len(dataset_matrix[0])):
        calculus_numerator += \
            (dataset_matrix[0][index] - dataset_matrix_x_mean) * (dataset_matrix[1][index] - dataset_matrix_y_mean)
        calculus_denominator += (dataset_matrix[0][index] - dataset_matrix_x_mean) ** 2

    return round(calculus_numerator / calculus_denominator, 4)

def get_calculated_beta_0(
    beta_1: float,
    dataset_matrix_x_mean: float,
    dataset_matrix_y_mean: float
) -> float:
    return round(dataset_matrix_y_mean - beta_1 * dataset_matrix_x_mean, 4)
