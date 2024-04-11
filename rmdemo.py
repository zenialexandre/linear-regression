import pandas as pd
import numpy as np
from plotter import create_regression_plots
from regression_utils import multiple_linear_regression

# dataset_dataframe = sizes, rooms, prices

dataset_dataframe: pd.DataFrame = pd.read_csv(r"content/data.csv", header=None)

def process_dataset_dataframe_analysis(
    dataset_dataframe: pd.DataFrame
) -> None:
    initial_dataset_dataframe_analysis(dataset_dataframe)

    # Matrix of the size and the number of rooms (independet variables)
    _ = dataset_dataframe.iloc[:, 0:2].values
    '''
    [
        [2104    3]
        [1600    3]
        [2400    3]
        [1416    2]
        [3000    4]
        [1985    4]
        [1534    3]
        [1427    3]
        [1380    3]
        [1494    3]
        [1940    4]
        [2000    3]
        [1890    3]
        [4478    5]
        [1268    3]
        [2300    4]
        [1320    2]
        [1236    3]
        [2609    4]
        [3031    4]
        [1767    3]
        [1888    2]
        [1604    3]
        [1962    4]
        [3890    3]
        [1100    3]
        [1458    3]
        [2526    3]
        [2200    3]
        [2637    3]
        [1839    2]
        [1000    1]
        [2040    4]
        [3137    3]
        [1811    4]
        [1437    3]
        [1239    3]
        [2132    4]
        [4215    4]
        [2162    4]
        [1664    2]
        [2238    3]
        [2567    4]
        [1200    3]
        [ 852    2]
        [1852    4]
        [1203    3]
    ]
    '''

    # Vector of the prices (dependent variable)
    _ = dataset_dataframe.iloc[:, 2].values
    '''
    [
        399900. 329900. 369000. 232000. 539900. 299900. 314900. 199000. 212000.
        242500. 240000. 347000. 330000. 699900. 259900. 449900. 299900. 199900.
        500000. 599000. 252900. 255000. 242900. 259900. 573900. 249900. 464500.
        469000. 475000. 299900. 349900. 169900. 314900. 579900. 285900. 249900.
        229900. 345000. 549000. 287000. 368500. 329900. 314000. 299000. 179900.
        299900. 239500.
    ]
    '''

    show_corr_and_linear_reg_plots(dataset_dataframe)
    calculate_multiple_linear_regression(dataset_dataframe)

def initial_dataset_dataframe_analysis(
    dataset_dataframe: pd.DataFrame
) -> None:
    # Describing the dataset for analysis
    pd.DataFrame.describe(dataset_dataframe)

    # Getting the mean of the prices
    dataset_dataframe_prices: pd.Series = dataset_dataframe[2]
    _ = round(dataset_dataframe_prices.mean(), 2)
    # Result: 340412.77

    # Getting the price of the smaller house
    dataset_dataframe_sizes: pd.Series = dataset_dataframe[0]
    _ = dataset_dataframe[dataset_dataframe_sizes == dataset_dataframe_sizes.min()][2]
    # Result: 179900.0

    # Getting the number of rooms of the most expensive house
    _ = dataset_dataframe[dataset_dataframe_prices == dataset_dataframe_prices.max()][1]
    # Result: 5

def show_corr_and_linear_reg_plots(
    dataset_dataframe: pd.DataFrame
) -> None:
    # Getting the Prices/Sizes and the Prices/Rooms matrices and creating it's scatter plots
    prices_and_sizes_matrix: pd.Series = [dataset_dataframe[0].values.tolist(), dataset_dataframe[2].values.tolist()]
    prices_and_rooms_matrix: pd.Series = [dataset_dataframe[1].values.tolist(), dataset_dataframe[2].values.tolist()]

    create_regression_plots([np.array(prices_and_sizes_matrix), np.array(prices_and_rooms_matrix)])

def calculate_multiple_linear_regression(
    dataset_dataframe: pd.DataFrame    
) -> None:
    # Calculate the Multiple Linear Regression
    dataset_dataframe.insert(loc=0, column="initial", value=1)

    _ = multiple_linear_regression(
        dataset_dataframe.iloc[:, 0:3].values,
        dataset_dataframe.iloc[:, 3].values
    )
    '''
    [
        356283.19500476 286121.03514098 397489.54286126 269244.19378551
        472278.00823128 330979.21406118 276933.13325406 262037.59534648
        255494.69551791 271364.70786805 324714.73550191 341805.28900112
        326492.11918958 669293.41082429 239903.10443707 374830.56397604
        255879.97285908 235448.36412826 417846.65008299 476593.53790544
        309369.21112759 334951.61334047 286677.87767959 327777.36946422
        604913.38849027 216515.71781581 266353.12502064 415030.0828272
        369647.41593119 430482.46327339 328130.2922426  220070.48519115
        338635.79896695 500087.78059856 306756.56363202 263429.70169298
        235865.99603221 351443.17735478 641418.92933144 355619.49639429
        303768.43117879 374937.4200479  411999.80342768 230436.78128085
        190729.39584272 312464.19965268 230854.4131848
    ]
    '''

process_dataset_dataframe_analysis(dataset_dataframe)
