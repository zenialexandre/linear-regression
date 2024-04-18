import pandas as pd
import numpy as np
from plotter import create_single_3d_scatter_plots

df = pd.read_csv(r"content/data_preg.csv", header=None)

#create_single_3d_scatter_plots(df)

print(np.polyfit(df.iloc[:,0:1], df.iloc[:,1], 1))