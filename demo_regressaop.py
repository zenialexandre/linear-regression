import pandas as pd
import numpy as np

df = pd.read_csv(r'content/dataset_third_phase.csv', header=None)

#create_single_3d_scatter_plots(df)

print(np.polyfit(df.iloc[:,0:1], df.iloc[:,1], 1))
