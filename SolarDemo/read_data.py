import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = "serif"

import os

file_path = os.path.join((os.path.dirname(__file__)),"demo_chart_raw.txt")
data_df = pd.read_csv(file_path, sep="\t")

month = data_df["Month"]
my_yield = data_df["Yield"]

plt.plot(my_yield)
plt.show()