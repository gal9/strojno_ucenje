import pandas as pd
import sys
sys.path.append("../utils")

from Data_handler import Data_handler

horizon = 5
additional = [1, 2, 3, 4, 5, 6, 7, 10]
sensor = "85012"

data = Data_handler(f"../data/ground/{sensor}.csv")
# Feature construction
data.construct_features(averages=additional, shifts=additional, skip=["level"], horizon=horizon)
data.construct_time_of_year()
df = data.dataframe
object = pd.read_pickle(f'log/{sensor}/generation_1000.pickle')
values=object.front.values()
for v in values:
    print(v.result.score)
    feat=df
    selected_features_h1 = feat.iloc[:, v.genes].columns
    print(selected_features_h1)