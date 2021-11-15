import pandas as pd
from typing import List

import pandas as pd
from typing import Any
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

class Data_handler:
    dataframe: Any

    def __init__(self, path: str) -> None:
        self.load_data_csv(path)

    def load_data_csv(self, path: str) -> None:
        # Load data from csv
        self.dataframe = pd.read_csv(path, index_col='date')

    def calculate_level_diff(self) -> None:
        # Calculate level change and insert new column
        diff = self.dataframe['level'] - self.dataframe['level'].shift(1)
        self.dataframe.insert(1, 'level_diff', diff)
        
        # remove the first line
        self.dataframe = self.dataframe[1:]

    def shift_features(self, max_shift: int, skip: List[str], horizon: int) -> None:
        # Loop over all features
        for feature in list(self.dataframe.columns):
            # Loop over all shifts
            for shift in range(1, max_shift+1):
                if(feature in skip and shift<horizon):
                    continue
                self.dataframe[f"{feature}_shift_{shift}"] = self.dataframe[feature].shift(shift)
    
    def average_features(self, max_average: int, skip: List[str]) -> None:
        # Loop over all features
        for feature in list(self.dataframe.columns):
            # Loop over all averages
            for average in range(2, max_average+1):
                if(feature in skip):
                    continue
                self.dataframe[f"{feature}_average_{average}"] = self.dataframe[feature].rolling(average).sum()/average

    def construct_features(self, max_average: int, max_shift: int, skip: List[str], horizon: int) -> None:
        self.shift_features(max_shift=max_shift, skip=skip, horizon=horizon)
        self.average_features(max_average=max_average, skip=skip)

        cut = max_shift + max_average -1
        self.dataframe = self.dataframe.iloc[cut:, :]

    def target_value_construction(self, horizons: List[int]) -> None:
        # Construct targert variables for all horizons
        for horizon in horizons:
            self.dataframe[f"level_diff_target_h{horizon}"] = self.dataframe["level_diff"].shift(-horizon)

        # Cut samples without data
        self.dataframe = self.dataframe.iloc[:-max(horizons), :]

    def show(self) -> None:
        display(self.dataframe)

    def select_k_best_features(self, k: int, target_column: str) -> None:
        # Create and fit selector
        selector = SelectKBest(f_regression, k=k)
        selector.fit(self.dataframe.drop(columns=["level", "level_diff", target_column]), self.dataframe[target_column])

        print(selector.get_feature_names_out())

        # get column indices and only keep those
        features = self.dataframe.drop(columns=["level", "level_diff", target_column]).iloc[:, selector.get_support(indices=True)]
        features[target_column] = self.dataframe[target_column]
        self.dataframe = features


    def graph(self, column: str) -> None:
        #plt.plot(self.dataframe.index, self.dataframe[column])
        #plt.show()
        self.dataframe.plot(y=column, use_index=True)
        plt.show()