import pandas as pd
from typing import List

import math
import numpy as np
from typing import Any
from IPython.display import display
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

class Data_handler:
    dataframe: Any
    pretrain_dataframe: Any
    totest_dataframe: Any

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

    def shift_features(self, shifts: List[int], skip: List[str], horizon: int) -> None:
        # Loop over all features
        for feature in list(self.dataframe.columns):
            # Loop over all shifts
            for shift in shifts:
                #if(feature in skip): #"""and shift<horizon):
                #   continue
                self.dataframe[f"{feature}_shift_{shift}"] = self.dataframe[feature].shift(shift)
    
    def average_features(self, averages: List[int], skip: List[str]) -> None:
        # Loop over all features
        for feature in list(self.dataframe.columns):
            # Loop over all averages
            for average in averages:
                #if(feature in skip):
                #    continue
                self.dataframe[f"{feature}_average_{average}"] = self.dataframe[feature].rolling(average).sum()/average

    def construct_time_of_year(self) -> None:
        self.dataframe.index = pd.to_datetime(self.dataframe.index)

        self.dataframe["month_normalized"] = 2 * math.pi * (self.dataframe.index.month-1) / 11

        self.dataframe["month_cos"] = np.cos(self.dataframe["month_normalized"])
        self.dataframe["month_sin"] = np.sin(self.dataframe["month_normalized"])

        self.dataframe = self.dataframe.drop(columns=["month_normalized"])

    def construct_features(self, averages: List[int], shifts: List[int], skip: List[str], horizon: int) -> None:
        self.shift_features(shifts=shifts, skip=skip, horizon=horizon)
        self.average_features(averages=averages, skip=skip)

        cut = shifts[-1] + averages[-1] - 1
        self.dataframe = self.dataframe.iloc[cut:, :]

    def select_features(self, features_names: List[str], target_column: str = None) -> None:
        features = self.dataframe[features_names]
        if(target_column is not None):
            features[target_column] = self.dataframe[target_column]
        
        self.dataframe = features

    def target_value_construction(self, horizon: int, target: str) -> None:
        # Construct targert variable
        self.dataframe[target + f"_target_h{horizon}"] = self.dataframe[target].shift(-horizon)

        # Cut samples without data
        self.dataframe = self.dataframe.iloc[:-horizon, :]

    def show(self) -> None:
        display(self.dataframe)

    def select_k_best_features(self, k: int, target_column: str, preselected: List[str]) -> None:
        # Create and fit selector
        selector = SelectKBest(f_regression, k=k)
        selector = selector.fit(self.dataframe.drop(columns=[target_column]), self.dataframe[target_column])

        column_names = preselected + list(set(selector.get_feature_names_out()) - set(preselected))

        print("selected features: ")
        #print(selector.get_feature_names_out())
        print(column_names)

        # get column indices and only keep those
        #features = self.dataframe.drop(columns=[target_column]).iloc[:, selector.get_support(indices=True)]
        features = self.dataframe[column_names]
        features[target_column] = self.dataframe[target_column]
        self.dataframe = features
        
        #print(self.dataframe.head(15))

    def pretrain_test_split(self, split_index: int) -> None:
        self.pretrain_dataframe = self.dataframe.iloc[:split_index, :]
        self.totest_dataframe = self.dataframe.iloc[split_index:, :]

    def graph(self, column: str) -> None:
        #plt.plot(self.dataframe.index, self.dataframe[column])
        #plt.show()
        self.dataframe.plot(y=column, use_index=True)
        plt.show()