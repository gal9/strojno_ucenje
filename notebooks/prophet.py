import sys
sys.path.append("../utils")
from Models import Live_prophet
import time
import pandas as pd
import numpy as np
import os


from Data_handler import Data_handler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from sklearn.metrics import r2_score
from statistics import stdev, mean

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

horizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
additional = [1, 2, 3, 4, 5, 6, 7, 10]
features = [
       "level", 
       "temperature_avg_average_1",
       "level_shift_4_average_6",
       "temperature_max_shift_1_average_7",
       "precipitation",
       "precipitation_average_3",
       "snow_accumulation_shift_6_average_4",
       "precipitation_shift_5_average_10",
       "level_average_5",
       "cloud_cover",
       "sun_duration",
       ]
features = []
features = [
    "level",
    "precipitation",
    "cloud_cover",
    "sun_duration",
    "snow_accumulation",
    "snow_depth",
    "temperature_avg",
    "temperature_min",
    "temperature_max"
]

target = "level"
K = 10
sensor_name = "85012"

output_file = open("prophet_2_results.txt", "w")

for horizon in horizons:
    print(f"Horizon {horizon}")
    data_handler = Data_handler(f"../data/ground/{sensor_name}.csv")
    
    data_handler.construct_time_of_year()
    data_handler.construct_features(averages=additional, shifts=additional, skip=["level"], horizon=horizon)

    # Must be after feature construction
    data_handler.target_value_construction(horizon=horizon, target=target)
    data_handler.select_features(features, target_column=f"{target}_target_h{horizon}")

    #data_handler.show()

    # Set pretrain interval to 1000 samples (~1/3 of all)
    data_handler.pretrain_test_split(800)

    #splits data considering it is a time series
    timeSeriesCV = TimeSeriesSplit(n_splits=3)

    model = Live_prophet(horizon=horizon, regressors=features)

    # Prepare pretrain data
    X_pretrain = data_handler.pretrain_dataframe.reset_index().rename(columns={"date": "ds"})
    X_pretrain = X_pretrain.rename(columns={f"{target}_target_h{horizon}": "y"})

    # Prepare the data for time series CV
    X = data_handler.totest_dataframe.reset_index().rename(columns={"date": "ds"})
    X = X.rename(columns={f"{target}_target_h{horizon}": "y"})

    # Prepre target variables for scoring
    y = data_handler.totest_dataframe[f"{target}_target_h{horizon}"].reset_index(drop=True)

    # cross validation
    scores = []
    for train_index, test_index in timeSeriesCV.split(X):
        # Train the model
        X_train = pd.concat([X_pretrain, X.iloc[train_index]])
        #start_time = time.time()
        model.train(X_train)
        #print("training time: " + time.time()-start_time)

        # Make predictions
        X_test = X.iloc[test_index].drop(columns=["y"])
        
        predictions = model.predict(X_test)

        
        true = y.iloc[test_index].values

        #print(predictions)
        #print(type(predictions))

        predictions = np.array([row["yhat"] for i, row in predictions.iterrows()])
        
        #print(true)
        #print(predictions)
        score = r2_score(true, predictions)
        scores.append(score)

    output_file.write(f"Horizon {horizon} => socre: {mean(scores)}\n")
    output_file.flush()