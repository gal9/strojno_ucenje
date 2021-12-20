import sys

from numpy.core.numeric import Inf
sys.path.append("../utils")

from Data_handler import Data_handler
from Models import Decaying_average
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from statistics import stdev, mean
import matplotlib.pyplot as plt
import numpy as np

horizons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
additional = []
sensor_name = "85012"
features = ["level"]
target = "level"
K = 10

output_file = open("decaying_average_results.txt", "w")
horizon_scores = []

for horizon in horizons:
    print(f"Horizon {horizon}")
    data_handler = Data_handler(f"../data/ground/{sensor_name}.csv")

    data_handler.target_value_construction(horizon=horizon, target=target)

    data_handler.select_features(features, target_column=f"{target}_target_h{horizon}")

    #data_handler.show()

    # Set pretrain interval to 1000 samples (~1/3 of all)
    data_handler.pretrain_test_split(800)

    #splits data considering it is a time series
    timeSeriesCV = TimeSeriesSplit(n_splits=10)
    decay_factor_scores = []
    decay_factor_means = []
    max_score = -Inf
    best_decay = -1
    for decay_factor in list(np.arange(0, 1, 0.01)):
        #print(f"Decay factor: {decay_factor}")

        model = Decaying_average(decay_factor=decay_factor)

        # Extract relevant time series
        X_pretrain = data_handler.pretrain_dataframe[target].values
        X = data_handler.totest_dataframe[target].values
        y = data_handler.totest_dataframe[f"{target}_target_h{horizon}"].values

        # cross validation
        scores = []
        for train_index, test_index in timeSeriesCV.split(X):
            # Join with pretrain
            X_train = np.concatenate((X_pretrain, X[train_index]), axis=0)

            # Train the model
            model.train(X_train)

            # Make predictions
            X_test = X[test_index]
            predictions = model.predict(X_test)
            true = y[test_index]
            
            score = r2_score(true, predictions)
            scores.append(score)

        decay_factor_scores.append(scores)
        decay_factor_means.append(mean(scores))
        if(mean(scores)>max_score):
            max_score = mean(scores)
            best_decay = decay_factor
        #print("CV scores of " + data_handler_name)
        #print("R2: ", mean(scores), " stdev: ", stdev(scores))
    #print(decay_factor_means)
    output_file.write(f"Horizon {horizon} => score: {max_score}, decay_rate: {best_decay}\n")
    print(f"Best decay rate: {best_decay} with score: {max_score}")
    horizon_scores.append(max_score)
    """plt.plot(list(np.arange(0, 1, 0.01)), decay_factor_means)
    plt.plot([best_decay], [max_score], 'ro', label="Best decay rate")
    plt.title(f"Best decay factor for horizon {horizon}")
    plt.xlabel("decay rate")
    plt.ylabel("R2 score")
    plt.legend()
    plt.show()"""

print(horizon_scores)

plt.plot(list(range(1, 11)), horizon_scores)
plt.title(f"Baseline score for horizon")
plt.xlabel("Horizon")
plt.ylabel("Best R2 score")
plt.show()
