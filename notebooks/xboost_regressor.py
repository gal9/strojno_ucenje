import sys
sys.path.append("../utils")

from Data_handler import Data_handler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

horizon = 1
additional = [1, 2, 3, 4, 5, 6, 7, 10]
features = {
    "85012": [
        'level', 'temperature_avg', 'level_shift_6_average_6',
       'level_shift_20_average_7', 'snow_accumulation_shift_6_average_4',
       'temperature_avg_shift_20_average_1',
       'temperature_min_shift_2_average_7',
       'temperature_max_shift_1_average_7'],
    "85012": [
       'level', 'temperature_avg', 'level_shift_10', 'level_shift_6_average_6',
       'level_shift_20_average_7', 'temperature_avg_shift_20_average_1',
       'temperature_min_shift_2_average_7',
       'temperature_max_shift_1_average_5',
       'temperature_max_shift_1_average_7'],
    "85012": [
        'level', 'cloud_cover', 'temperature_avg', 'level_shift_10',
       'level_shift_6_average_6', 'level_shift_20_average_7',
       'precipitation_shift_5_average_10',
       'temperature_avg_shift_20_average_1',
       'temperature_min_shift_1_average_10',
       'temperature_min_shift_4_average_10',
       'temperature_max_shift_1_average_7'],
    "85012": [
       'level', 'temperature_avg_average_1', 'level_shift_4_average_6',
       'temperature_min_shift_3_average_7',
       'temperature_max_shift_1_average_7'],
    "85012": [
       'level', 
       'temperature_avg_average_1',
       'level_shift_4_average_6',
       'temperature_max_shift_1_average_7',
       'precipitation',
       'precipitation_average_3',
       'snow_accumulation_shift_6_average_4',
       'precipitation_shift_5_average_10',
       'level_average_5',
       'cloud_cover',
       'sun_duration',
       ]
    }

target = "level"
K = 10

data_handlers = {}

for stream in features:
    data = Data_handler(f"../data/ground/{stream}.csv")
    data.construct_time_of_year()
    data.construct_features(averages=additional, shifts=additional, skip=["level"], horizon=horizon)

    # Must be after feature construction
    data.target_value_construction(horizon=horizon, target=target)
    data.select_features(features[stream], target_column=f"{target}_target_h{horizon}")

    # Save data handler
    data_handlers[stream] = data


for data_handler_name in data_handlers:
    data_handler = data_handlers[data_handler_name]
    data_handler.show()

    #splits data considering it is a time series
    timeSeriesCV = TimeSeriesSplit(n_splits=3)
    model=GradientBoostingRegressor(n_estimators=100, loss="squared_error", learning_rate=0.1, subsample=1.0, criterion="friedman_mse", min_samples_split=2, max_depth=3)

    y = data_handler.dataframe[f"{target}_target_h{horizon}"]
    X = data_handler.dataframe.drop(columns=[f"{target}_target_h{horizon}"])

    cvs=cross_val_score(model, X, y, scoring='r2', cv=timeSeriesCV, n_jobs=-1)
    print("CV scores of " + data_handler_name)
    print("R2: ", cvs.mean(), " stdev: ", cvs.std())
    print(cvs)
