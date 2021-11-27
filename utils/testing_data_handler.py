from Data_handler import Data_handler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

horizon = 1
additional = [1, 2, 3, 4, 5, 6, 7, 10, 20]
preselected = [
    #"level",
    "precipitation",
    "precipitation_average_3",
    "precipitation_average_7",
    "precipitation_shift_1",    
    "precipitation_shift_2",
    "snow_accumulation",
    "snow_depth",
    "temperature_avg",
    "sun_duration",
    "month_cos",
    "month_sin"
]
target = "level"

data = Data_handler("../data/ground/85012.csv")
data.construct_time_of_year()
#data.calculate_level_diff()
data.construct_features(averages=additional, shifts=additional, skip=["level"], horizon=horizon)

# Must be after feature construction
data.target_value_construction(horizon=horizon, target=target)
#data.show()
#data.show()
data.select_k_best_features(10, target + f"_target_h{horizon}", preselected=preselected)
data.show()

timeSeriesCV=TimeSeriesSplit(n_splits=4)
model=GradientBoostingRegressor(n_estimators=300)

y = data.dataframe[target + f"_target_h{horizon}"]
X = data.dataframe.drop(columns=[target + f"_target_h{horizon}"])

cvs=cross_val_score(model, X, y, scoring='r2', cv=timeSeriesCV)
print("R2: ", cvs.mean(), " stdev: ", cvs.std())