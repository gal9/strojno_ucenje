from Data_handler import Data_handler

data=Data_handler()
data.load_data_csv("data/ground/85012.csv")
data.calculate_level_diff()
data.construct_features(max_average=2, max_shift=0, skip=["level", "level_diff"], horizon=3)
data.target_value_construction(horizons=[3])
data.show()
data.graph("level")