from Data_handler import Data_handler

data = Data_handler("../data/ground/85012.csv")
data.calculate_level_diff()
data.construct_features(max_average=20, max_shift=20, skip=["level", "level_diff"], horizon=3)
data.target_value_construction(horizons=[3])
data.select_k_best_features(20, "level_diff_target_h3")
data.show()