from typing import Dict, Any
import sys
import os
from statistics import mean
import pandas as pd


import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
gparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, gparent_dir)
sys.path.append("../utils")
sys.path.append("..")

from Data_handler import Data_handler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

#za algoritem
from src.EAlgoritem import EAlgoritem
from src.Gen import Interval, Mnozica, Urejena_mnozica
from src.Krizanje import Diagonalno_krizanje


horizons = [3, 4, 5, 6, 7, 8, 9, 10]
additional = [1, 2, 3, 4, 5, 6, 7, 10]
features = [
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
target = "level"
K = 10
sensor_name = "85012"


normalize_y = Mnozica('normalize_y', [True, False])
n_restarts_optimizer = Urejena_mnozica( "n_restarts_optimizer", list(range(0, 10)), stdev = 1)
alpha = Interval("alpha", zacetek=1e-13, konec=1e-8, stdev=1e-10)

geni = [normalize_y, n_restarts_optimizer, alpha]

populacija = [{"n_restarts_optimizer": 0, "normalize_y": False, "alpha": 1e-10}]


output_file = open("GP_regressor_results_correct.txt", "w")

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
    
    def scoring_function(hiperparametri: Dict[str, Any]) -> float:
        #zasnovana mora biti tako da se rezultat (vrnjena vrednost) maksimizira
        timeSeriesCV = TimeSeriesSplit(n_splits=10)
        
        # Extract relevant time series
        y_pretrain = data_handler.pretrain_dataframe[f"{target}_target_h{horizon}"].reset_index(drop=True)
        X_pretrain = data_handler.pretrain_dataframe.drop(columns=[f"{target}_target_h{horizon}"]).reset_index(drop=True)
        X = data_handler.totest_dataframe.drop(columns=[f"{target}_target_h{horizon}"]).reset_index(drop=True)
        y = data_handler.totest_dataframe[f"{target}_target_h{horizon}"].reset_index(drop=True)

        r = GaussianProcessRegressor()
        r.set_params(**hiperparametri)

        # cross validation
        scores = []
        for train_index, test_index in timeSeriesCV.split(X):
            # Join with pretrain
            X_train = pd.concat([X_pretrain, X.iloc[train_index]])
            y_train = pd.concat([y_pretrain, y.iloc[train_index]])

            # Train the model
            r.fit(X_train, y_train)

            # Make predictions
            X_test = X.iloc[test_index]
            predictions = r.predict(X_test)
            true = y.iloc[test_index]
            
            score = r2_score(true, predictions)
            scores.append(score)
        
        return mean(scores)

    kriz = Diagonalno_krizanje(stevilo_starsev = 2)
    alg = EAlgoritem(velikost_populacije = 15, omejitev_stevila_ocenjevanj = 55, stevilo_starsev_za_izbiro = 8,
                        stevilo_potomcev = 6, cenilna_funkcija = scoring_function, tipi_genov = geni, krizanje = kriz,
                        zacetna_populacija = populacija, verjetnost_mutacije=1/10)


    alg.pozeni()

    s = ""
    for g in alg.najboljsi.geni[:]:
        s = s + f"{g.tip.ime}: {g.vrednost}, "
    s = s + f"=> ocena: {alg.najboljsi.ocena}"

    output_file.write(f"Horizon: {horizon}: {s} \n")
    output_file.flush()

output_file.close()
