import sys
sys.path.append("../utils")

from Data_handler import Data_handler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from typing import Any, Optional, List
from fastener.random_utils import shuffle
from fastener.item import Result, Genes, RandomFlipMutationStrategy, RandomEveryoneWithEveryone, \
    IntersectionMating, UnionMating, IntersectionMatingWithInformationGain, \
    IntersectionMatingWithWeightedRandomInformationGain
from fastener import fastener

horizon = 5
additional = [1, 2, 3, 4, 5, 6, 7, 10]
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
K = 10
sensors = ["85012"]

for sensor in sensors:
    data = Data_handler(f"../data/ground/{sensor}.csv")
    # Feature construction
    data.construct_features(averages=additional, shifts=additional, skip=["level"], horizon=horizon)
    data.construct_time_of_year()
    data.show()
    # Target construction
    # Must be after feature construction
    data.target_value_construction(horizon=horizon, target=target)

    df = data.dataframe

    general_model = LinearRegression

    #sets number of samples and number of semples used for testing
    n_sample=df.shape[0]
    n_test=int(n_sample*0.8)

    #le = preprocessing.LabelEncoder()
    #le.fit(df['h1'].values.astype(float))

    labels_train=df[f"{target}_target_h{horizon}"].values.astype(float)[:n_test]
    labels_test=df[f"{target}_target_h{horizon}"].values.astype(float)[n_test:]

    df = df.drop(columns=[f"{target}_target_h{horizon}"])

    XX_train=df.to_numpy()[:n_test, :]
    XX_test=df.to_numpy()[n_test:, :]


    def eval_fun(model: Any, genes: "Genes", shuffle_indices: Optional[List[int]] = None) -> "Result":
        test_data = XX_test[:, genes]
        if shuffle_indices:
            test_data = test_data.copy()
            for j in shuffle_indices:
                shuffle(test_data[:, j])
        pred = model.predict(test_data)
        res = Result(r2_score(labels_test, pred))
        return res

    number_of_genes = XX_train.shape[1]

    # 2d array of indices of genes (features)
    initial_genes = [
        [0, 1, 2, 3, 4, 5, 6]
    ]
    # Select mating strategies
    mating = RandomEveryoneWithEveryone(pool_size=5, mating_strategy=IntersectionMatingWithWeightedRandomInformationGain(regression=True))

    # Random mutation
    mutation = RandomFlipMutationStrategy(1 / number_of_genes)

    entropy_optimizer = fastener.EntropyOptimizer(
        general_model, XX_train, labels_train, eval_fun,
        number_of_genes, mating, mutation, initial_genes=initial_genes,
        config=fastener.Config(output_folder=sensor, random_seed=2020, reset_to_pareto_rounds=5, number_of_rounds=2000)
    )

    entropy_optimizer.mainloop()

    object = pd.read_pickle(f'log/{sensor}/generation_1000.pickle')
    values=object.front.values()
    m=0
    for v in values:
        if v.result.score>m:
            m=v.result.score
            win=v
    feat=df
    selected_features_h1 = feat.iloc[:, win.genes].columns
    print(selected_features_h1)