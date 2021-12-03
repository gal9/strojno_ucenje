from abc import ABC, abstractmethod
from typing import Any, Iterable, List
from fbprophet import Prophet

from numpy.core.fromnumeric import mean

class TS_model(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self, data: Any) -> None:
        pass

    @abstractmethod
    def _predict(self, sample: Any) -> Any:
        pass

    def predict(self, stream: Iterable) -> List[float]:
        results = []
        for sample in stream:
            results.append(self._predict(sample))
        
        return results

class Decaying_average(TS_model):

    mean: float
    decay_factor: float

    def __init__(self, decay_factor: float = 0.7) -> None:
        self.decay_factor = decay_factor
        super().__init__()
    
    def train(self, data: Any) -> None:
        for sample in data:
            self._predict(sample)

    def _predict(self, sample: float) -> Any:
        if(not hasattr(self, "mean")):
            self.mean = sample
        else:
            self.mean = self.mean * self.decay_factor + (1 - self.decay_factor) * sample
        return self.mean

class Live_prophet(TS_model):
    
    model: Any
    data: Any
    horizon: List[int]
    regressors: List[str]

    def __init__(self, horizon=1, regressors: List[str]="") -> None:
        self.regressors = regressors
        self.horizon = horizon
        super().__init__()

    def train(self, data:Any) -> None:
        self.model = Prophet()
        for r in self.regressors:
            self.model.add_regressor(r)
        self.model.fit(data)
        self.data=data

    def predict(self, dataframe: Any) -> None:
        forecast = self.model.predict(dataframe)
        return forecast

    def _predict(self, sample: Any) -> Any:
        self.data.append(sample)
        sample.drop(columns=[{f"level_target_h{self.horizon}": "y"}])
        future = self.model.make_future_dataframe(periods=self.horizon)
        print(future)
        forecast = self.model.predict(future)

        returned = []

        returned.append(forecast["yhat"].tolist())

        print(returned)

        self.train(self.data)

        return returned