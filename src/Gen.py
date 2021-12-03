import random

from abc import ABC, abstractmethod
from typing import List, Any, Optional

class Gen_tip(ABC):
    ime: str

    def __init__(self, ime: str):
        self.ime = ime
    
    @abstractmethod
    def nakljucna(self):
        pass

class Interval(Gen_tip):
    zacetek: float
    konec: float
    stdev: float

    def __init__(self, ime: str, zacetek: float, konec: float, stdev: float = 0.1) -> None:
        super().__init__(ime)
        self.zacetek = zacetek
        self.konec = konec
        self.stdev = stdev

    def nakljucna(self) -> float:
        return random.uniform(self.zacetek, self.konec)


class Mnozica(Gen_tip):
    vrednosti: List[Any]

    def __init__(self, ime: str, vrednosti: List[Any]):
        super().__init__(ime)
        self.vrednosti = vrednosti

    def nakljucna(self) -> Any:
        return random.choice(self.vrednosti)

class Urejena_mnozica(Gen_tip):
    vrednosti: List[Any]
    stdev: float

    def __init__(self, ime:str, vrednosti: List[Any], stdev: float = 1):
        super().__init__(ime)
        self.vrednosti = vrednosti
        self.stdev = stdev
    
    def nakljucna(self) -> Any:
        return random.choice(self.vrednosti)

class Gen():
    vrednost: Any
    tip: Gen_tip
    indeks: int

    def __init__(self, tip: Gen_tip, vrednost: Optional[Any] = None):
        self.tip = tip
        if (vrednost == None):
            self.vrednost=self.tip.nakljucna()
        else:
            self.vrednost = vrednost
        
        #ce gre za mnozico ali urejeno mnozico nastavi se indeks
        if (isinstance(self.tip, Mnozica) or isinstance(self.tip, Urejena_mnozica)):
            self.indeks = self.tip.vrednosti.index(self.vrednost)
