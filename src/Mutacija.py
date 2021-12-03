from abc import ABC, abstractmethod
from typing import List
from random import random, gauss
from math import ceil, floor

from src.Gen import Mnozica, Urejena_mnozica, Interval
from src.Posameznik import Posameznik

class Mutacija(ABC):
    #float med 0 in 1
    verjetnost_mutacije: float

    @abstractmethod
    def mutiraj(self, posamezniki: List["Posameznik"]) -> List["Posameznik"]:
        pass

class Mutacija1(Mutacija):
    def mutiraj(self, posamezniki: List["Posameznik"]) -> List["Posameznik"]:
        #iterira cez mnozico posameznikov
        for p in posamezniki:
            for g in p.geni:
                #odloci ali gen mutira
                if (random() < self.verjetnost_mutacije):
                    if (isinstance(g.tip, Mnozica)):
                        g.vrednost = g.tip.nakljucna()
                        g.indeks = g.tip.vrednosti.index(g.vrednost)
                    elif (isinstance(g.tip, Urejena_mnozica)):
                        #vedno zaokrozujemo stran od 0 ker nocemo da se ne spremeni
                        r = gauss(mu=0, sigma=g.tip.stdev)
                        if(r<0):
                            sprememba = floor(r)
                        else:
                            sprememba = ceil(r)
                        
                        #posodobi indeks in preveri ce je se v mejah
                        g.indeks += sprememba
                        if (g.indeks < 0):
                            g.indeks = 0
                        elif (g.indeks >= len(g.tip.vrednosti)):
                            g.indeks = len(g.tip.vrednosti)-1
                        
                        g.vrednost = g.tip.vrednosti[g.indeks]
                    elif (isinstance(g.tip, Interval)):
                        r = gauss(mu=0, sigma=g.tip.stdev)
                        
                        #posodobi vrednost in preveri ce je v mejah
                        g.vrednost += r
                        if (g.vrednost < g.tip.zacetek):
                            g.vrednost = g.tip.zacetek
                        elif (g.vrednost > g.tip.konec):
                            g.vrednost = g.tip.konec
        
        return posamezniki