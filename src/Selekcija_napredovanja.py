from abc import ABC, abstractmethod
from typing import List
from random import random
from bisect import bisect

from src.Posameznik import Ocenjen_posameznik

class Selekcija_napredovanja(ABC):
    velikost_populacije: int

    @abstractmethod
    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        pass

class Selekcija_najboljsih(Selekcija_napredovanja):

    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        return populacija[:self.velikost_populacije]

class Ranku_proporcionalna_selekcija(Selekcija_napredovanja):
    def izberi_posamezno(self, meje: List[float]) -> int:
        nakjucna = random()
        return bisect(meje, nakjucna)

    def doloci_meje(self, populacija: List["Ocenjen_posameznik"]) -> List[float]:
        ocene = [len(populacija)-i for i in range(len(populacija))]
        vsota_vseh = sum(ocene)
        vsota_do = 0
        meje = []

        for o in ocene:
            vsota_do += o
            meje.append(vsota_do/vsota_vseh)
        
        return meje

    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        kopija = populacija.copy()

        #implementacija elitizma -> najboljsi je vedno izbran
        izbrani = [kopija[0]]
        kopija.pop(0)

        for i in range(self.velikost_populacije):
            if(len(kopija) == 1):
                indeks=0
            else:
                #doloci meje po vsakem dodanem posamezniku
                meje = self.doloci_meje(kopija)
                indeks = self.izberi_posamezno(meje)
            izbrani.append(kopija[indeks])
            kopija.pop(indeks)
        #Uredi starse glede na oceno
        izbrani.sort(key=lambda s: s.ocena, reverse=True)
        return izbrani