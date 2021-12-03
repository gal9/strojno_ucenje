from abc import ABC, abstractmethod
from typing import List
from random import random, sample
from bisect import bisect
import math
from statistics import mean, stdev

from src.Posameznik import Ocenjen_posameznik

class Selekcija_starsev(ABC):
    stevilo_starsev_za_izbiro: int

    @abstractmethod
    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        pass

class Oceni_proporcionalna_selekcija_starsev(Selekcija_starsev):
    def izberi_posamezno(self, meje: List[float]) -> int:
        nakjucna = random()
        return bisect(meje, nakjucna)

    def doloci_meje(self, populacija: List["Ocenjen_posameznik"]) -> List[int]:
        ocene = [x.ocena for x in populacija]
        vsota_vseh = sum(ocene)
        vsota_do = 0
        meje = []

        for o in ocene:
            vsota_do += o
            meje.append(vsota_do/vsota_vseh)
        
        return meje

    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        meje = self.doloci_meje(populacija)
        starsi = []

        for i in range(self.stevilo_starsev_za_izbiro):
            indeks = self.izberi_posamezno(meje)
            starsi.append(populacija[indeks])
        #Uredi starse glede na oceno
        starsi.sort(key=lambda s: s.ocena, reverse=True)
        return starsi

class Oceni_proporcionalna_selekcija_starsev_brez_ponovitev(Selekcija_starsev):
    def izberi_posamezno(self, meje: List[float]) -> int:
        nakjucna = random()
        return bisect(meje, nakjucna)

    def doloci_meje(self, populacija: List["Ocenjen_posameznik"]) -> List[float]:
        ocene = [x.ocena for x in populacija]
        vsota_vseh = sum(ocene)
        vsota_do = 0
        meje = []

        for o in ocene:
            vsota_do += o
            meje.append(vsota_do/vsota_vseh)
        
        return meje

    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        kopija = populacija.copy()

        starsi = []

        for i in range(self.stevilo_starsev_za_izbiro):
            if(len(kopija) == 1):
                indeks=0
            else:
                #doloci meje po vsakem dodanem posamezniku
                meje = self.doloci_meje(kopija)
                indeks = self.izberi_posamezno(meje)
            starsi.append(kopija[indeks])
            kopija.pop(indeks)
        #Uredi starse glede na oceno
        starsi.sort(key=lambda s: s.ocena, reverse=True)
        return starsi

class Oceni_proporcionalna_selekcija_starsev_brez_ponovitev_sigma_skalirana(Selekcija_starsev): #str 95 c=2
    #resi problem izgube "selection pressure" in zamaknjenih ocen
    def izberi_posamezno(self, meje: List[float]) -> int:
        nakjucna = random()
        return bisect(meje, nakjucna)

    def doloci_meje(self, populacija: List["Ocenjen_posameznik"]) -> List[float]:
        ocene1 = [x.ocena for x in populacija]
        m = mean(ocene1)
        sigma = stdev(ocene1)

        #enacba v literaturi
        ocene = [max((i-(m-2*sigma), 0)) for i in ocene1]

        vsota_vseh = sum(ocene)
        vsota_do = 0
        meje = []

        for o in ocene:
            vsota_do += o
            meje.append(vsota_do/vsota_vseh)
        
        return meje

    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        kopija = populacija.copy()

        starsi = []

        for i in range(self.stevilo_starsev_za_izbiro):
            if(len(kopija) == 1):
                indeks=0
            else:
                #doloci meje po vsakem dodanem posamezniku
                meje = self.doloci_meje(kopija)
                indeks = self.izberi_posamezno(meje)
            starsi.append(kopija[indeks])
            kopija.pop(indeks)
        #Uredi starse glede na oceno
        starsi.sort(key=lambda s: s.ocena, reverse=True)
        return starsi


class Rangu_lin_proporcionalna_selekcija_starsev_brez_ponovitev(Selekcija_starsev): #stran 96 s=1.5
    def izberi_posamezno(self, meje: List[int]) -> int:
        nakjucna = random()
        return bisect(meje, nakjucna)

    def doloci_meje(self, populacija: List["Ocenjen_posameznik"]) -> List[float]:
        #enacba opisana v literaturi
        ocene = [((0.5/len(populacija)) + ((len(populacija)-i)/(len(populacija)*(len(populacija)-1)))) for i in range(len(populacija))]
        vsota_vseh = sum(ocene)
        vsota_do = 0
        meje = []

        for o in ocene:
            vsota_do += o
            meje.append(vsota_do/vsota_vseh)
        
        return meje

    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        kopija = populacija.copy()

        starsi = []

        for i in range(self.stevilo_starsev_za_izbiro):
            if(len(kopija) == 1):
                indeks=0
            else:
                #doloci meje po vsakem dodanem posamezniku
                meje = self.doloci_meje(kopija)
                indeks = self.izberi_posamezno(meje)
            starsi.append(kopija[indeks])
            kopija.pop(indeks)
        #Uredi starse glede na oceno
        starsi.sort(key=lambda s: s.ocena, reverse=True)
        return starsi


class Rangu_exp_proporcionalna_selekcija_starsev_brez_ponovitev(Selekcija_starsev): #stran 96
    def izberi_posamezno(self, meje: List[int]) -> int:
        nakjucna = random()
        return bisect(meje, nakjucna)

    def doloci_meje(self, populacija: List["Ocenjen_posameznik"]) -> List[float]:
        #enacba opisana v literaturi
        ocene = [(1-math.exp(-(len(populacija)-(i+1)))) for i in range(len(populacija))]
        vsota_vseh = sum(ocene)
        vsota_do = 0
        meje = []

        for o in ocene:
            vsota_do += o
            meje.append(vsota_do/vsota_vseh)
        
        return meje

    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        kopija = populacija.copy()

        starsi = []

        for i in range(self.stevilo_starsev_za_izbiro):
            if(len(kopija) == 1):
                indeks=0
            else:
                #doloci meje po vsakem dodanem posamezniku
                meje = self.doloci_meje(kopija)
                indeks = self.izberi_posamezno(meje)
            starsi.append(kopija[indeks])
            kopija.pop(indeks)
        #Uredi starse glede na oceno
        starsi.sort(key=lambda s: s.ocena, reverse=True)
        return starsi

class Nakljucna_selekcija_starsev(Selekcija_starsev):
    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        starsi = sample(populacija, self.stevilo_starsev_za_izbiro)
        #Uredi starse glede na oceno
        starsi.sort(key=lambda s: s.ocena, reverse=True)
        return starsi

class Selekcija_najboljsih_starsev(Selekcija_starsev):
    def izberi(self, populacija: List["Ocenjen_posameznik"]) -> List["Ocenjen_posameznik"]:
        return populacija[:self.stevilo_starsev_za_izbiro]