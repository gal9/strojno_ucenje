from abc import ABC, abstractmethod
from typing import List
import random

from src.Posameznik import Ocenjen_posameznik, Posameznik
from src.Gen import Gen, Interval, Urejena_mnozica, Mnozica

class Krizanje(ABC):
    stevilo_potomcev: int

    @abstractmethod
    def krizaj(self, mnozica_starsev: List["Ocenjen_posemaznik"]) -> List["Posameznik"]:
        pass

class Diagonalno_krizanje(Krizanje):
    stevilo_starsev: int

    def __init__(self, stevilo_starsev: int) -> None:
        self.stevilo_starsev = stevilo_starsev

    def krizaj(self, mnozica_za_izbiro_starsev: List["Ocenjen_posemaznik"]) -> List["Posameznik"]:
        assert self.stevilo_starsev <= len(mnozica_za_izbiro_starsev[0].geni), "Stevilo starsev ne sme presegati stevila genov"

        potomci = []
        stevilo_ustvarjenih = 0

        while (stevilo_ustvarjenih<self.stevilo_potomcev):
            mnozica_starsev = random.sample(mnozica_za_izbiro_starsev, self.stevilo_starsev)

            #nakljucno izbere stevilo_starsev-1 tock za rezanje posameznika
            tocke_rezanja = random.sample(range(1, len(mnozica_starsev[0].geni)), self.stevilo_starsev-1)
            tocke_rezanja.sort()

            """for p in mnozica_starsev:
                p.izpisi()
            print(tocke_rezanja)"""

            #doda zacetek in konec
            tocke_rezanja.insert(0, 0)
            tocke_rezanja.append(len(mnozica_starsev[0].geni))
            
            for i in range(self.stevilo_starsev):
                geni = []
                for pozicija in range(self.stevilo_starsev):
                    #doloci starsa od katereg bo vzel del genoma
                    stars = mnozica_starsev[(i + pozicija)%self.stevilo_starsev]

                    #doloci del starsa ki se bo prekopiral v istolezeci del otroka
                    del_starsa = stars.geni[tocke_rezanja[pozicija]:tocke_rezanja[pozicija+1]]

                    #kopira (ustvari novo kopijo) del starsa v istolezeci del otroka
                    for g in del_starsa:
                        kopija_gena = Gen(vrednost = g.vrednost, tip = g.tip)
                        geni.append(kopija_gena)
                
                potomci.append(Posameznik(geni = geni))
                stevilo_ustvarjenih += 1

        #odrezemo odvecne ustvarjene
        potomci = potomci[:self.stevilo_potomcev]
        """for p in potomci:
            p.izpisi()"""

        return potomci

class N_tockovno_krizanje(Krizanje):
    stevilo_tock_rezanja: int

    def __init__(self, stevilo_tock_rezanja: int) -> None:
        self.stevilo_tock_rezanja = stevilo_tock_rezanja

    def krizaj(self, mnozica_za_izbiro_starsev: List["Ocenjen_posemaznik"]) -> List["Posameznik"]:
        assert self.stevilo_tock_rezanja < len(mnozica_za_izbiro_starsev[0].geni), "Stevilo tock rezanja ne sme presegati stevila genov"

        potomci = []
        stevilo_ustvarjenih = 0

        while (stevilo_ustvarjenih<self.stevilo_potomcev):
            #nakljucno izberemo 2 starsa
            mnozica_starsev = random.sample(mnozica_za_izbiro_starsev, 2)
            stars1 = mnozica_starsev[0]
            stars2 = mnozica_starsev[1]

            #nakljucno izbere stevilo_starsev-1 tock za rezanje posameznika
            tocke_rezanja = random.sample(range(1, len(mnozica_starsev[0].geni)), self.stevilo_tock_rezanja)
            tocke_rezanja.sort()

            """stars1.izpisi()
            stars2.izpisi()
            print(tocke_rezanja)"""

            #doda zacetek in konec
            tocke_rezanja.insert(0, 0)
            tocke_rezanja.append(len(mnozica_starsev[0].geni))
            
            for i in range(2):
                geni = []
                for pozicija in range(self.stevilo_tock_rezanja+1):
                    #doloci starsa od katereg bo vzel del genoma
                    stars = stars1 if (pozicija+i)%2 == 0 else stars2

                    #doloci del starsa ki se bo prekopiral v istolezeci del otroka
                    del_starsa = stars.geni[tocke_rezanja[pozicija]:tocke_rezanja[pozicija+1]]

                    #kopira (ustvari novo kopijo) del starsa v istolezeci del otroka
                    for g in del_starsa:
                        kopija_gena = Gen(vrednost = g.vrednost, tip = g.tip)
                        geni.append(kopija_gena)
                
                potomci.append(Posameznik(geni = geni))
                stevilo_ustvarjenih += 1

        #odrezemo odvecne ustvarjene
        potomci = potomci[:self.stevilo_potomcev]

        """for p in potomci:
            p.izpisi()"""

        return potomci

class Utezeno_globalno_krizanje(Krizanje):
    def krizaj(self, mnozica_za_izbiro_starsev: List["Ocenjen_posemaznik"]) -> List["Posameznik"]:
        potomci = []
        stevilo_ustvarjenih = 0

        while (stevilo_ustvarjenih<self.stevilo_potomcev):
            geni = []

            #print("nov")

            #za vsak gen izberemo nova dva starsa
            for i in range(len(mnozica_za_izbiro_starsev[0].geni)):
                starsa = random.sample(mnozica_za_izbiro_starsev, 2)

                """#TODO
                print(i)
                starsa[0].izpisi()
                starsa[1].izpisi()"""

                ocena1 = starsa[0].ocena
                ocena2 = starsa[1].ocena

                if(ocena1 < 0 and ocena2 < 0):
                    ocena1 = -(1/ocena1)
                    ocena2 = -(1/ocena2)

                if (ocena1 < 0):
                    ocena1 = 0
                if (ocena2 < 0):
                    ocena2 = 0

                if (ocena1 == 0 and ocena2 == 0):
                    utez=1/2
                else:
                    utez = ocena2/(ocena1+ocena2)

                """#TODO
                print(utez)"""

                tip = starsa[0].geni[i].tip
                if (isinstance(tip, Interval)):
                    vrednost1 = starsa[0].geni[i].vrednost
                    vrednost2 = starsa[1].geni[i].vrednost

                    vrednost = vrednost1 + utez * (vrednost2 - vrednost1)
                    nov_gen = Gen(vrednost = vrednost, tip=tip)
                    geni.append(nov_gen)
                elif(isinstance(tip, Urejena_mnozica)):
                    indeks1 = starsa[0].geni[i].indeks
                    indeks2 = starsa[1].geni[i].indeks

                    indeks = int(round(indeks1 + utez * (indeks2 - indeks1)))
                    nov_gen = Gen(vrednost = tip.vrednosti[indeks], tip=tip)
                    nov_gen.indeks = indeks
                    geni.append(nov_gen)
                elif(isinstance(tip, Mnozica)):
                    if (random.random() > utez):
                        vrednost = starsa[0].geni[i].vrednost
                    else:
                        vrednost = starsa[1].geni[i].vrednost
                    
                    nov_gen = Gen(vrednost = vrednost, tip=tip)
                    nov_gen.indeks = tip.vrednosti.index(vrednost)
                    geni.append(nov_gen)
            p = Posameznik(geni = geni)

            """#TODO
            p.izpisi()"""

            potomci.append(p)
            stevilo_ustvarjenih += 1

        return potomci 

class Utezeno_diskretno_globalno_krizanje(Krizanje):
    def krizaj(self, mnozica_za_izbiro_starsev: List["Ocenjen_posemaznik"]) -> List["Posameznik"]:
        potomci = []
        stevilo_ustvarjenih = 0

        while (stevilo_ustvarjenih<self.stevilo_potomcev):
            geni = []

            #za vsak gen izberemo nova dva starsa
            for i in range(len(mnozica_za_izbiro_starsev[0].geni)):
                starsa = random.sample(mnozica_za_izbiro_starsev, 2)

                ocena1 = starsa[0].ocena
                ocena2 = starsa[1].ocena
                utez = ocena2/(ocena1+ocena2)

                tip = starsa[0].geni[i].tip
                if (isinstance(tip, Interval)):
                    vrednost1 = starsa[0].geni[i].vrednost
                    vrednost2 = starsa[1].geni[i].vrednost

                    vrednost = vrednost1 + utez * (vrednost2 - vrednost1)
                    nov_gen = Gen(vrednost = vrednost, tip=tip)
                    geni.append(nov_gen)
                elif(isinstance(tip, Mnozica) or isinstance(tip, Urejena_mnozica)):
                    if (random.random() > utez):
                        vrednost = starsa[0].geni[i].vrednost
                    else:
                        vrednost = starsa[1].geni[i].vrednost
                    
                    nov_gen = Gen(vrednost = vrednost, tip=tip)
                    nov_gen.indeks = tip.vrednosti.index(vrednost)
                    geni.append(nov_gen)
            potomci.append(Posameznik(geni = geni))
            stevilo_ustvarjenih += 1
        
        return potomci 

class Globalno_krizanje(Krizanje):
    def krizaj(self, mnozica_za_izbiro_starsev: List["Ocenjen_posemaznik"]) -> List["Posameznik"]:
        potomci = []
        stevilo_ustvarjenih = 0

        while (stevilo_ustvarjenih<self.stevilo_potomcev):
            geni = []

            #za vsak gen izberemo nova dva starsa
            for i in range(len(mnozica_za_izbiro_starsev[0].geni)):
                starsa = random.sample(mnozica_za_izbiro_starsev, 2)

                tip = starsa[0].geni[i].tip
                if (isinstance(tip, Interval)):
                    vrednost1 = starsa[0].geni[i].vrednost
                    vrednost2 = starsa[1].geni[i].vrednost

                    vrednost = vrednost1 + 0.5 * (vrednost2 - vrednost1)
                    nov_gen = Gen(vrednost = vrednost, tip=tip)
                    geni.append(nov_gen)
                elif(isinstance(tip, Urejena_mnozica)):
                    indeks1 = starsa[0].geni[i].indeks
                    indeks2 = starsa[1].geni[i].indeks

                    indeks = int(round(indeks1 + 0.5 * (indeks2 - indeks1)))
                    nov_gen = Gen(vrednost = tip.vrednosti[indeks], tip=tip)
                    nov_gen.indeks = indeks
                    geni.append(nov_gen)
                elif(isinstance(tip, Mnozica)):
                    if (random.random() > 0.5):
                        vrednost = starsa[0].geni[i].vrednost
                    else:
                        vrednost = starsa[1].geni[i].vrednost
                    
                    nov_gen = Gen(vrednost = vrednost, tip=tip)
                    nov_gen.indeks = tip.vrednosti.index(vrednost)
                    geni.append(nov_gen)
            potomci.append(Posameznik(geni = geni))
            stevilo_ustvarjenih += 1
        
        return potomci 