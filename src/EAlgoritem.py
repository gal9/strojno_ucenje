import time
from typing import Callable, Any, List, Optional, Dict
from statistics import mean

from src.Gen import Gen, Gen_tip
from src.Krizanje import Krizanje, Diagonalno_krizanje
from src.Mutacija import Mutacija, Mutacija1
from src.Selekcija_starsev import Selekcija_starsev, Oceni_proporcionalna_selekcija_starsev, Nakljucna_selekcija_starsev, \
    Oceni_proporcionalna_selekcija_starsev_brez_ponovitev, Rangu_lin_proporcionalna_selekcija_starsev_brez_ponovitev, \
    Oceni_proporcionalna_selekcija_starsev_brez_ponovitev_sigma_skalirana, Rangu_exp_proporcionalna_selekcija_starsev_brez_ponovitev
from src.Selekcija_napredovanja import Selekcija_najboljsih, Selekcija_napredovanja, Ranku_proporcionalna_selekcija
from src.Posameznik import Posameznik, Ocenjen_posameznik

#TODO: logiranje


class EAlgoritem():
    spomin = Dict[str, List[float]]
    generacija: int
    stevilo_ocenjevanj: int
    stevilo_dodatnih_ocenjevanj: int
    cas_ocenjevanja: float
    populacija: List["Ocenjen_posameznik"]
    najboljse_ocene: List[float]
    najboljsi: "Ocenjen_posameznik"

    cenilna_funkcija: Callable[["Posameznik"], float]
    tipi_genov: List[Gen_tip]
    omejitev_generacij: int
    omejitev_stevila_ocenjevanj: int
    min_rezultat: float
    stevilo_potomcev: int
    stevilo_starsev_za_izbiro: int
    verjetnost_mutacije: float
    velikost_populacije: int
    krizanje: "Krizanje"
    mutacija: "Mutacija"
    selekcija_starsev: "Selekcija_starsev"
    selekcija_napredovanja: "Selekcija_napredovanja"
    zacetna_populacija: List[Dict[str, Any]]    
    interval_dodatnega_testiranja: int

    def __init__(self, 
                cenilna_funkcija: Callable[[Dict[str, Any]], float],
                tipi_genov: List["Gen_tip"],
                omejitev_generacij: Optional[int] = None,
                omejitev_stevila_ocenjevanj: Optional[int] = None,
                min_rezultat: Optional[float] = None,
                stevilo_potomcev: Optional[int] = None,
                stevilo_starsev_za_izbiro: Optional[int] = None,
                verjetnost_mutacije: Optional[float] = None,
                velikost_populacije: Optional[int] = 10,
                krizanje: Optional["Krizanje"] = Diagonalno_krizanje(stevilo_starsev = 2),
                mutacija: Optional["Mutacija"] = Mutacija1(),
                selekcija_starsev: Optional["Selekcija_starsev"] = Rangu_lin_proporcionalna_selekcija_starsev_brez_ponovitev(),
                selekcija_napredovanja: Optional["Selekcija_napredovanja"] = Ranku_proporcionalna_selekcija(),
                zacetna_populacija: Optional[List[Dict[str, Any]]] = [],
                interval_dodatnega_testiranja: Optional[int] = None) -> None:
                
        self.velikost_populacije = velikost_populacije
        self.omejitev_generacij = omejitev_generacij
        self.omejitev_stevila_ocenjevanj = omejitev_stevila_ocenjevanj
        self.min_rezultat = min_rezultat
        self.cenilna_funkcija = cenilna_funkcija
        self.tipi_genov = tipi_genov        
        self.mutacija = mutacija
        self.krizanje = krizanje
        self.selekcija_napredovanja = selekcija_napredovanja
        self.selekcija_starsev = selekcija_starsev
        self.interval_dodatnega_testiranja = interval_dodatnega_testiranja

        self.cas_ocenjevanja = 0
        self.spomin = {}

        self.zacetna_populacija = zacetna_populacija
        assert len(self.zacetna_populacija) <= velikost_populacije, "Velikost zacetne populacije naj ne bo vecja od velikosti populacije."

        self.stevilo_starsev_za_izbiro = int(velikost_populacije/2) if stevilo_starsev_za_izbiro is None else stevilo_starsev_za_izbiro
        assert self.stevilo_starsev_za_izbiro <= self.velikost_populacije, "stevilo_starsev_za_izbiro ne sme presegati velikost populacije"

        self.stevilo_potomcev = stevilo_starsev_za_izbiro if stevilo_potomcev is None else stevilo_potomcev

        self.verjetnost_mutacije = (2/(3*self.stevilo_potomcev)) if verjetnost_mutacije is None else verjetnost_mutacije

        #USKLAJEVANJE PARAMETROV     
        self.selekcija_napredovanja.velikost_populacije = self.velikost_populacije
        self.selekcija_starsev.stevilo_starsev_za_izbiro = self.stevilo_starsev_za_izbiro
        self.krizanje.stevilo_potomcev = self.stevilo_potomcev  
        self.mutacija.verjetnost_mutacije = self.verjetnost_mutacije      


    def izpis_populacije(self, populacija: List[Any]) -> None:
        for p in populacija:
            p.izpisi()

    def nadaljuj(self) -> bool:
        if(self.generacija == 0):
            return True

        #preveri generacije
        if ((self.omejitev_generacij is not None) and self.generacija > self.omejitev_generacij):
            return False

        #preveri stevilo ocenjevanj
        if ((self.omejitev_stevila_ocenjevanj is not None) and self.stevilo_ocenjevanj >= self.omejitev_stevila_ocenjevanj):
            return False

        #preveri najboljsi rezultat
        if ((self.min_rezultat is not None) and self.najboljsi.ocena >= self.min_rezultat):
            return False

        return True
    
    def oceni(self, posamezniki: List["Posameznik"], zacetna: bool) -> List["Ocenjen_posameznik"]:
        ocenjeni = []
        for p in posamezniki:
            #ce smo dosegli omejitev ocenjevanj ali minimalni rezultat se ustavi
            if (self.nadaljuj()):
                #pretvori posameznika v slovar
                s = p.v_slovar()
                n = p.v_niz()

                #oceni ga samo ce se ni bil ocenjen
                if (n not in self.spomin):
                    zacetek = time.time()
                    o = self.cenilna_funkcija(s)
                    konec = time.time()

                    self.cas_ocenjevanja += (konec-zacetek)
                    self.stevilo_ocenjevanj += 1

                    #ocenjenega doda v slovar
                    self.spomin[n] = [o] 
                    op = Ocenjen_posameznik(p.geni, o)

                    
                    if(not zacetna):
                        #med izgradnjo prve generacije tega ne preverjamo

                        #preveri ce je zadnji testirani boljsi od trenutnega najboljsega
                        if(self.najboljsi.ocena < o):
                            self.najboljsi = op

                        #dodamo novo najboljso oceno
                        self.najboljse_ocene.append(self.najboljsi.ocena)      
                else:
                    #potegnemo ocene iz spomina in jih povprečimo
                    o = mean(self.spomin.get(n))
                    op = Ocenjen_posameznik(p.geni, o)

                
                ocenjeni.append(op)  
       
            else:
                break
        
        return ocenjeni

    def testiranje_prvega(self) -> None:
        ocene_prvega = self.spomin.get(self.najboljsi.v_niz())
        stevilo_potrebnih_testiranj = int((self.generacija/self.interval_dodatnega_testiranja)+1)
        stevilo_testiranj_prvega = len(ocene_prvega)
        #print(ocene_prvega)
        #print(stevilo_potrebnih_testiranj)

        while(stevilo_testiranj_prvega < stevilo_potrebnih_testiranj):
            manjkajoca = stevilo_potrebnih_testiranj - stevilo_testiranj_prvega
            n = self.najboljsi.v_niz()
            s = self.najboljsi.v_slovar()

            for i in range(manjkajoca):
                zacetek = time.time()
                o = self.cenilna_funkcija(s)
                konec = time.time()

                self.cas_ocenjevanja += (konec-zacetek)
                self.stevilo_ocenjevanj += 1
                self.stevilo_dodatnih_ocenjevanj += 1

                self.spomin.get(n).append(o)
            
            self.najboljsi.posodobi_oceno(mean(self.spomin[n]))
            #posodobi ocene morebitnim duplikatom
            for p in self.populacija:
                p.posodobi_oceno(mean(self.spomin[p.v_niz()]))  
            #print(self.najboljsi.ocena)
            self.populacija.sort(key=lambda p: p.ocena, reverse=True)            

            #print(self.najboljsi.ocena)
            ocene_prvega = self.spomin.get(self.najboljsi.v_niz())
            stevilo_testiranj_prvega = len(ocene_prvega)
            #print(ocene_prvega)        

    def pripravi(self) -> None:
        self.generacija = 0
        self.stevilo_ocenjevanj = 0
        self.stevilo_dodatnih_ocenjevanj = 0
        self.najboljse_ocene = []

        #PRIPRAVA POPULACIJE
        neocenjeni=[]
        #iz zacetna_populacija naredi posameznike
        for p in self.zacetna_populacija:
            zbrani_geni = []
            for t in self.tipi_genov:
                vrednost = p.get(t.ime)
                g = Gen(tip=t, vrednost=vrednost)
                zbrani_geni.append(g)
            neocenjeni.append(Posameznik(geni=zbrani_geni))

        manjkajoci = self.velikost_populacije - len(neocenjeni)

        #manjkajoce posameznike doloci nakljucno
        for i in range(manjkajoci):
            zbrani_geni = []
            for t in self.tipi_genov:
                g = Gen(tip=t)
                zbrani_geni.append(g)
            neocenjeni.append(Posameznik(geni = zbrani_geni))

        self.populacija = self.oceni(neocenjeni, zacetna=True)
        #Uredi populacijo glede na oceno
        self.populacija.sort(key=lambda p: p.ocena, reverse=True)

        self.najboljsi = self.populacija[0]
        self.najboljse_ocene.append(self.najboljsi.ocena)

        self.generacija = 0


    def pozeni(self) -> None:
        self.pripravi()

        print("zacetna")
        self.izpis_populacije(self.populacija)
       
        while(self.nadaljuj()):
            #izbere mnozico starsev
            starsi = self.selekcija_starsev.izberi(populacija = self.populacija)
            
            #print("starsi")
            #self.izpis_populacije(starsi)

            #izvede krizanje
            potomci = self.krizanje.krizaj(mnozica_za_izbiro_starsev = starsi)
            #print("potomci")
            #self.izpis_populacije(potomci)

            potomci = self.mutacija.mutiraj(potomci)
            #print("po")
            #self.izpis_populacije(potomci)

            #oceni potomce
            ocenjeni_potomci = self.oceni(potomci, zacetna=False)

            #razširi populacijo in jo uredi glede na oceno
            self.populacija = self.populacija + ocenjeni_potomci
            self.populacija.sort(key=lambda p: p.ocena, reverse=True)

            #print("celotna")
            #self.izpis_populacije(self.populacija)

            if(self.interval_dodatnega_testiranja is not None):
                self.testiranje_prvega()

            #izberi populacijo za napredovanje
            self.populacija = self.selekcija_napredovanja.izberi(self.populacija)

            self.generacija += 1

            print("generacija", self.generacija)
            self.izpis_populacije(self.populacija)

            #print(self.stevilo_ocenjevanj)