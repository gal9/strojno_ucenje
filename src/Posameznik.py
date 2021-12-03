from typing import List, Dict, Any

from src.Gen import Gen
class Posameznik():
    geni: List[Gen]

    def __init__(self, geni: List[Gen]) -> None:
        self.geni = geni

    def v_slovar(self) -> Dict[str, Any]:
        slovar = {}
        for g in self.geni:
            slovar[g.tip.ime] = g.vrednost
        
        return slovar

    def izpisi(self) -> None:
        print("{", end="")
        #izpisi vse razen zadnjega
        for g in self.geni[:-1]:
            print(g.tip.ime, ": ", g.vrednost, end=", ")
        #izpisi se zadnjega
        print(self.geni[-1].tip.ime, ": ",  self.geni[-1].vrednost, end="}\n")

    def v_niz(self):
        niz = ""
        for g in self.geni[:-1]:
            if(hasattr(g, 'indeks')):
                niz = niz + str(g.indeks) + "|"
            else:
                niz = niz + str(g.vrednost) + "|"
        if(hasattr(self.geni[-1], 'indeks')):
            niz = niz + str(self.geni[-1].indeks)
        else:
            niz = niz + str(self.geni[-1].vrednost)
        #print(niz)
        return niz

class Ocenjen_posameznik():
    geni: List[Gen]
    ocena: float

    def __init__(self, geni: List[Gen], ocena: float) -> None:
        self.geni = geni
        self.ocena = ocena

    def v_slovar(self) -> Dict[str, Any]:
        slovar = {}
        for g in self.geni:
            slovar[g.tip.ime] = g.vrednost
        
        return slovar

    def posodobi_oceno(self, nova_ocena: float) -> None:
        self.ocena = nova_ocena

    def izpisi(self) -> None:
        print("{", end="")
        #izpisi vse razen zadnjega
        for g in self.geni[:-1]:
            print(g.tip.ime, ": ", g.vrednost, end=", ")
        #izpisi se zadnjega
        print(self.geni[-1].tip.ime, ": ", self.geni[-1].vrednost, end=" => ocena: ")
        print(self.ocena, "}")

    def v_niz(self):
        niz = ""
        for g in self.geni[:-1]:
            if(hasattr(g, 'indeks')):
                niz = niz + str(g.indeks) + "|"
            else:
                niz = niz + str(g.vrednost) + "|"
        if(hasattr(self.geni[-1], 'indeks')):
            niz = niz + str(self.geni[-1].indeks)
        else:
            niz = niz + str(self.geni[-1].vrednost)
        #print(niz)
        return niz