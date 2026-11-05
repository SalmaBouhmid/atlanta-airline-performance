"""La conclusion de chaque compagnie est tirée des résultats, sans rien inventer."""

from conclusions import conclusions_par_compagnie
from results import charger_resultats


def test_une_conclusion_par_compagnie_dans_l_ordre_du_classement():
    resultats = charger_resultats()
    conclusions = conclusions_par_compagnie(resultats)
    assert len(conclusions) == 14
    rangs = sorted(conclusions.values(), key=lambda v: v["rang"])
    assert rangs[0]["pct_retard_ou_annule"] == min(c["pct_retard_ou_annule"] for c in resultats["compagnies"].values())
    assert [v["rang"] for v in rangs] == list(range(1, 15))
    assert conclusions["Endeavor (Delta Connection)"]["rang"] == 1
    assert conclusions["JetBlue"]["position"] == "parmi les moins ponctuelles"
