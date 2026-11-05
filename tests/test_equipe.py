"""Tests du modèle (data scientist) et des analyses (data analyst), lus dans la base SQLite."""

import pytest

from results import charger_resultats


@pytest.fixture(scope="module")
def equipe():
    resultats = charger_resultats().get("equipe")
    if resultats is None:
        pytest.skip("base SQLite absente : lancer python pipeline.py")
    return resultats


def test_modele_sens_des_effets(equipe):
    """Les effets attendus vont dans le bon sens (à conditions égales)."""
    rc = equipe["modele"]["rapports_de_cotes"]
    assert rc["Heure : Soir"]["rapport_de_cotes"] > 2
    assert rc["Vol précédent : vol précédent en retard"]["rapport_de_cotes"] > 2
    assert rc["Compagnie : Endeavor (Delta Connection)"]["ic_haut"] < 1
    assert rc["Pluie à l'heure du départ"]["ic_bas"] > 1


def test_modele_honnete_sur_sa_precision(equipe):
    """Le modèle explique une partie seulement du retard : on l'affiche, on ne le cache pas."""
    assert 0.6 < equipe["modele"]["auc"] < 0.8
    assert max(equipe["modele"]["importance_pseudo_r2"], key=equipe["modele"]["importance_pseudo_r2"].get) == "Heure de départ"


def test_meteo_le_retard_augmente_avec_la_pluie(equipe):
    classes = list(equipe["meteo_par_classe"].values())
    assert classes[0]["pct_retard"] < classes[-1]["pct_retard"]


def test_jours_non_expliques_ont_une_meteo_anormale(equipe):
    mediane = equipe["meteo_mediane"]
    for jour in equipe["jours_non_expliques_meteo"]:
        assert jour["heures_de_pluie"] > 2 * mediane["heures_de_pluie"] or jour["rafales_max_kmh"] > 2 * mediane["rafales_max_kmh"]


def test_compagnies_sourcees(equipe):
    assert len(equipe["compagnies_contexte"]) == 14
