"""Tests des statistiques descriptives, de la comparaison des compagnies et de resultats.json.

Les formules sont vérifiées sur de petits exemples construits à la main, dont on connaît la réponse.
"""

import pandas as pd
import pytest

from analysis import marge_erreur_95, statistiques_descriptives
from comparison import (comparaison_brut_standardise, correlation_globale_et_par_groupe,
                        effet_domino_par_compagnie, profil_compagnies, taux_standardise)
from results import calculer_resultats


def mini_vols(lignes):
    """Petit dataset de test : (compagnie, tranche, en_retard)."""
    vols = pd.DataFrame(lignes, columns=["compagnie", "tranche_horaire", "DepDel15"])
    vols["Cancelled"] = False
    return vols


def test_standardisation_sans_effet_si_meme_composition():
    """Deux compagnies avec la même répartition horaire : brut = standardisé."""
    vols = mini_vols([("A", "Matin", 0), ("A", "Soir", 1), ("B", "Matin", 0), ("B", "Soir", 0)])
    tableau = taux_standardise(vols)
    assert tableau.loc["A", "taux_standardise_pct"] == pytest.approx(50)
    assert tableau.loc["B", "taux_standardise_pct"] == pytest.approx(0)


def test_standardisation_corrige_un_effet_de_composition():
    """A et B ont exactement le même taux à chaque tranche, mais A vole surtout le soir.

    Brut : A paraît plus en retard. À tranche égale : A et B sont identiques.
    """
    a = [("A", "Matin", 0)] * 1 + [("A", "Soir", 1)] * 6 + [("A", "Soir", 0)] * 3
    b = [("B", "Matin", 0)] * 9 + [("B", "Soir", 1)] * 2 + [("B", "Soir", 0)] * 1
    tableau = comparaison_brut_standardise(mini_vols(a + b))
    assert tableau.loc["A", "pct_retard_brut"] > tableau.loc["B", "pct_retard_brut"]
    assert tableau.loc["A", "taux_standardise_pct"] == pytest.approx(tableau.loc["B", "taux_standardise_pct"])


def test_correlation_masquee_par_le_melange_des_groupes():
    """Dans chaque groupe, x et y montent ensemble ; mélangés, la relation s'inverse (type Simpson)."""
    vols = pd.DataFrame({
        "compagnie": ["A"] * 3 + ["B"] * 3,
        "x": [1, 2, 3, 11, 12, 13],
        "y": [11, 12, 13, 1, 2, 3],
        "Cancelled": False,
    })
    globale, par_groupe = correlation_globale_et_par_groupe(vols, "x", "y")
    assert globale < 0
    assert (par_groupe["correlation"] == 1).all()


def test_marge_erreur():
    """p = 50 %, n = 100 : marge ≈ 1,96 × 5 = 9,8 points."""
    assert marge_erreur_95(0.5, 100) == pytest.approx(9.8)


def test_statistiques_descriptives_coherentes(vols):
    tableau = statistiques_descriptives(vols)
    retard = tableau.loc["Retard au départ (min)"]
    assert retard["médiane"] == pytest.approx(vols["DepDelay"].median(), abs=0.1)
    assert retard["Q1"] <= retard["médiane"] <= retard["Q3"]
    assert retard["moyenne - médiane"] > 0  # asymétrie à droite : quelques très gros retards


def test_profil_compagnies(vols):
    profil = profil_compagnies(vols)
    assert profil["nb_vols"].sum() == len(vols)
    assert profil["part_vols_pct"].sum() == pytest.approx(100, abs=0.1)
    assert (profil["pct_retard_ou_annule"] >= profil["pct_retard"] - 0.1).all()


def test_domino_par_compagnie_ignore_les_petits_groupes(vols):
    assert (effet_domino_par_compagnie(vols)["n_vol_precedent_en_retard"] >= 30).all()


def test_resultats_complets_et_coherents(vols):
    resultats = calculer_resultats(vols)
    assert resultats["population"]["nb_vols"] == 319335
    assert resultats["population"]["nb_jours"] == 365
    assert resultats["zoom_ete_2022"]["pct_retard"] == pytest.approx(26.2)
    assert len(resultats["retard_par_mois_pct"]) == 12
